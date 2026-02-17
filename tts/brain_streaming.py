"""
Renan VTuber Brain — Real token streaming + real-time TTS architecture
=====================================================================

Goals:
- True LLM token streaming via LangGraph astream_events(v2)
- Low-latency speech: chunk early, speak while generating
- Barge-in cancellation: new question stops current speech immediately
- Pluggable TTS backend:
    * Recommended: RealtimeTTS (designed for text-stream -> immediate audio)
    * Fallback: EdgeTTS engine (works, but true streaming playback is trickier)

Notes:
- Do NOT use asyncio.run() inside library code; keep everything async.
- Keep sync wrappers only for CLI entry points.
"""

import os
import asyncio
from dataclasses import dataclass
from typing import Optional, AsyncIterator, List, Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# ──────────────────────────────────────────────────────────────────────────────
# Config / env
# ──────────────────────────────────────────────────────────────────────────────

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("GOOGLE_API_KEY missing")

DB_PATH = "../db_clone"
PROMPT_PATH = "/prompt_clone_tts.txt"

if os.path.exists(PROMPT_PATH):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        MASTER_PROMPT = f.read()
else:
    MASTER_PROMPT = "Você é Renan Santos, um analista político brasileiro."


# ──────────────────────────────────────────────────────────────────────────────
# RAG tool
# ──────────────────────────────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True},
)

if os.path.exists(DB_PATH):
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.4},
    )
else:
    retriever = None


@tool
def pesquisar_memoria_renan(query: str) -> str:
    """Busca trechos de lives e pensamentos do Renan Santos sobre um tema."""
    if not retriever:
        return "RAG não disponível"
    docs = retriever.invoke(query)
    parts = []
    for i, doc in enumerate(docs, 1):
        trecho = doc.page_content[:250]
        fonte = doc.metadata.get("fonte", "Fonte desconhecida")
        fonte_limpa = fonte.replace(".pt.srt", "").replace(".srt", "")
        parts.append(f"Fonte {i} ({fonte_limpa}): {trecho}...")
    return "\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True,
).bind_tools([pesquisar_memoria_renan])


# ──────────────────────────────────────────────────────────────────────────────
# Real-time chunker (token -> speechable segments)
# ──────────────────────────────────────────────────────────────────────────────

class TokenChunker:
    """
    Converts token stream into short speakable chunks.
    Long-run behavior:
    - Very small first chunk to start voice fast
    - Then prefer sentence-ish chunks to reduce TTS calls

    This outputs strings, already suitable to send to TTS.
    """

    def __init__(
        self,
        min_first_chunk_chars: int = 24,
        min_chunk_chars: int = 45,
        max_chunk_chars: int = 220,
    ):
        self.min_first_chunk_chars = min_first_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self._buf = ""
        self._did_first_flush = False

    def reset(self):
        self._buf = ""
        self._did_first_flush = False

    def push(self, token: str) -> List[str]:
        out: List[str] = []
        if not token:
            return out

        self._buf += token

        # hard cap: if buffer gets too large, flush
        if len(self._buf) >= self.max_chunk_chars:
            out.append(self._flush())
            return out

        # early first chunk (perceived latency)
        if not self._did_first_flush and len(self._buf) >= self.min_first_chunk_chars:
            # avoid flushing mid-word if possible
            if self._buf.endswith(" ") or self._buf.endswith(","):
                out.append(self._flush(first=True))
                return out

        # sentence-ish boundaries
        if any(self._buf.rstrip().endswith(p) for p in [".", "!", "?", "…", "\n"]):
            if len(self._buf.strip()) >= (self.min_chunk_chars if self._did_first_flush else self.min_first_chunk_chars):
                out.append(self._flush())
                return out

        return out

    def finish(self) -> List[str]:
        if self._buf.strip():
            return [self._flush()]
        return []

    def _flush(self, first: bool = False) -> str:
        txt = self._buf.strip()
        self._buf = ""
        if first:
            self._did_first_flush = True
        else:
            self._did_first_flush = True
        return txt


# ──────────────────────────────────────────────────────────────────────────────
# Cancellable TTS manager (session-aware)
# ──────────────────────────────────────────────────────────────────────────────

class TTSSessionCancelled(Exception):
    pass


class AsyncSpeaker:
    """
    Session-aware speaker:
    - enqueue(text) speaks sequentially
    - cancel() stops current and clears queue
    - new session id invalidates old work immediately

    Plug in any backend with: await backend.speak(text, session_id)
    """

    def __init__(self, backend):
        self.backend = backend
        self._session_id = 0
        self._q: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    @property
    def session_id(self) -> int:
        return self._session_id

    async def start(self):
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def new_session(self) -> int:
        async with self._lock:
            self._session_id += 1
            # clear queue immediately
            self._clear_queue()
            # tell backend to cancel any ongoing playback
            await self.backend.cancel(self._session_id)
            return self._session_id

    async def enqueue(self, text: str, session_id: int):
        if not text.strip():
            return
        if session_id != self._session_id:
            return
        await self._q.put(text)

    async def wait_idle(self, session_id: int):
        # waits until queue drained AND backend is idle
        while session_id == self._session_id:
            if self._q.empty() and await self.backend.is_idle(session_id):
                return
            await asyncio.sleep(0.03)

    def _clear_queue(self):
        try:
            while True:
                self._q.get_nowait()
        except asyncio.QueueEmpty:
            pass

    async def _worker(self):
        while True:
            text = await self._q.get()
            sid = self._session_id
            if sid != self._session_id:
                continue
            try:
                await self.backend.speak(text, sid)
            except TTSSessionCancelled:
                continue
            except Exception as e:
                print(f"[TTS worker] error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# RealtimeTTS backend (recommended long-run)
# ──────────────────────────────────────────────────────────────────────────────

class RealtimeTTSBackend:
    """
    Long-run best: use RealtimeTTS (text stream -> immediate audio).
    You can pick an engine inside RealtimeTTS (OpenAI/Azure/ElevenLabs/etc).
    For MVP you can still use Edge-like voices elsewhere, but the key is:
    - the library is built around *streaming playback*.
    """

    def __init__(self, output_device_name: Optional[str] = None):
        self.output_device_name = output_device_name
        self._current_sid = 0
        self._playing = False

        # Lazy import so project still runs without it
        # pip install realtimetts
        from RealtimeTTS import TextToAudioStream, SystemEngine  # type: ignore

        self.TextToAudioStream = TextToAudioStream
        self.engine = SystemEngine()  # change to OpenAIEngine/AzureEngine/etc
        self.stream = self.TextToAudioStream(self.engine)

    async def speak(self, text: str, session_id: int):
        if session_id != self._current_sid:
            # session changed before we started
            raise TTSSessionCancelled()

        self._playing = True
        try:
            # Feed text and start playback immediately.
            # RealtimeTTS handles buffering & audio output.
            self.stream.feed(text)
            self.stream.play_async()  # plays in background thread
            # We don’t block fully here; keep it short so barge-in is responsive.
            # Wait a bit so audio starts, then return control.
            await asyncio.sleep(0.01)
        finally:
            # Mark idle when the stream finished current buffer
            self._playing = False

    async def cancel(self, session_id: int):
        self._current_sid = session_id
        # Stop any ongoing playback & clear buffers
        try:
            self.stream.stop()
        except Exception:
            pass
        self._playing = False

    async def is_idle(self, session_id: int) -> bool:
        if session_id != self._current_sid:
            return True
        return not self._playing


# ──────────────────────────────────────────────────────────────────────────────
# Brain
# ──────────────────────────────────────────────────────────────────────────────

class RenanBrain:
    def __init__(self, speaker: Optional[AsyncSpeaker] = None):
        self.speaker = speaker
        self.chunker = TokenChunker()

        self._build_graph()

    def _chatbot_node(self, state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=MASTER_PROMPT)] + messages
        response = llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        wf = StateGraph(MessagesState)
        wf.add_node("chatbot", self._chatbot_node)
        wf.add_node("tools", ToolNode([pesquisar_memoria_renan]))
        wf.add_edge(START, "chatbot")
        wf.add_conditional_edges("chatbot", tools_condition)
        wf.add_edge("tools", "chatbot")
        self.agent = wf.compile()

    async def answer_streaming(
        self,
        user_message: str,
        history: List,
        speak: bool = True,
    ) -> str:
        """
        Core: token streaming via astream_events(v2).
        - Ignores tool-call content
        - Emits tokens as they arrive
        - Chunker feeds speaker in near real-time
        """

        # New speech session (barge-in)
        sid = None
        if speak and self.speaker:
            await self.speaker.start()
            sid = await self.speaker.new_session()
            self.chunker.reset()

        messages = list(history)
        messages.append(HumanMessage(content=user_message))
        inputs = {"messages": messages}

        full = ""
        in_tool = False

        async for event in self.agent.astream_events(inputs, version="v2"):
            et = event.get("event", "")

            if et == "on_tool_start":
                in_tool = True
                continue

            if et == "on_tool_end":
                in_tool = False
                continue

            if et == "on_chat_model_stream":
                if in_tool:
                    continue

                chunk = event.get("data", {}).get("chunk")
                if not chunk:
                    continue

                # LangChain tokens can be in .content or .text depending on model wrapper.
                token = getattr(chunk, "content", None) or getattr(chunk, "text", None)
                if not token:
                    continue

                # If this chunk contains tool call blocks, ignore them
                if getattr(chunk, "tool_call_chunks", None):
                    continue

                full += token

                # Speak ASAP
                if sid is not None and self.speaker:
                    for piece in self.chunker.push(token):
                        await self.speaker.enqueue(piece, sid)

        # Flush remainder
        if sid is not None and self.speaker:
            for piece in self.chunker.finish():
                await self.speaker.enqueue(piece, sid)
            await self.speaker.wait_idle(sid)

        return full


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry (safe sync wrapper)
# ──────────────────────────────────────────────────────────────────────────────

async def cli():
    # Choose backend (recommended long-run)
    tts_backend = RealtimeTTSBackend(output_device_name="CABLE Input")
    speaker = AsyncSpeaker(tts_backend)
    brain = RenanBrain(speaker=speaker)

    history: List = []
    print("🧠 RENAN — streaming real-time (type 'sair')")

    while True:
        text = input("👤 Você: ").strip()
        if not text:
            continue
        if text.lower() in ("sair", "exit", "quit"):
            break

        resp = await brain.answer_streaming(text, history, speak=True)
        history.append(HumanMessage(content=text))
        history.append(AIMessage(content=resp))
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    asyncio.run(cli())
