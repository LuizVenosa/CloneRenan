import os
from pathlib import Path
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from core.speaker_manager import SpeakerManager
from speech.token_chunker import TokenChunker


class RenanBrain:
    def __init__(
        self,
        speaker: Optional[SpeakerManager] = None,
        db_path: str = "./db_clone",
        prompt_path: str = "./prompt_clone.txt",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
    ):
        load_dotenv()
        self.speaker = speaker
        self.chunker = TokenChunker()

        self.master_prompt = self._load_prompt(prompt_path)
        self.retriever = self._build_retriever(db_path)
        self.tool = self._build_tool()

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            streaming=True,
        ).bind_tools([self.tool])

        self.agent = self._build_graph()

    @staticmethod
    def _load_prompt(prompt_path: str) -> str:
        p = Path(prompt_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return "Você é Renan Santos, um analista político brasileiro."

    @staticmethod
    def _build_retriever(db_path: str):
        if not Path(db_path).exists():
            return None
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        return vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.4},
        )

    def _build_tool(self):
        retriever = self.retriever

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

        return pesquisar_memoria_renan

    def _chatbot_node(self, state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.master_prompt)] + messages
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        workflow = StateGraph(MessagesState)
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.add_node("tools", ToolNode([self.tool]))
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", tools_condition)
        workflow.add_edge("tools", "chatbot")
        return workflow.compile()

    async def answer_streaming(
        self,
        user_message: str,
        history: List[Any],
        speak: bool = True,
        session_id: Optional[int] = None,
    ) -> str:
        sid: Optional[int] = None
        if speak and self.speaker:
            await self.speaker.start()
            sid = session_id if session_id is not None else await self.speaker.new_session()
            self.chunker.reset()

        messages = list(history)
        messages.append(HumanMessage(content=user_message))
        inputs = {"messages": messages}

        in_tool = False
        full_text = ""

        async for event in self.agent.astream_events(inputs, version="v2"):
            event_type = event.get("event", "")

            if event_type == "on_tool_start":
                in_tool = True
                continue

            if event_type == "on_tool_end":
                in_tool = False
                continue

            if event_type != "on_chat_model_stream" or in_tool:
                continue

            chunk = event.get("data", {}).get("chunk")
            token = self._extract_token(chunk)
            if not token:
                continue

            full_text += token
            if sid is not None and self.speaker:
                for piece in self.chunker.push(token):
                    await self.speaker.enqueue(piece, sid)

        if sid is not None and self.speaker:
            for piece in self.chunker.finish():
                await self.speaker.enqueue(piece, sid)
            await self.speaker.wait_idle(sid)

        return full_text

    @staticmethod
    def _extract_token(chunk: Any) -> str:
        if not chunk:
            return ""

        if getattr(chunk, "tool_call_chunks", None):
            return ""

        content = getattr(chunk, "content", None)
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            if parts:
                return "".join(parts)

        text = getattr(chunk, "text", None)
        if isinstance(text, str):
            return text

        return ""
