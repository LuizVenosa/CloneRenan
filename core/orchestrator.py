import asyncio
import os
import time
from typing import List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from brain.renan_brain import RenanBrain
from chat.twitch_adapter import ChatMessage, TwitchAdapter
from core.speaker_manager import SpeakerManager, SpeechChunk
from obs.obs_client import OBSClient
from tts.edge_backend import EdgeBackend
from tts.realtimetts_backend import RealtimeTTSBackend
from vts.vts_client import VTSClient


class Orchestrator:
    def __init__(
        self,
        brain: RenanBrain,
        speaker: SpeakerManager,
        input_queue: asyncio.Queue,
        vts: Optional[VTSClient] = None,
        obs: Optional[OBSClient] = None,
        caption_source: str = "RenanCaptions",
        scene_thinking: Optional[str] = None,
        scene_speaking: Optional[str] = None,
        print_local_response: bool = True,
    ):
        self.brain = brain
        self.speaker = speaker
        self.input_queue = input_queue
        self.vts = vts
        self.obs = obs
        self.caption_source = caption_source
        self.scene_thinking = scene_thinking
        self.scene_speaking = scene_speaking
        self.print_local_response = print_local_response

        self.history: List = []

    async def run(self) -> None:
        await self.speaker.start()
        while True:
            message: ChatMessage = await self.input_queue.get()
            sid = await self.speaker.new_session()

            await self._set_thinking_state()
            response = await self.brain.answer_streaming(
                user_message=message.text,
                history=self.history,
                speak=True,
                session_id=sid,
            )
            await self._set_idle_state()
            if self.print_local_response and message.user == "local":
                print(f"Renan: {response}", flush=True)

            self.history.append(HumanMessage(content=message.text))
            self.history.append(AIMessage(content=response))
            if len(self.history) > 20:
                self.history = self.history[-20:]

    async def on_chunk_enqueued(self, chunk: SpeechChunk) -> None:
        if self.vts:
            await self.vts.set_state("speaking")
        if self.scene_speaking and self.obs:
            self.obs.switch_scene(self.scene_speaking)

    async def on_chunk_spoken(self, chunk: SpeechChunk) -> None:
        if self.obs:
            self.obs.set_text_source(self.caption_source, chunk.text)

    async def _set_thinking_state(self) -> None:
        if self.vts:
            await self.vts.set_state("thinking")
        if self.scene_thinking and self.obs:
            self.obs.switch_scene(self.scene_thinking)

    async def _set_idle_state(self) -> None:
        if self.vts:
            await self.vts.set_state("idle")


def build_default_orchestrator() -> tuple[Orchestrator, Optional[TwitchAdapter]]:
    in_queue: asyncio.Queue = asyncio.Queue()

    use_realtime = os.getenv("USE_REALTIMETTS", "1") == "1"
    if use_realtime:
        backend = RealtimeTTSBackend(
            voice=os.getenv("RTTTS_VOICE", "").strip() or None,
            rate=os.getenv("RTTTS_RATE", "").strip() or None,
            pitch=os.getenv("RTTTS_PITCH", "").strip() or None,
        )
    else:
        output_device = os.getenv("TTS_OUTPUT_DEVICE", "CABLE Input").strip() or None
        backend = EdgeBackend(
            voice=os.getenv("EDGE_VOICE", "pt-BR-AntonioNeural"),
            rate=os.getenv("EDGE_RATE", "+10%"),
            pitch=os.getenv("EDGE_PITCH", "+0Hz"),
            output_device_name=output_device,
        )

    # Instantiate without callbacks first, then bind orchestrator-aware callbacks.
    speaker = SpeakerManager(backend=backend)
    brain = RenanBrain(speaker=speaker)

    obs_client = None
    if os.getenv("OBS_ENABLED", "0") == "1":
        obs_client = OBSClient(
            host=os.getenv("OBS_HOST", "127.0.0.1"),
            port=int(os.getenv("OBS_PORT", "4455")),
            password=os.getenv("OBS_PASSWORD", ""),
        )
        obs_client.connect()

    vts_client = None
    if os.getenv("VTS_ENABLED", "0") == "1":
        vts_client = VTSClient(
            plugin_name=os.getenv("VTS_PLUGIN_NAME", "RenanOrchestrator"),
            plugin_developer=os.getenv("VTS_PLUGIN_DEVELOPER", "Codex"),
        )

    orch = Orchestrator(
        brain=brain,
        speaker=speaker,
        input_queue=in_queue,
        vts=vts_client,
        obs=obs_client,
        caption_source=os.getenv("OBS_CAPTION_SOURCE", "RenanCaptions"),
        scene_thinking=os.getenv("OBS_SCENE_THINKING"),
        scene_speaking=os.getenv("OBS_SCENE_SPEAKING"),
        print_local_response=os.getenv("PRINT_LOCAL_RESPONSE", "1") == "1",
    )

    # Bind callbacks after orchestrator exists.
    speaker.on_enqueued = orch.on_chunk_enqueued
    speaker.on_spoken = orch.on_chunk_spoken

    twitch = None
    if os.getenv("TWITCH_ENABLED", "0") == "1":
        twitch = TwitchAdapter(
            token=os.getenv("TWITCH_TOKEN", ""),
            client_id=os.getenv("TWITCH_CLIENT_ID", ""),
            nick=os.getenv("TWITCH_NICK", ""),
            prefix=os.getenv("TWITCH_PREFIX", "!ask"),
        )

    return orch, twitch


async def run_default() -> None:
    orch, twitch = build_default_orchestrator()

    if orch.vts:
        await orch.vts.connect(host=os.getenv("VTS_HOST", "127.0.0.1"), port=int(os.getenv("VTS_PORT", "8001")))
        await orch.vts.authenticate(auth_token=os.getenv("VTS_AUTH_TOKEN"))

    if twitch:
        channel = os.getenv("TWITCH_CHANNEL", "")
        if not channel:
            raise RuntimeError("TWITCH_CHANNEL is required when TWITCH_ENABLED=1")
        cooldown = float(os.getenv("TWITCH_COOLDOWN_SECONDS", "15"))
        priority_users = {
            u.strip().lower()
            for u in os.getenv("TWITCH_PRIORITY_USERS", "").split(",")
            if u.strip()
        }

        await asyncio.gather(
            twitch.start(
                channel=channel,
                out_queue=orch.input_queue,
                user_cooldown_seconds=cooldown,
                priority_users=priority_users,
            ),
            orch.run(),
        )
        return

    # Local stdin fallback for phase-1/2 testing.
    async def _stdin_bridge():
        while True:
            text = await asyncio.to_thread(input, "You (!ask ...): ")
            text = text.strip()
            if not text:
                continue
            if text.lower() in {"quit", "exit", "sair"}:
                raise SystemExit(0)
            if not text.startswith("!ask"):
                continue
            await orch.input_queue.put(
                ChatMessage(user="local", text=text[len("!ask") :].strip(), timestamp=time.time())
            )

    await asyncio.gather(_stdin_bridge(), orch.run())


if __name__ == "__main__":
    asyncio.run(run_default())
