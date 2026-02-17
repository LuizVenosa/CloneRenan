import asyncio
import io
from typing import Optional

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf


class EdgeBackend:
    """Async Edge-TTS fallback backend."""

    def __init__(
        self,
        voice: str = "pt-BR-AntonioNeural",
        rate: str = "+20%",
        pitch: str = "+0Hz",
        output_device_name: Optional[str] = None,
    ):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.output_device = self._find_device(output_device_name)

        self._active_session_id = 0
        self._playing = False

    async def speak(self, text: str, session_id: int) -> None:
        if session_id != self._active_session_id:
            return

        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        mp3_data = b""
        async for chunk in communicate.aiter():
            if session_id != self._active_session_id:
                return
            if chunk.get("type") == "audio" and chunk.get("data"):
                mp3_data += chunk["data"]

        if not mp3_data or session_id != self._active_session_id:
            return

        self._playing = True
        try:
            audio, sample_rate = await asyncio.to_thread(self._decode_mp3, mp3_data)
            if audio is None:
                return
            await asyncio.to_thread(sd.play, audio, sample_rate, self.output_device)
            await asyncio.to_thread(sd.wait)
        finally:
            self._playing = False

    async def cancel(self, session_id: int) -> None:
        self._active_session_id = session_id
        try:
            await asyncio.to_thread(sd.stop)
        except Exception:
            pass
        self._playing = False

    async def is_idle(self, session_id: int) -> bool:
        return session_id != self._active_session_id or not self._playing

    @staticmethod
    def _decode_mp3(mp3_data: bytes):
        try:
            data, sample_rate = sf.read(io.BytesIO(mp3_data), dtype="float32")
            if isinstance(data, np.ndarray) and data.size > 0:
                return data, sample_rate
        except Exception:
            pass
        return None, None

    @staticmethod
    def _find_device(output_device_name: Optional[str]):
        if not output_device_name:
            return None
        try:
            devices = sd.query_devices()
            for idx, device in enumerate(devices):
                if output_device_name.lower() in device.get("name", "").lower():
                    return idx
        except Exception:
            return None
        return None
