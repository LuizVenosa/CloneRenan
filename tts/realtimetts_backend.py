import asyncio
import inspect
import json
import os
from typing import Optional


class RealtimeTTSBackend:
    """RealtimeTTS wrapper for async session-aware speech control."""

    def __init__(
        self,
        engine=None,
        engine_name: Optional[str] = None,
        engine_kwargs: Optional[dict] = None,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
    ):
        try:
            import RealtimeTTS  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "RealtimeTTS is not available. Install with: pip install realtimetts[all]"
            ) from exc

        self._stream_cls = RealtimeTTS.TextToAudioStream
        self._engine = engine or self._build_engine(
            realtime_tts_mod=RealtimeTTS,
            engine_name=engine_name or os.getenv("RTTTS_ENGINE", "system"),
            engine_kwargs=engine_kwargs or self._parse_engine_kwargs(os.getenv("RTTTS_ENGINE_KWARGS", "")),
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
        self._apply_engine_settings(voice=voice, rate=rate, pitch=pitch)
        self._stream = self._stream_cls(self._engine)

        self._active_session_id = 0
        self._pending = 0
        self._lock = asyncio.Lock()

    async def speak(self, text: str, session_id: int) -> None:
        if session_id != self._active_session_id:
            return

        async with self._lock:
            if session_id != self._active_session_id:
                return
            self._pending += 1
            try:
                self._stream.feed(text)
                await asyncio.to_thread(self._stream.play_async)
                await asyncio.sleep(0.01)
            finally:
                self._pending = max(0, self._pending - 1)

    async def cancel(self, session_id: int) -> None:
        self._active_session_id = session_id
        self._pending = 0
        try:
            await asyncio.to_thread(self._stream.stop)
        except Exception:
            pass

    async def is_idle(self, session_id: int) -> bool:
        if session_id != self._active_session_id:
            return True
        playing = False
        is_playing = getattr(self._stream, "is_playing", None)
        if callable(is_playing):
            try:
                playing = await asyncio.to_thread(is_playing)
            except Exception:
                playing = False
        return self._pending == 0 and not playing

    @staticmethod
    def _build_engine(
        realtime_tts_mod,
        engine_name: str,
        engine_kwargs: Optional[dict],
        voice: Optional[str],
        rate: Optional[str],
        pitch: Optional[str],
    ):
        cls = RealtimeTTSBackend._resolve_engine_class(realtime_tts_mod, engine_name)
        if cls is None:
            cls = RealtimeTTSBackend._resolve_engine_class(realtime_tts_mod, "system")
            if cls is None:
                raise RuntimeError("RealtimeTTS SystemEngine is unavailable.")

        try:
            sig = inspect.signature(cls)
        except Exception:
            return cls()

        kwargs = dict(engine_kwargs or {})
        params = sig.parameters

        if voice and "voice" in params:
            kwargs["voice"] = voice
        elif voice and "voice_name" in params:
            kwargs["voice_name"] = voice

        if rate and "rate" in params:
            kwargs["rate"] = RealtimeTTSBackend._coerce_number(rate)
        if pitch and "pitch" in params:
            kwargs["pitch"] = RealtimeTTSBackend._coerce_number(pitch)

        try:
            return cls(**kwargs)
        except Exception:
            return cls()

    @staticmethod
    def _resolve_engine_class(realtime_tts_mod, engine_name: str):
        name = (engine_name or "system").strip()
        if not name:
            name = "system"

        candidates = {name, f"{name}Engine", name.lower(), f"{name.lower()}engine"}

        # Try exact module attrs first.
        for attr in dir(realtime_tts_mod):
            value = getattr(realtime_tts_mod, attr, None)
            if not inspect.isclass(value):
                continue
            if attr in candidates:
                return value

        # Then case-insensitive compare.
        target_norms = {c.lower() for c in candidates}
        for attr in dir(realtime_tts_mod):
            value = getattr(realtime_tts_mod, attr, None)
            if not inspect.isclass(value):
                continue
            if attr.lower() in target_norms:
                return value

        return None

    @staticmethod
    def _parse_engine_kwargs(raw: str) -> dict:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    def _apply_engine_settings(self, voice: Optional[str], rate: Optional[str], pitch: Optional[str]) -> None:
        if voice:
            self._set_value(
                value=voice,
                method_names=("set_voice", "setVoice", "set_voice_name"),
                attr_names=("voice", "voice_name"),
                property_names=("voice",),
            )
        if rate:
            rate_value = self._coerce_number(rate)
            self._set_value(
                value=rate_value,
                method_names=("set_rate", "setRate"),
                attr_names=("rate",),
                property_names=("rate",),
            )
        if pitch:
            pitch_value = self._coerce_number(pitch)
            self._set_value(
                value=pitch_value,
                method_names=("set_pitch", "setPitch"),
                attr_names=("pitch",),
                property_names=("pitch",),
            )

    def _set_value(
        self,
        value,
        method_names: tuple[str, ...],
        attr_names: tuple[str, ...],
        property_names: tuple[str, ...],
    ) -> None:
        for method_name in method_names:
            method = getattr(self._engine, method_name, None)
            if callable(method):
                try:
                    method(value)
                    return
                except Exception:
                    pass

        set_property = getattr(self._engine, "setProperty", None)
        if callable(set_property):
            for name in property_names:
                try:
                    set_property(name, value)
                    return
                except Exception:
                    pass

        for attr_name in attr_names:
            if hasattr(self._engine, attr_name):
                try:
                    setattr(self._engine, attr_name, value)
                    return
                except Exception:
                    pass

    @staticmethod
    def _coerce_number(value):
        if isinstance(value, str):
            cleaned = value.strip()
            try:
                if "." in cleaned:
                    return float(cleaned)
                return int(cleaned)
            except Exception:
                return value
        return value
