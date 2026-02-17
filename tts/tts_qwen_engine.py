"""
Qwen3-TTS Engine
- Supports CustomVoice, VoiceClone, and VoiceDesign modes via qwen-tts
- Plays audio through configured devices (CABLE Input + optional monitor)
"""

from typing import Optional, Tuple, List
import os
import torch
import soundfile as sf
from tts_base import TTSEngineBase

try:
    from qwen_tts import Qwen3TTSModel
except Exception as exc:  # pragma: no cover
    Qwen3TTSModel = None
    _QWEN_IMPORT_ERROR = exc
else:
    _QWEN_IMPORT_ERROR = None


class QwenTTSEngine(TTSEngineBase):
    """
    Qwen3-TTS engine wrapper.

    Modes:
    - custom_voice: uses a built-in speaker from CustomVoice model
    - voice_clone: uses ref_audio + ref_text with Base model
    - voice_design: uses an instruction to design a voice (Base model)
    """

    def __init__(
        self,
        output_device_name: Optional[str] = "CABLE Input",
        enable_monitor: bool = True,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        mode: str = "custom_voice",
        language: str = "Portuguese",
        speaker: Optional[str] = None,
        instruct: str = "",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        x_vector_only_mode: bool = False,
        device_map: Optional[str] = "cuda:0",
        dtype: str = "float16",
        attn_implementation: Optional[str] = None,
    ):
        super().__init__(output_device_name, enable_monitor)

        if Qwen3TTSModel is None:
            raise ImportError(
                "qwen-tts is not installed. Install with: pip install -U qwen-tts"
            ) from _QWEN_IMPORT_ERROR

        self.model_id = model_id
        self.mode = mode
        self.language = language
        self.speaker = speaker
        self.instruct = instruct
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.x_vector_only_mode = x_vector_only_mode
        self.device_map = device_map
        self.dtype = self._resolve_dtype(dtype)
        self.attn_implementation = attn_implementation

        self.model = self._load_model()

    def _resolve_dtype(self, dtype: str):
        dtype = (dtype or "").lower()
        if dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype in ("fp32", "float32"):
            return torch.float32
        return torch.float16

    def _load_model(self):
        kwargs = {"torch_dtype": self.dtype}
        if self.device_map:
            kwargs["device_map"] = self.device_map
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        return Qwen3TTSModel.from_pretrained(self.model_id, **kwargs)

    def _select_speaker(self) -> Optional[str]:
        if self.speaker:
            return self.speaker
        try:
            speakers = self.get_supported_speakers()
            return speakers[0] if speakers else None
        except Exception:
            return None

    def get_supported_speakers(self) -> List[str]:
        if hasattr(self.model, "get_supported_speakers"):
            return list(self.model.get_supported_speakers())
        return []

    def generate_audio(self, text: str) -> Tuple[list, int]:
        if not text or not text.strip():
            return [], 0

        mode = (self.mode or "custom_voice").lower()

        if mode == "custom_voice":
            speaker = self._select_speaker()
            if not speaker:
                raise ValueError("No speaker available for CustomVoice model")
            wavs, sr = self.model.generate(
                text,
                speaker=speaker,
                language=self.language,
            )
        elif mode == "voice_clone":
            if not self.ref_audio or not self.ref_text:
                raise ValueError("voice_clone requires ref_audio and ref_text")
            wavs, sr = self.model.generate_voice_clone(
                text,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
                language=self.language,
                x_vector_only_mode=self.x_vector_only_mode,
            )
        elif mode == "voice_design":
            if not self.instruct:
                raise ValueError("voice_design requires instruct")
            wavs, sr = self.model.generate_voice_design(
                text,
                instruct=self.instruct,
                language=self.language,
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return wavs, sr

    def speak(self, text: str, blocking: bool = True):
        wavs, sr = self.generate_audio(text)
        if not wavs or sr <= 0:
            return
        self.play_audio_data(wavs[0], sr)

    def set_voice(self, voice: str):
        self.speaker = voice

    def set_speed(self, speed: float):
        # Qwen3-TTS does not expose a direct speed control in the public API.
        # Keep for interface compatibility.
        pass

    @staticmethod
    def list_available_voices():
        # Requires a model instance; use get_supported_speakers() on an engine.
        return []

    def save_wav(self, text: str, output_path: str):
        wavs, sr = self.generate_audio(text)
        if not wavs or sr <= 0:
            raise RuntimeError("No audio generated")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, wavs[0], sr)
