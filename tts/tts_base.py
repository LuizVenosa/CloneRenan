"""
Interface base para engines de TTS
Facilita trocar entre Edge-TTS, Coqui, ElevenLabs, etc.
"""

from abc import ABC, abstractmethod
from typing import Optional, Generator
import sounddevice as sd

class TTSEngineBase(ABC):
    """
    Classe abstrata para engines de TTS
    Qualquer engine novo deve implementar esses métodos
    """
    
    def __init__(self, 
                 output_device_name: Optional[str] = "CABLE Input",
                 enable_monitor: bool = True):
        
        self.output_device = None
        self.speaker_device = None
        self.enable_monitor = enable_monitor
        
        # Encontrar dispositivos de áudio
        self._setup_audio_devices(output_device_name)
    
    def _setup_audio_devices(self, output_device_name: Optional[str]):
        """Configura dispositivos de áudio"""
        devices = sd.query_devices()
        
        # VTube Studio device
        if output_device_name:
            for i, device in enumerate(devices):
                if output_device_name.lower() in device['name'].lower():
                    self.output_device = i
                    print(f"✓ VTube device: {device['name']}")
                    break
        
        # Monitor device (speakers)
        if self.enable_monitor:
            for i, device in enumerate(devices):
                if 'speakers' in device['name'].lower() or 'alto-falantes' in device['name'].lower():
                    self.speaker_device = i
                    print(f"✓ Monitor device: {device['name']}")
                    break
    
    @abstractmethod
    def generate_audio(self, text: str) -> bytes:
        """
        Gera áudio para o texto completo
        
        Returns:
            bytes: Audio data
        """
        pass
    
    @abstractmethod
    def speak(self, text: str, blocking: bool = True):
        """
        Fala o texto
        
        Args:
            text: Texto para falar
            blocking: Se True, espera terminar
        """
        pass
    
    def play_audio_data(self, audio_data, samplerate: int):
        """
        Toca dados de áudio nos dispositivos configurados
        
        Args:
            audio_data: Numpy array com dados de áudio
            samplerate: Taxa de amostragem
        """
        # Toca no CABLE Input (VTube Studio)
        if self.output_device is not None:
            sd.play(audio_data, samplerate, device=self.output_device)
        
        # Toca nos speakers (monitor)
        if self.enable_monitor and self.speaker_device is not None:
            sd.play(audio_data, samplerate, device=self.speaker_device)
        
        sd.wait()
    
    @abstractmethod
    def set_voice(self, voice: str):
        """Muda a voz"""
        pass
    
    @abstractmethod
    def set_speed(self, speed: float):
        """Muda a velocidade (1.0 = normal, 1.2 = 20% mais rápido)"""
        pass
    
    @staticmethod
    @abstractmethod
    def list_available_voices():
        """Lista vozes disponíveis"""
        pass


class StreamingTTSEngineBase(TTSEngineBase):
    """
    Interface para engines com suporte a streaming
    Engines que podem gerar áudio em chunks
    """
    
    @abstractmethod
    def generate_audio_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        Gera áudio em chunks (streaming)
        
        Yields:
            bytes: Chunks de áudio
        """
        pass
    
    @abstractmethod
    def speak_stream(self, text: str):
        """
        Fala texto usando streaming
        Começa a tocar antes de gerar todo o áudio
        """
        pass