"""
Edge-TTS Engine - Implementação otimizada
"""

import asyncio
import edge_tts
import soundfile as sf
import os
import tempfile
from typing import Optional
from tts_base import TTSEngineBase

class EdgeTTSEngine(TTSEngineBase):
    """
    Edge-TTS implementation
    Rápido, gratuito, boa qualidade
    """
    
    def __init__(self, 
                 output_device_name: Optional[str] = "CABLE Input",
                 enable_monitor: bool = True,
                 voice: str = "pt-BR-AntonioNeural",
                 rate: str = "+20%",  # Aumentado para +20% (mais rápido)
                 pitch: str = "+0Hz"):
        
        super().__init__(output_device_name, enable_monitor)
        
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        
        print(f"✓ Edge-TTS configurado: {self.voice}")
    
    def generate_audio(self, text: str) -> str:
        """
        Gera áudio e retorna caminho do arquivo temporário
        
        Returns:
            str: Caminho do arquivo .mp3
        """
        return asyncio.run(self._generate_async(text))
    
    async def _generate_async(self, text: str) -> str:
        """Gera áudio assincronamente"""
        # Arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()
        
        # Gera áudio
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        await communicate.save(temp_path)
        
        return temp_path
    
    def speak(self, text: str, blocking: bool = True):
        """Fala o texto"""
        if not text or not text.strip():
            return
        
        try:
            # Gera áudio
            audio_path = self.generate_audio(text)
            
            # Carrega e toca
            data, samplerate = sf.read(audio_path)
            self.play_audio_data(data, samplerate)
            
            # Limpa arquivo temporário
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
        
        except Exception as e:
            print(f"❌ Erro TTS: {e}")
    
    def set_voice(self, voice: str):
        """Muda a voz"""
        self.voice = voice
        print(f"✓ Voz: {voice}")
    
    def set_speed(self, speed: float):
        """
        Muda a velocidade
        
        Args:
            speed: 1.0 = normal, 1.2 = 20% mais rápido, 0.8 = 20% mais lento
        """
        # Converte para formato Edge-TTS
        percent = int((speed - 1.0) * 100)
        self.rate = f"{percent:+d}%"
        print(f"✓ Velocidade: {self.rate}")
    
    @staticmethod
    def list_available_voices():
        """Lista vozes portuguesas"""
        return [
            "pt-BR-AntonioNeural",      # Masculina, natural (RECOMENDADO)
            "pt-BR-FranciscaNeural",    # Feminina
            "pt-BR-BrendaNeural",       # Feminina
            "pt-BR-DonatoNeural",       # Masculina
            "pt-BR-FabioNeural",        # Masculina
            "pt-BR-HumbertoNeural",     # Masculina, profunda
            "pt-BR-JulioNeural",        # Masculina
            "pt-BR-NicolauNeural",      # Masculina
            "pt-BR-ValerioNeural",      # Masculina
        ]