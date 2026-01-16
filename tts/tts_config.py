"""
Configuração centralizada para TTS
Facilita trocar engines e ajustar parâmetros
"""

from enum import Enum
from typing import Optional
from tts_base import TTSEngineBase

class TTSEngine(Enum):
    """Engines de TTS disponíveis"""
    EDGE_TTS = "edge"
    # COQUI = "coqui"      # Adicione quando implementar
    # ELEVENLABS = "elevenlabs"
    # BARK = "bark"


class TTSConfig:
    """
    Configuração global de TTS
    Facilita mudanças sem alterar código
    """
    
    # Engine padrão
    DEFAULT_ENGINE = TTSEngine.EDGE_TTS
    
    # Configurações de áudio
    OUTPUT_DEVICE = "CABLE Input"  # Para VTube Studio
    ENABLE_MONITOR = True          # Ouvir nos speakers também
    
    # Configurações Edge-TTS
    EDGE_VOICE = "pt-BR-AntonioNeural"
    EDGE_SPEED = 1.2  # 20% mais rápido
    EDGE_PITCH = "+0Hz"
    
    # Configurações de streaming
    MIN_CHUNK_LENGTH = 20   # Caracteres mínimos para processar
    MAX_CHUNK_LENGTH = 200  # Máximo por chunk
    
    # Configurações Coqui (quando implementar)
    # COQUI_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
    # COQUI_VOICE_SAMPLE = "renan_sample.wav"
    
    # Configurações ElevenLabs (quando implementar)
    # ELEVENLABS_API_KEY = ""
    # ELEVENLABS_VOICE_ID = ""
    
    @classmethod
    def create_engine(cls, 
                     engine_type: Optional[TTSEngine] = None,
                     **override_kwargs) -> TTSEngineBase:
        """
        Factory method para criar engine de TTS
        
        Args:
            engine_type: Tipo de engine (None = usa padrão)
            **override_kwargs: Sobrescreve configurações padrão
        
        Returns:
            TTSEngineBase: Engine configurado
        
        Examples:
            # Usar padrões
            engine = TTSConfig.create_engine()
            
            # Trocar voz
            engine = TTSConfig.create_engine(voice="pt-BR-HumbertoNeural")
            
            # Trocar engine completamente (quando implementar)
            # engine = TTSConfig.create_engine(TTSEngine.COQUI)
        """
        engine_type = engine_type or cls.DEFAULT_ENGINE
        
        if engine_type == TTSEngine.EDGE_TTS:
            from tts_edge_engine import EdgeTTSEngine
            
            # Configurações padrão
            config = {
                "output_device_name": cls.OUTPUT_DEVICE,
                "enable_monitor": cls.ENABLE_MONITOR,
                "voice": cls.EDGE_VOICE,
                "rate": f"+{int((cls.EDGE_SPEED - 1.0) * 100)}%",
                "pitch": cls.EDGE_PITCH
            }
            
            # Sobrescreve com kwargs
            config.update(override_kwargs)
            
            return EdgeTTSEngine(**config)
        
        # elif engine_type == TTSEngine.COQUI:
        #     from tts_coqui_engine import CoquiEngine
        #     ...
        
        else:
            raise ValueError(f"Engine não implementado: {engine_type}")
    
    @classmethod
    def create_brain(cls, **kwargs):
        """
        Factory method para criar brain com configurações padrão
        
        Args:
            **kwargs: Sobrescreve configurações
        
        Returns:
            RenanBrainStreaming configurado
        
        Examples:
            # Usar padrões
            brain = TTSConfig.create_brain()
            
            # Customizar
            brain = TTSConfig.create_brain(
                tts_speed=1.5,
                tts_voice="pt-BR-HumbertoNeural"
            )
        """
        from brain_streaming import RenanBrainStreaming
        
        config = {
            "enable_tts": True,
            "tts_output_device": cls.OUTPUT_DEVICE,
            "tts_monitor": cls.ENABLE_MONITOR,
            "tts_voice": cls.EDGE_VOICE,
            "tts_speed": cls.EDGE_SPEED
        }
        
        config.update(kwargs)
        
        return RenanBrainStreaming(**config)


# ============================================================================
# PERFIS PRÉ-CONFIGURADOS
# ============================================================================

class TTSPresets:
    """Perfis pré-configurados para diferentes casos de uso"""
    
    @staticmethod
    def fast():
        """Máxima velocidade (testes rápidos)"""
        return TTSConfig.create_brain(
            tts_speed=1.5,
        )
    
    @staticmethod
    def natural():
        """Velocidade natural (produção)"""
        return TTSConfig.create_brain(
            tts_speed=1.0,
        )
    
    @staticmethod
    def quality():
        """Máxima qualidade (apresentações)"""
        return TTSConfig.create_brain(
            tts_speed=0.95,
        )
    
    @staticmethod
    def stream():
        """Otimizado para streaming (Twitch/YouTube)"""
        return TTSConfig.create_brain(
            tts_speed=1.1,
            tts_monitor=False  # Só VTube Studio
        )
    
    @staticmethod
    def development():
        """Desenvolvimento (ouve e envia para VTube)"""
        return TTSConfig.create_brain(
            tts_speed=1.3,
            tts_monitor=True
        )


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("🎛️ TTS Configuration Demo\n")
    
    # Opção 1: Usar configurações padrão
    print("1. Configurações padrão:")
    brain1 = TTSConfig.create_brain()
    print(f"   ✓ Brain criado com configurações padrão\n")
    
    # Opção 2: Customizar
    print("2. Customizado:")
    brain2 = TTSConfig.create_brain(
        tts_speed=1.5,
        tts_voice="pt-BR-HumbertoNeural"
    )
    print(f"   ✓ Brain criado com voz Humberto, velocidade 1.5x\n")
    
    # Opção 3: Usar preset
    print("3. Usando preset 'stream':")
    brain3 = TTSPresets.stream()
    print(f"   ✓ Brain otimizado para streaming\n")
    
    # Opção 4: Criar só engine
    print("4. Criar apenas engine:")
    engine = TTSConfig.create_engine(voice="pt-BR-FranciscaNeural")
    print(f"   ✓ Engine Edge-TTS criado com voz Francisca\n")
    
    print("="*60)
    print("Para usar no seu código:")
    print("="*60)
    print("""
# Jeito fácil (padrões)
from tts_config import TTSConfig
brain = TTSConfig.create_brain()
brain.chat_session()

# Jeito rápido (preset)
from tts_config import TTSPresets
brain = TTSPresets.stream()
brain.chat_session()

# Jeito customizado
brain = TTSConfig.create_brain(
    tts_speed=1.2,
    tts_voice="pt-BR-HumbertoNeural",
    tts_monitor=False
)
brain.chat_session()
    """)