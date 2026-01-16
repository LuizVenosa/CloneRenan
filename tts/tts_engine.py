import asyncio
import edge_tts
import sounddevice as sd
import soundfile as sf
import os
from typing import Optional

class RenanTTS:
    def __init__(self, 
                 voice_sample_path: Optional[str] = None,  # Mantém compatibilidade
                 output_device_name: str = "CABLE Input",
                 enable_monitor: bool = True):
        print("🎙️ Configurando Edge-TTS...")
        
        # Listar dispositivos
        devices = sd.query_devices()
        self.output_device = None
        self.speaker_device = None
        
        # CABLE Input (VTube Studio)
        for i, device in enumerate(devices):
            if output_device_name.lower() in device['name'].lower():
                self.output_device = i
                print(f"✓ VTube device: {device['name']}")
                break
        
        # Speakers (monitor)
        if enable_monitor:
            for i, device in enumerate(devices):
                if 'speakers' in device['name'].lower() or 'alto-falantes' in device['name'].lower():
                    self.speaker_device = i
                    print(f"✓ Monitor device: {device['name']}")
                    break
        
        self.enable_monitor = enable_monitor
        self.voice = "pt-BR-AntonioNeural"
        self.rate = "+10%"  # Velocidade (+/- 50%)
        self.pitch = "+0Hz"  # Tom
        
        print(f"✓ Voz: {self.voice}")

    async def _speak_async(self, text):
        """Gera e toca o áudio (async)"""
        temp_file = "temp_speech.mp3"
        
        # Gerar áudio
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        await communicate.save(temp_file)
        
        # Carregar e tocar
        data, samplerate = sf.read(temp_file)
        
        # Tocar no CABLE Input (VTube Studio)
        if self.output_device is not None:
            sd.play(data, samplerate, device=self.output_device)
        
        # Tocar nos speakers (monitor)
        if self.enable_monitor and self.speaker_device is not None:
            sd.play(data, samplerate, device=self.speaker_device)
        
        sd.wait()
        
        # Limpar
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def speak(self, text):
        """Fala o texto (interface síncrona)"""
        print(f"🗣️ Falando: {text}")
        asyncio.run(self._speak_async(text))
        print("✓ Áudio reproduzido")

    async def list_voices(self):
        """Lista todas as vozes portuguesas disponíveis"""
        voices = await edge_tts.list_voices()
        print("\n=== Vozes Portuguesas Disponíveis ===")
        for voice in voices:
            if voice['Locale'].startswith('pt'):
                print(f"- {voice['ShortName']}: {voice['Gender']} ({voice['Locale']})")

# --- TESTE ---
if __name__ == "__main__":
    # Instalar: pip install edge-tts soundfile
    
    tts = RenanTTS(output_device_name="CABLE Input")
    
    # Opcional: ver todas as vozes
    # asyncio.run(tts.list_voices())
    
    print("\n=== TESTE 1: Frase curta ===")
    tts.speak("A política brasileira é um teatro de sombras, meu caro.")
    
    print("\n=== TESTE 2: Frase longa ===")
    tts.speak("Bem-vindo ao meu canal. Hoje vamos discutir economia, filosofia e os absurdos do nosso tempo.")
    
    print("\n✓ Fase 1 completa! Agora configure o VTube Studio:")
    print("  1. Abra VTube Studio")
    print("  2. Settings → Microphone → CABLE Output")
    print("  3. Teste novamente e veja o avatar se mover!")