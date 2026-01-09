"""
Sistema de streaming TTS
Quebra texto em sentenças e processa em paralelo
"""

import re
import threading
import queue
import time
from typing import Optional, Callable
from tts_base import TTSEngineBase

class StreamingTTS:
    """
    Orquestrador de TTS streaming
    
    Quebra texto em chunks e processa em paralelo:
    1. LLM gera tokens → Buffer acumula
    2. Detecta fim de sentença → Envia para fila
    3. Worker processa fila → Gera áudio → Toca
    """
    
    def __init__(self, 
                 tts_engine: TTSEngineBase,
                 min_chunk_length: int = 20,  # Mínimo de caracteres para processar
                 max_chunk_length: int = 300):  # Máximo de caracteres por chunk
        
        self.engine = tts_engine
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        
        # Fila de sentenças para processar
        self.sentence_queue = queue.Queue()
        
        # Controle de estado
        self.is_speaking = False
        self.should_stop = False
        
        # Worker thread
        self.worker_thread = None
        
        # Buffer para texto parcial
        self.text_buffer = ""
        
        # Padrões para detectar fim de sentença
        self.sentence_endings = re.compile(r'[.!?]+\s+|[.!?]+$|\n+')
    
    def start(self):
        """Inicia worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.should_stop = False
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("🔊 TTS Worker iniciado")
    
    def stop(self):
        """Para worker thread"""
        self.should_stop = True
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        print("🔇 TTS Worker parado")
    
    def _worker_loop(self):
        """Loop do worker que processa fila de sentenças"""
        while not self.should_stop:
            try:
                # Pega próxima sentença (timeout 0.5s)
                sentence = self.sentence_queue.get(timeout=0.5)
                
                if sentence is None:  # Sinal de parada
                    break
                
                # Marca que está falando
                self.is_speaking = True
                
                # Gera e toca áudio
                print(f"🗣️ [{len(sentence)} chars] {sentence[:50]}...")
                self.engine.speak(sentence, blocking=True)
                
                # Marca tarefa como concluída
                self.sentence_queue.task_done()
                self.is_speaking = False
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ Erro no worker TTS: {e}")
                self.is_speaking = False
    
    def add_text(self, text: str):
        """
        Adiciona texto ao buffer
        Detecta sentenças completas e envia para fila
        
        Args:
            text: Texto novo (pode ser 1 token ou vários)
        """
        self.text_buffer += text
        
        # Tenta extrair sentenças completas
        sentences = self._extract_sentences()
        
        for sentence in sentences:
            if len(sentence.strip()) >= self.min_chunk_length:
                self.sentence_queue.put(sentence.strip())
    
    def flush(self):
        """
        Força processar texto restante no buffer
        Chame isso quando o LLM terminar de gerar
        """
        if self.text_buffer.strip():
            # Processa o que sobrou
            remaining = self.text_buffer.strip()
            
            # Quebra em chunks se for muito longo
            chunks = self._split_long_text(remaining)
            
            for chunk in chunks:
                if len(chunk.strip()) > 0:
                    self.sentence_queue.put(chunk.strip())
            
            self.text_buffer = ""
    
    def _extract_sentences(self) -> list:
        """
        Extrai sentenças completas do buffer
        Mantém texto incompleto no buffer
        
        Returns:
            list: Lista de sentenças completas
        """
        sentences = []
        
        # Encontra todas as posições de fim de sentença
        matches = list(self.sentence_endings.finditer(self.text_buffer))
        
        if not matches:
            return sentences
        
        last_end = 0
        
        for match in matches:
            sentence = self.text_buffer[last_end:match.end()].strip()
            
            if sentence:
                sentences.append(sentence)
            
            last_end = match.end()
        
        # Atualiza buffer com texto restante
        self.text_buffer = self.text_buffer[last_end:]
        
        return sentences
    
    def _split_long_text(self, text: str) -> list:
        """
        Divide texto longo em chunks menores
        Tenta quebrar em vírgulas ou espaços
        """
        if len(text) <= self.max_chunk_length:
            return [text]
        
        chunks = []
        
        # Tenta quebrar em vírgulas
        parts = text.split(',')
        
        current_chunk = ""
        
        for part in parts:
            if len(current_chunk) + len(part) <= self.max_chunk_length:
                current_chunk += part + ","
            else:
                if current_chunk:
                    chunks.append(current_chunk.rstrip(','))
                current_chunk = part + ","
        
        if current_chunk:
            chunks.append(current_chunk.rstrip(','))
        
        return chunks
    
    def wait_until_done(self):
        """Espera até todas as sentenças serem processadas"""
        self.sentence_queue.join()
    
    def reset(self):
        """Reseta estado (limpa buffer e fila)"""
        self.text_buffer = ""
        
        # Limpa fila
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except queue.Empty:
                break


class TextStreamHandler:
    """
    Helper class para processar stream de texto do LLM
    Usado com LangChain/LangGraph
    """
    
    def __init__(self, streaming_tts: StreamingTTS):
        self.tts = streaming_tts
        self.full_text = ""
    
    def on_token(self, token: str):
        """
        Callback para cada token do LLM
        
        Args:
            token: Novo token gerado
        """
        self.full_text += token
        self.tts.add_text(token)
    
    def on_finish(self):
        """Callback quando LLM termina"""
        self.tts.flush()
    
    def get_full_text(self) -> str:
        """Retorna texto completo gerado"""
        return self.full_text
    
    def reset(self):
        """Reseta handler"""
        self.full_text = ""


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    from tts_edge_engine import EdgeTTSEngine
    import time
    
    print("\n" + "="*60)
    print("TESTE: TTS STREAMING")
    print("="*60 + "\n")
    
    # Cria engine
    engine = EdgeTTSEngine(
        output_device_name="CABLE Input",
        enable_monitor=True,
        rate="+20%"  # Mais rápido
    )
    
    # Cria streaming TTS
    streaming = StreamingTTS(engine, min_chunk_length=15)
    streaming.start()
    
    # Simula LLM gerando texto aos poucos
    texto_simulado = """
    A política brasileira é um teatro de sombras, meu caro. 
    Não se iluda com discursos pomposos. 
    O que importa é o poder, sempre foi e sempre será. 
    E o povo? Ah, o povo é apenas espectador nessa peça macabra.
    """
    
    print("📝 Simulando LLM gerando texto...\n")
    
    # Simula streaming (adiciona palavra por palavra)
    palavras = texto_simulado.split()
    
    for palavra in palavras:
        streaming.add_text(palavra + " ")
        time.sleep(0.1)  # Simula latência do LLM
    
    # Força processar o resto
    streaming.flush()
    
    # Espera terminar
    print("\n⏳ Aguardando TTS terminar...")
    streaming.wait_until_done()
    
    print("✅ Teste concluído!")
    
    streaming.stop()