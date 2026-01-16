"""
Filtros para limpar texto antes de enviar ao TTS
Remove URLs, caracteres especiais, etc.
"""

import re
from typing import Optional

class TTSTextFilter:
    """
    Filtra e limpa texto para TTS
    Remove conteúdo que não deve ser falado
    """
    
    def __init__(self,
                 remove_urls: bool = True,
                 remove_markdown: bool = True,
                 remove_special_chars: bool = True,
                 max_sentence_length: int = 300):
        
        self.remove_urls = remove_urls
        self.remove_markdown = remove_markdown
        self.remove_special_chars = remove_special_chars
        self.max_sentence_length = max_sentence_length
        
        # Padrões regex
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.markdown_bold = re.compile(r'\*\*(.*?)\*\*')
        self.markdown_italic = re.compile(r'\*(.*?)\*')
        self.markdown_code = re.compile(r'`(.*?)`')
        self.markdown_link = re.compile(r'\[(.*?)\]\(.*?\)')
    
    def filter(self, text: str) -> str:
        """
        Aplica todos os filtros no texto
        
        Args:
            text: Texto original
        
        Returns:
            str: Texto filtrado
        """
        if not text:
            return text
        
        # Remove URLs
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # Remove markdown
        if self.remove_markdown:
            text = self._clean_markdown(text)
        
        # Remove caracteres especiais
        if self.remove_special_chars:
            text = self._clean_special_chars(text)
        
        # Limita tamanho de sentenças
        text = self._limit_sentence_length(text)
        
        # Limpa espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs do texto"""
        return self.url_pattern.sub('', text)
    
    def _clean_markdown(self, text: str) -> str:
        """Remove formatação markdown"""
        # Remove bold mantendo conteúdo
        text = self.markdown_bold.sub(r'\1', text)
        
        # Remove italic mantendo conteúdo
        text = self.markdown_italic.sub(r'\1', text)
        
        # Remove code mantendo conteúdo
        text = self.markdown_code.sub(r'\1', text)
        
        # Remove links mantendo texto
        text = self.markdown_link.sub(r'\1', text)
        
        return text
    
    def _clean_special_chars(self, text: str) -> str:
        """Remove caracteres especiais desnecessários"""
        # Remove múltiplas pontuações
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Remove símbolos estranhos comuns em transcrições
        text = text.replace('｜', '')
        text = text.replace('⧸', '/')
        text = text.replace('>>', '')
        
        return text
    
    def _limit_sentence_length(self, text: str) -> str:
        """Limita tamanho de sentenças muito longas"""
        sentences = re.split(r'([.!?]+\s+)', text)
        result = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence) > self.max_sentence_length:
                # Quebra em vírgulas
                parts = sentence.split(',')
                current = ""
                
                for part in parts:
                    if len(current) + len(part) < self.max_sentence_length:
                        current += part + ","
                    else:
                        if current:
                            result.append(current.rstrip(',') + '.')
                        current = part + ","
                
                if current:
                    result.append(current.rstrip(',') + '.')
            else:
                result.append(sentence)
        
        return ' '.join(result)
    
    def should_skip_sentence(self, sentence: str) -> bool:
        """
        Verifica se uma sentença deve ser completamente ignorada
        
        Args:
            sentence: Sentença para verificar
        
        Returns:
            bool: True se deve pular, False caso contrário
        """
        sentence_lower = sentence.lower().strip()
        
        # Skip se é muito curta
        if len(sentence_lower) < 10:
            return True
        
        # Skip se contém apenas URL
        if self.url_pattern.search(sentence) and len(self.url_pattern.sub('', sentence).strip()) < 10:
            return True
        
        # Skip se começa com "Fonte", "Link:", etc.
        skip_prefixes = ['fonte', 'link:', 'http', 'youtube.com', 'youtu.be']
        if any(sentence_lower.startswith(prefix) for prefix in skip_prefixes):
            return True
        
        return False


# ============================================================================
# WRAPPER PARA STREAMING TTS
# ============================================================================

class FilteredStreamingTTS:
    """
    Wrapper para StreamingTTS que aplica filtros
    """
    
    def __init__(self, streaming_tts, text_filter: Optional[TTSTextFilter] = None):
        self.streaming_tts = streaming_tts
        self.filter = text_filter or TTSTextFilter()
    
    def add_text(self, text: str):
        """Adiciona texto com filtro"""
        # Filtra antes de enviar
        filtered = self.filter.filter(text)
        
        if filtered:
            self.streaming_tts.add_text(filtered)
    
    def flush(self):
        """Flush do streaming TTS"""
        self.streaming_tts.flush()
    
    def wait_until_done(self):
        """Espera terminar"""
        self.streaming_tts.wait_until_done()
    
    def reset(self):
        """Reseta"""
        self.streaming_tts.reset()
    
    def start(self):
        """Inicia worker"""
        self.streaming_tts.start()
    
    def stop(self):
        """Para worker"""
        self.streaming_tts.stop()


# ============================================================================
# TESTES
# ============================================================================

if __name__ == "__main__":
    print("🧪 Testando filtros de TTS\n")
    
    filter = TTSTextFilter()
    
    # Teste 1: URLs
    texto1 = "Veja mais em https://youtube.com/watch?v=123 este vídeo incrível."
    print(f"Original: {texto1}")
    print(f"Filtrado: {filter.filter(texto1)}")
    print()
    
    # Teste 2: Markdown
    texto2 = "Este é um **texto em negrito** e *itálico* com `código`."
    print(f"Original: {texto2}")
    print(f"Filtrado: {filter.filter(texto2)}")
    print()
    
    # Teste 3: Caracteres especiais
    texto3 = "Análise ｜ 31⧸07⧸2025 >> texto normal aqui!!!"
    print(f"Original: {texto3}")
    print(f"Filtrado: {filter.filter(texto3)}")
    print()
    
    # Teste 4: Should skip
    sentences = [
        "Link: https://youtube.com/watch?v=123",
        "Fonte 1 (Video sobre política):",
        "Esta é uma sentença normal que deve ser falada.",
        "http://example.com",
        "Ok"
    ]
    
    print("Teste de skip:\n")
    for s in sentences:
        should_skip = filter.should_skip_sentence(s)
        status = "❌ SKIP" if should_skip else "✅ FALAR"
        print(f"{status}: {s}")