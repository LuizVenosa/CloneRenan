import os
import sys
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Optional

# Adiciona pasta pai ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# TTS imports
from tts_edge_engine import EdgeTTSEngine
from tts_streaming import StreamingTTS, TextStreamHandler
from tts_filters import TTSTextFilter, FilteredStreamingTTS

# CORREÇÃO: Carrega .env da pasta pai
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

# Verificar se API key foi carregada
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ ERRO: GOOGLE_API_KEY não encontrada!")
    print("   Verifique o arquivo .env na pasta raiz do projeto")
    exit(1)
else:
    print("✅ API Key carregada")

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True}
)
DB_PATH = "../db_clone"

# Carregar Master Prompt
prompt_path = "../prompt_clone.txt"
if os.path.exists(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        master_prompt = f.read()
    print("✅ Master prompt carregado")
else:
    master_prompt = "Você é Renan Santos, um analista político brasileiro."
    print("⚠️ prompt_clone.txt não encontrado, usando padrão")

# ============================================================================
# RAG
# ============================================================================

if os.path.exists(DB_PATH):
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.4}
    )
    print("✅ ChromaDB carregado")
else:
    retriever = None
    print("⚠️ ChromaDB não encontrado")

@tool
def pesquisar_memoria_renan(query: str) -> str:
    """Busca trechos de lives e pensamentos do Renan Santos sobre um tema."""
    if not retriever:
        return "RAG não disponível"
    
    docs = retriever.invoke(query)
    print(f"\n[DEBUG RAG] Query: {query} | Docs: {len(docs)}")
    
    resultado = []
    for i, doc in enumerate(docs, 1):
        trecho = doc.page_content[:200]
        fonte = doc.metadata.get('fonte', 'Fonte desconhecida')
        fonte_limpa = fonte.replace('.pt.srt', '').replace('.srt', '')
        resultado.append(f"Fonte {i} ({fonte_limpa}): {trecho}...")
    
    return "\n\n".join(resultado)

# ============================================================================
# LLM - CONFIGURAÇÃO CORRIGIDA
# ============================================================================

print("🤖 Inicializando Gemini...")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    streaming=True,  # CRÍTICO para streaming
).bind_tools([pesquisar_memoria_renan])

print("✅ Gemini configurado")

# ============================================================================
# BRAIN COM STREAMING - VERSÃO CORRIGIDA
# ============================================================================

class RenanBrainStreaming:
    """Brain com TTS streaming - VERSÃO CORRIGIDA"""
    
    def __init__(self,
                 enable_tts: bool = True,
                 tts_output_device: Optional[str] = "CABLE Input",
                 tts_monitor: bool = True,
                 tts_voice: str = "pt-BR-AntonioNeural",
                 tts_speed: float = 1.2):
        
        self.enable_tts = enable_tts
        self.streaming_tts = None
        
        if enable_tts:
            print("🎙️ Inicializando TTS streaming...")
            
            engine = EdgeTTSEngine(
                output_device_name=tts_output_device,
                enable_monitor=tts_monitor,
                voice=tts_voice,
                rate=f"+{int((tts_speed - 1.0) * 100)}%"
            )
            
            base_streaming = StreamingTTS(
                tts_engine=engine,
                min_chunk_length=20,
                max_chunk_length=200
            )
            
            self.streaming_tts = FilteredStreamingTTS(
                base_streaming,
                text_filter=TTSTextFilter(
                    remove_urls=True,
                    remove_markdown=True,
                    remove_special_chars=True
                )
            )
            
            self.streaming_tts.start()
            print("✅ TTS streaming pronto")
        
        self._build_graph()
    
    def _chatbot_node(self, state: MessagesState):
        """
        Nó do chatbot - CORREÇÃO: adiciona system message corretamente
        """
        # IMPORTANTE: Adiciona system instruction via messages
        messages = state["messages"]
        
        # Se não tem system message, adiciona
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=master_prompt)] + messages
        
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def _build_graph(self):
        """Constrói grafo"""
        workflow = StateGraph(MessagesState)
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.add_node("tools", ToolNode([pesquisar_memoria_renan]))
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", tools_condition)
        workflow.add_edge("tools", "chatbot")
        self.agent = workflow.compile()
    
    def chat(self, user_message: str, speak: bool = None) -> str:
        """
        Chat com streaming TTS - VERSÃO CORRIGIDA
        """
        should_speak = speak if speak is not None else self.enable_tts
        
        if should_speak and self.streaming_tts:
            stream_handler = TextStreamHandler(self.streaming_tts)
            stream_handler.reset()
            self.streaming_tts.reset()
        
        inputs = {"messages": [HumanMessage(content=user_message)]}
        
        print(f"\n👤 Você: {user_message}")
        print("🤖 Renan: ", end="", flush=True)
        
        full_response = ""
        
        # CORREÇÃO: Usa stream() sem stream_mode
        for event in self.agent.stream(inputs):
            # Processa cada nó
            for node_name, node_output in event.items():
                # Só processa saída do chatbot
                if node_name == "chatbot" and "messages" in node_output:
                    messages = node_output["messages"]
                    
                    if messages:
                        last_msg = messages[-1]
                        
                        # Verifica se é AIMessage com conteúdo
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            content = last_msg.content
                            
                            # Pega novo conteúdo
                            new_content = content[len(full_response):]
                            
                            if new_content:
                                # Imprime
                                print(new_content, end="", flush=True)
                                
                                # Envia para TTS
                                if should_speak and self.streaming_tts:
                                    stream_handler.on_token(new_content)
                                
                                full_response = content
        
        print()
        
        # Finaliza TTS
        if should_speak and self.streaming_tts and full_response:
            stream_handler.on_finish()
            self.streaming_tts.wait_until_done()
            print()
        
        return full_response
    
    def chat_session(self):
        """Sessão interativa"""
        messages = []
        
        print("\n" + "="*60)
        print("🧠 RENAN SANTOS AI - STREAMING TTS (CORRIGIDO)")
        print("="*60)
        
        if self.enable_tts:
            print("🎙️ TTS STREAMING: ATIVADO")
            print("   → Fala enquanto pensa!")
        else:
            print("🔇 TTS: DESATIVADO")
        
        print("\nComandos:")
        print("  'sair' - Encerra")
        print("  'falar' - Liga/desliga TTS")
        print("  'limpar' - Limpa histórico")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("👤 Você: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['sair', 'exit']:
                    print("\n👋 Até logo!")
                    break
                
                if user_input.lower() == 'falar':
                    self.enable_tts = not self.enable_tts
                    status = "ATIVADO" if self.enable_tts else "DESATIVADO"
                    print(f"🎙️ TTS: {status}")
                    continue
                
                if user_input.lower() == 'limpar':
                    messages = []
                    print("🗑️ Histórico limpo")
                    continue
                
                # Adiciona ao histórico
                messages.append(HumanMessage(content=user_input))
                
                # Prepara TTS
                if self.enable_tts and self.streaming_tts:
                    stream_handler = TextStreamHandler(self.streaming_tts)
                    stream_handler.reset()
                    self.streaming_tts.reset()
                
                inputs = {"messages": messages}
                
                print("🤖 Renan: ", end="", flush=True)
                
                full_response = ""
                
                # Stream com histórico
                for event in self.agent.stream(inputs):
                    for node_name, node_output in event.items():
                        if node_name == "chatbot" and "messages" in node_output:
                            node_messages = node_output["messages"]
                            
                            if node_messages:
                                last_msg = node_messages[-1]
                                
                                if isinstance(last_msg, AIMessage) and last_msg.content:
                                    content = last_msg.content
                                    new_content = content[len(full_response):]
                                    
                                    if new_content:
                                        print(new_content, end="", flush=True)
                                        
                                        if self.enable_tts and self.streaming_tts:
                                            stream_handler.on_token(new_content)
                                        
                                        full_response = content
                
                print()
                
                # Adiciona resposta ao histórico
                messages.append(AIMessage(content=full_response))
                
                # Finaliza TTS
                if self.enable_tts and self.streaming_tts and full_response:
                    stream_handler.on_finish()
                    self.streaming_tts.wait_until_done()
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                import traceback
                traceback.print_exc()
        
        if self.streaming_tts:
            self.streaming_tts.stop()


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tts", action="store_true", help="Desabilita TTS")
    parser.add_argument("--speed", type=float, default=1.2, help="Velocidade TTS")
    args = parser.parse_args()
    
    brain = RenanBrainStreaming(
        enable_tts=not args.no_tts,
        tts_monitor=True,
        tts_speed=args.speed
    )
    
    brain.chat_session()