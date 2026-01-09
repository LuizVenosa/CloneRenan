import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Optional

# TTS imports
from tts_edge_engine import EdgeTTSEngine
from tts_streaming import StreamingTTS, TextStreamHandler
from tts_filters import TTSTextFilter, FilteredStreamingTTS

load_dotenv()

# ============================================================================
# CONFIGURAÇÃO (igual brain.py)
# ============================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True}
)
DB_PATH = "./db_clone"

if os.path.exists("prompt_clone.txt"):
    with open("prompt_clone.txt", "r", encoding="utf-8") as f:
        master_prompt = f.read()
else:
    master_prompt = "Você é um assistente."

# ============================================================================
# RAG
# ============================================================================

vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.4}
)

@tool
def pesquisar_memoria_renan(query: str) -> str:
    """Busca trechos de lives e pensamentos do Renan Santos sobre um tema."""
    docs = retriever.invoke(query)
    print(f"\n[DEBUG RAG] Query: {query} | Docs: {len(docs)}")
    
    resultado = []
    for i, doc in enumerate(docs, 1):
        trecho = doc.page_content  
        url = doc.metadata.get('url', 'URL não disponível')
        fonte = doc.metadata.get('fonte', 'Fonte desconhecida')
        fonte_limpa = fonte.replace('.pt.srt', '').replace('.srt', '')
        
        resultado.append(
            f"Fonte {i} ({fonte_limpa}):\n"
            f'"{trecho}..."\n'
            f"Link: {url}"
        )
    
    return "\n\n".join(resultado)

# ============================================================================
# LLM
# ============================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    streaming=True,  # IMPORTANTE: Habilita streaming
    model_kwargs={"system_instruction": master_prompt}
).bind_tools([pesquisar_memoria_renan])

# ============================================================================
# BRAIN COM STREAMING
# ============================================================================

class RenanBrainStreaming:
    """
    Brain do Renan com TTS streaming
    Fala enquanto gera texto
    """
    
    def __init__(self,
                 enable_tts: bool = True,
                 tts_output_device: Optional[str] = "CABLE Input",
                 tts_monitor: bool = True,
                 tts_voice: str = "pt-BR-AntonioNeural",
                 tts_speed: float = 1.2):  # 20% mais rápido
        
        self.enable_tts = enable_tts
        self.streaming_tts = None
        
        # Inicializa TTS se habilitado
        if enable_tts:
            print("🎙️ Inicializando TTS streaming...")
            
            # Cria engine
            engine = EdgeTTSEngine(
                output_device_name=tts_output_device,
                enable_monitor=tts_monitor,
                voice=tts_voice,
                rate=f"+{int((tts_speed - 1.0) * 100)}%"
            )
            
            # Cria streaming TTS
            base_streaming = StreamingTTS(
                tts_engine=engine,
                min_chunk_length=20,  # Sentenças curtas começam mais rápido
                max_chunk_length=200
            )
            
            # Envolve com filtros (remove URLs, etc.)
            self.streaming_tts = FilteredStreamingTTS(
                base_streaming,
                text_filter=TTSTextFilter(
                    remove_urls=True,
                    remove_markdown=True,
                    remove_special_chars=True
                )
            )
            
            # Inicia worker
            self.streaming_tts.start()
            
            print("✓ TTS streaming pronto")
        
        # Cria grafo
        self._build_graph()
    
    def _chatbot_node(self, state: MessagesState):
        """Nó do chatbot"""
        messages = [SystemMessage(content=master_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def _build_graph(self):
        """Constrói grafo LangGraph"""
        workflow = StateGraph(MessagesState)
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.add_node("tools", ToolNode([pesquisar_memoria_renan]))
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", tools_condition)
        workflow.add_edge("tools", "chatbot")
        self.agent = workflow.compile()
    
    def chat(self, user_message: str, speak: bool = None) -> str:
        """
        Chat com streaming TTS
        
        Args:
            user_message: Mensagem do usuário
            speak: Se deve falar (None = usa self.enable_tts)
        
        Returns:
            Resposta completa
        """
        should_speak = speak if speak is not None else self.enable_tts
        
        # Cria handler de streaming
        if should_speak and self.streaming_tts:
            stream_handler = TextStreamHandler(self.streaming_tts)
            stream_handler.reset()
            self.streaming_tts.reset()
        
        inputs = {"messages": [HumanMessage(content=user_message)]}
        
        print(f"\n👤 Você: {user_message}")
        print("🤖 Renan: ", end="", flush=True)
        
        full_response = ""
        in_final_response = False
        
        # Processa stream do LLM
        for chunk in self.agent.stream(inputs, stream_mode="values"):
            if "messages" in chunk:
                last_msg = chunk["messages"][-1]
                
                # FILTRO: Só processa se for mensagem do ASSISTENTE (AIMessage)
                if hasattr(last_msg, '__class__') and last_msg.__class__.__name__ == 'AIMessage':
                    # Se tem content string (não é tool call)
                    if hasattr(last_msg, 'content') and isinstance(last_msg.content, str) and last_msg.content:
                        in_final_response = True
                        
                        # Pega novo conteúdo
                        new_content = last_msg.content[len(full_response):]
                        
                        if new_content:
                            # Imprime
                            print(new_content, end="", flush=True)
                            
                            # Envia para TTS (se habilitado)
                            if should_speak and self.streaming_tts:
                                stream_handler.on_token(new_content)
                            
                            full_response = last_msg.content
        
        print()  # Nova linha
        
        # Finaliza TTS
        if should_speak and self.streaming_tts and in_final_response:
            stream_handler.on_finish()
            self.streaming_tts.wait_until_done()
            print()
        else:
            print()
        
        return full_response
    
    def chat_session(self):
        """Sessão de chat interativa"""
        messages = []
        
        print("\n" + "="*60)
        print("🧠 RENAN SANTOS AI - STREAMING TTS")
        print("="*60)
        
        if self.enable_tts:
            print("🎙️ TTS STREAMING: ATIVADO")
            print("   → Fala enquanto pensa (mais natural!)")
        else:
            print("🔇 TTS: DESATIVADO")
        
        print("\nComandos:")
        print("  'sair' - Encerra")
        print("  'falar' - Liga/desliga TTS")
        print("  'limpar' - Limpa histórico")
        print("  'velocidade X' - Muda velocidade (ex: velocidade 1.5)")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("👤 Você: ").strip()
                
                if not user_input:
                    continue
                
                # Comandos
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
                
                if user_input.lower().startswith('velocidade '):
                    try:
                        speed = float(user_input.split()[1])
                        if self.streaming_tts:
                            self.streaming_tts.engine.set_speed(speed)
                        print(f"⚡ Velocidade: {speed}x")
                    except:
                        print("❌ Uso: velocidade 1.2")
                    continue
                
                # Processa mensagem
                messages.append(HumanMessage(content=user_input))
                
                # Cria handler de streaming
                if self.enable_tts and self.streaming_tts:
                    stream_handler = TextStreamHandler(self.streaming_tts)
                    stream_handler.reset()
                    self.streaming_tts.reset()
                
                inputs = {"messages": messages}
                
                print("🤖 Renan: ", end="", flush=True)
                
                full_response = ""
                in_final_response = False
                
                for chunk in self.agent.stream(inputs, stream_mode="values"):
                    if "messages" in chunk:
                        last_msg = chunk["messages"][-1]
                        
                        # FILTRO: Só processa AIMessage com content string
                        if hasattr(last_msg, '__class__') and last_msg.__class__.__name__ == 'AIMessage':
                            if hasattr(last_msg, 'content') and isinstance(last_msg.content, str) and last_msg.content:
                                in_final_response = True
                                
                                new_content = last_msg.content[len(full_response):]
                                
                                if new_content:
                                    print(new_content, end="", flush=True)
                                    
                                    if self.enable_tts and self.streaming_tts:
                                        stream_handler.on_token(new_content)
                                    
                                    full_response = last_msg.content
                
                print()
                
                messages.append(AIMessage(content=full_response))
                
                # Finaliza TTS
                if self.enable_tts and self.streaming_tts and in_final_response:
                    stream_handler.on_finish()
                    self.streaming_tts.wait_until_done()
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}\n")
        
        # Para TTS ao sair
        if self.streaming_tts:
            self.streaming_tts.stop()
    
    def __del__(self):
        """Cleanup ao destruir objeto"""
        if self.streaming_tts:
            self.streaming_tts.stop()


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    brain = RenanBrainStreaming(
        enable_tts=True,
        tts_monitor=True,
        tts_speed=1.2  # 20% mais rápido
    )
    
    brain.chat_session()