import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from tts_engine import RenanTTS  # Import do seu TTS
from typing import Optional

load_dotenv()

# ============================================================================
# CONFIGURAÇÃO DE AMBIENTE (igual brain.py)
# ============================================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True}
)
DB_PATH = "../db_clone"

# Carregar Master Prompt
if os.path.exists("prompt_clone_tts.txt"):
    with open("prompt_clone_tts.txt", "r", encoding="utf-8") as f:
        master_prompt = f.read()
else:
    master_prompt = "Você é um assistente."

# ============================================================================
# FERRAMENTA DE BUSCA (RAG) - Igual brain.py
# ============================================================================

vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 60, "lambda_mult": 0.3}
)

@tool
def pesquisar_memoria_renan(query: str) -> str:
    """Busca trechos de lives e pensamentos do Renan Santos sobre um tema."""
    docs = retriever.invoke(query)
    
    print(f"\n[DEBUG RAG] Query: {query} | Docs encontrados: {len(docs)}")
    
    resultado = []
    for i, doc in enumerate(docs, 1):
        trecho = doc.page_content  
        url = doc.metadata.get('url', 'URL não disponível')
        fonte = doc.metadata.get('fonte', 'Fonte desconhecida')
        
        fonte_limpa = fonte.replace('.pt.srt', '').replace('.srt', '')
        
        resultado.append(
            f"Fonte {i} ({fonte_limpa}):\n"
            f'"{trecho}..."\n'
            f"Link com timestamp: {url}"
        )
    
    return "\n\n".join(resultado)

# ============================================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    model_kwargs={"system_instruction": master_prompt}
).bind_tools([pesquisar_memoria_renan])

# ============================================================================
# LÓGICA DO GRAFO COM TTS (NOVO)
# ============================================================================

class RenanBrainWithTTS:
    """
    Brain do Renan com suporte a TTS
    Encapsula LangGraph + TTS em uma classe configurável
    """
    
    def __init__(self, 
                 enable_tts: bool = False,
                 tts_output_device: Optional[str] = "CABLE Input",
                 tts_monitor: bool = True,
                 tts_voice: str = "pt-BR-AntonioNeural"):
        
        self.enable_tts = enable_tts
        self.tts = None
        
        # Inicializa TTS se habilitado
        if enable_tts:
            print("🎙️ Inicializando TTS...")
            self.tts = RenanTTS(
                voice_sample_path=None,  # Usando Edge-TTS, não precisa de sample
                output_device_name=tts_output_device
            )
            # Se você quiser usar Edge-TTS do tts_streamlit.py:
            # from tts_streamlit import get_tts
            # self.tts = get_tts(tts_output_device, tts_monitor, tts_voice)
            print("✓ TTS pronto")
        
        # Cria o grafo
        self._build_graph()
    
    def _chatbot_node(self, state: MessagesState):
        """Nó do chatbot - processa mensagens"""
        messages = [SystemMessage(content=master_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    def _build_graph(self):
        """Constrói o grafo LangGraph"""
        workflow = StateGraph(MessagesState)
        
        # Adiciona nós
        workflow.add_node("chatbot", self._chatbot_node)
        workflow.add_node("tools", ToolNode([pesquisar_memoria_renan]))
        
        # Adiciona edges
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", tools_condition)
        workflow.add_edge("tools", "chatbot")
        
        # Compila
        self.agent = workflow.compile()
    
    def chat(self, user_message: str, speak: bool = None) -> str:
        """
        Processa uma mensagem do usuário
        
        Args:
            user_message: Mensagem do usuário
            speak: Se deve falar a resposta (None = usa self.enable_tts)
        
        Returns:
            Resposta do assistente (texto)
        """
        from langchain_core.messages import HumanMessage
        
        # Decide se fala ou não
        should_speak = speak if speak is not None else self.enable_tts
        
        # Cria input para o agente
        inputs = {"messages": [HumanMessage(content=user_message)]}
        
        print(f"\n👤 Você: {user_message}")
        print("🤖 Renan: ", end="", flush=True)
        
        # Processa com streaming
        full_response = ""
        
        for chunk in self.agent.stream(inputs):
            for node, values in chunk.items():
                if node == "chatbot":
                    msg = values["messages"][-1]
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content if isinstance(msg.content, str) else msg.content[0].get('text', '')
                        full_response = content
                        print(content, end="", flush=True)
        
        print()  # Nova linha após resposta
        
        # Fala a resposta se TTS estiver habilitado
        if should_speak and self.tts and full_response:
            print("🗣️ Falando resposta...")
            self.tts.speak(full_response)
            print("✓ Áudio concluído")
        
        return full_response
    
    def chat_session(self):
        """
        Inicia sessão de chat interativa no terminal
        """
        from langchain_core.messages import HumanMessage, AIMessage
        
        messages = []
        
        print("\n" + "="*60)
        print("🧠 RENAN SANTOS AI - CHAT COM TTS")
        print("="*60)
        
        if self.enable_tts:
            print("🎙️ TTS: ATIVADO")
        else:
            print("🔇 TTS: DESATIVADO (use enable_tts=True)")
        
        print("\nComandos:")
        print("  'sair' ou 'exit' - Encerra")
        print("  'falar' - Ativa/desativa TTS")
        print("  'limpar' - Limpa histórico")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("👤 Você: ").strip()
                
                if not user_input:
                    continue
                
                # Comandos especiais
                if user_input.lower() in ['sair', 'exit', 'quit']:
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
                
                # Adiciona mensagem do usuário
                messages.append(HumanMessage(content=user_input))
                
                # Processa com o agente
                inputs = {"messages": messages}
                
                print("🤖 Renan: ", end="", flush=True)
                
                full_response = ""
                for chunk in self.agent.stream(inputs):
                    for node, values in chunk.items():
                        if node == "chatbot":
                            msg = values["messages"][-1]
                            if hasattr(msg, 'content') and msg.content:
                                content = msg.content if isinstance(msg.content, str) else msg.content[0].get('text', '')
                                full_response = content
                                print(content, end="", flush=True)
                
                print()  # Nova linha
                
                # Adiciona resposta ao histórico
                messages.append(AIMessage(content=full_response))
                
                # Fala se TTS estiver ativado
                if self.enable_tts and self.tts and full_response:
                    print("🗣️ Falando...")
                    self.tts.speak(full_response)
                    print("✓ Concluído\n")
                else:
                    print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrompido pelo usuário. Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}\n")


# ============================================================================
# INSTÂNCIA GLOBAL (para importar em outros scripts)
# ============================================================================

# Sem TTS por padrão (compatibilidade com brain.py original)
agent = StateGraph(MessagesState)
agent.add_node("chatbot", lambda state: {
    "messages": [llm.invoke([SystemMessage(content=master_prompt)] + state["messages"])]
})
agent.add_node("tools", ToolNode([pesquisar_memoria_renan]))
agent.add_edge(START, "chatbot")
agent.add_conditional_edges("chatbot", tools_condition)
agent.add_edge("tools", "chatbot")
agent = agent.compile()

# ============================================================================
# TESTE DIRETO
# ============================================================================

if __name__ == "__main__":
    # Cria brain com TTS
    brain = RenanBrainWithTTS(
        enable_tts=True,  # Mude para False para desabilitar TTS
        tts_output_device="CABLE Input",  # Nome do seu cabo virtual
        tts_monitor=True  # True = ouve nos speakers também
    )
    
    # Inicia sessão interativa
    brain.chat_session()