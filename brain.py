import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --- CONFIGURAÇÃO DE AMBIENTE ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    encode_kwargs={"normalize_embeddings": True}
)
DB_PATH = "./db_clone"

# Carregar Master Prompt
if os.path.exists("prompt_clone.txt"):
    with open("prompt_clone.txt", "r", encoding="utf-8") as f:
        master_prompt = f.read()
else:
    master_prompt = "Você é um assistente."



# --- FERRAMENTA DE BUSCA (RAG) COM LINKS ---
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 60, "lambda_mult": 0.4}
)
@tool
def pesquisar_memoria_renan(query: str) -> str:
    """Busca trechos de lives e pensamentos do Renan Santos sobre um tema."""
    docs = retriever.invoke(query)
    
    print(f"\n[DEBUG RAG] Query: {query} | Docs encontrados: {len(docs)}")
    
    # Formata com links clicáveis
    resultado = []
    for i, doc in enumerate(docs, 1):
        trecho = doc.page_content  
        url = doc.metadata.get('url', 'URL não disponível')
        fonte = doc.metadata.get('fonte', 'Fonte desconhecida')
        
        # Remove extensão .srt do nome
        fonte_limpa = fonte.replace('.pt.srt', '').replace('.srt', '')
        
        resultado.append(
            f"Fonte {i} ({fonte_limpa}):\n"
            f'"{trecho}..."\n'
            f"Link com timestamp: {url}"
        )
    print("\n\n".join(resultado))
    return "\n\n".join(resultado)

# --- CONFIGURAÇÃO DO MODELO ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    model_kwargs={"system_instruction": master_prompt}
).bind_tools([pesquisar_memoria_renan])

# --- LÓGICA DO GRAFO ---
def chatbot_renan(state: MessagesState):
    messages = [SystemMessage(content=master_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("chatbot", chatbot_renan)
workflow.add_node("tools", ToolNode([pesquisar_memoria_renan]))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

# Exporta o agente compilado
agent = workflow.compile()