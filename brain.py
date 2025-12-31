import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --- CONFIGURAÇÃO DE AMBIENTE ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
DB_PATH = "./db_clone"

# Carregar Master Prompt
if os.path.exists("prompt_clone.txt"):
    with open("prompt_clone.txt", "r", encoding="utf-8") as f:
        master_prompt = f.read()
else:
    master_prompt = "Você é um assistente."

# --- FERRAMENTA DE BUSCA (RAG) ---
vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

@tool
def pesquisar_memoria_renan(query: str) -> str:
    """Busca trechos de colunas e pensamentos do Renan Santos sobre um tema."""
    docs = retriever.invoke(query)
    
    # --- DIAGNÓSTICO ---
    print(f"\n[DEBUG RAG] Query de busca: {query}")
    print(f"[DEBUG RAG] Documentos encontrados: {len(docs)}")
    for i, d in enumerate(docs):
        print(f"  - Trecho {i+1}: {d.page_content[:100]}...")
    # -------------------

    return "\n\n".join([f"[Trecho]: {d.page_content}\n[Link]: {d.metadata.get('url', '')}" for d in docs])
# --- CONFIGURAÇÃO DO MODELO ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    model_kwargs={"system_instruction": master_prompt}
).bind_tools([pesquisar_memoria_renan])

# --- LÓGICA DO GRAFO ---
def chatbot_renan(state: MessagesState):
    # Reforçamos o prompt de sistema em cada chamada
    messages = [SystemMessage(content=master_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("chatbot", chatbot_renan)
workflow.add_node("tools", ToolNode([pesquisar_memoria_renan]))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges("chatbot", tools_condition)
workflow.add_edge("tools", "chatbot")

# Exportamos o agente compilado
agent = workflow.compile()