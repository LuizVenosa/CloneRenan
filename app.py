import streamlit as st
import json
import os
import pandas as pd
from langchain_core.messages import HumanMessage, AIMessage
from brain import agent  
from streamlit_agraph import agraph, Node, Edge, Config

# Configuração da página para ocupar o espaço total
st.set_page_config(page_title="Renan Santos AI", layout="wide", initial_sidebar_state="collapsed")

# CSS personalizado para remover margens e estilizar botões
st.markdown("""
    <style>
    .stApp { margin-top: -50px; }
    .nav-button {
        display: inline-block;
        padding: 0.5em 1.5em;
        text-decoration: none;
        border-radius: 10px;
        transition: 0.3s;
    }
    </style>
""", unsafe_allow_html=True)

# --- GERENCIAMENTO DE NAVEGAÇÃO ---
if "page" not in st.session_state:
    st.session_state.page = "chat"

def change_page(name):
    st.session_state.page = name

# Cabeçalho de Navegação Intuitivo
col_nav1, col_nav2 = st.columns([8, 2])
with col_nav1:
    st.title("🧠 Clone Renan Santos")
with col_nav2:
    if st.session_state.page == "chat":
        if st.button("📊 Ver Análises ➔", use_container_width=True):
            change_page("analise")
            st.rerun()
    else:
        if st.button("⬅ Voltar ao Chat", use_container_width=True):
            change_page("chat")
            st.rerun()

st.divider()

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def carregar_dados():
    if os.path.exists("analise_completa.json"):
        with open("analise_completa.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

dados = carregar_dados()

# ==================== PÁGINA 1: CHAT ====================
if st.session_state.page == "chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Container de mensagens
    for message in st.session_state.messages:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(message.content)

    if prompt := st.chat_input("Pergunte ao Renan sobre estética, política ou decadência..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            inputs = {"messages": st.session_state.messages}
            
            for chunk in agent.stream(inputs):
                for node, values in chunk.items():
                    if node == "chatbot":
                        msg = values["messages"][-1]
                        if hasattr(msg, 'content') and msg.content:
                            full_response = msg.content if isinstance(msg.content, str) else msg.content[0].get('text', '')
                            placeholder.markdown(full_response)
            
            st.session_state.messages.append(AIMessage(content=full_response))

# ==================== PÁGINA 2: ANÁLISE ====================
elif st.session_state.page == "analise":
    tab_stats, tab_grafo = st.tabs(["📊 Estatísticas e Correlações", "🕸️ Grafo de Conexões Inteligente"])

    with tab_stats:
        st.subheader("Ranking de Relevância")
        df = pd.DataFrame(dados['stats']['ranking'], columns=['Entidade/Tema', 'Menções'])
        st.bar_chart(df.set_index('Entidade/Tema').head(15))
        
        st.divider()
        
        st.subheader("🔗 Correlações de Temas Naturais")
        # Mostra quais temas aparecem mais vezes juntos (edges com maior peso)
        g_data = dados.get('grafo', {})
        edges_raw = g_data.get('edges', [])
        
        correlacoes = sorted(edges_raw, key=lambda x: x.get('weight', 0), reverse=True)
        
        cols = st.columns(3)
        for i, corr in enumerate(correlacoes[:9]): # Top 9 correlações
            with cols[i % 3]:
                st.info(f"**{corr['source']}** ↔️ **{corr['target']}**\n\nFrequência: {corr['weight']}")

    with tab_grafo:
        st.subheader("Mapa Mental de Influência")
        st.caption("Nós maiores indicam temas/pessoas discutidos com mais frequência (mínimo 5 menções).")
        
        g_data = dados.get('grafo', {})
        nodes_raw = g_data.get('nodes', [])
        edges_raw = g_data.get('edges', [])
        
        # Estilização dinâmica
        nodes = [
            Node(
                id=n['id'], 
                label=n['id'], 
                # Tamanho proporcional às menções (mencoes calculadas no backend)
                size=10 + (n.get('mencoes', 5) * 1.5), 
                color="#FF4B4B" if n['id'].isupper() else "#4ECDC4",
                font={'size': 12, 'color': 'white'}
            ) for n in nodes_raw
        ]
        
        edges = [Edge(source=e['source'], target=e['target'], width=e.get('weight', 1)) for e in edges_raw]
        
        if nodes:
            # CONFIGURAÇÃO DE FÍSICA PARA PARAR DE MOVER
            config = Config(
                width=1000, 
                height=800, 
                directed=False,
                nodeHighlightBehavior=True, 
                collapsible=False,
                physics={
                    "enabled": True,
                    "stabilization": {"iterations": 200, "updateInterval": 10},
                    "barnesHut": {"gravitationalConstant": -15000, "centralGravity": 0.1, "springLength": 100}
                }
            )
            agraph(nodes=nodes, edges=edges, config=config)