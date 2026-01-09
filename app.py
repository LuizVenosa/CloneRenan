import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from networkx.readwrite import json_graph
from datetime import datetime
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain_core.messages import HumanMessage, AIMessage
from brain import agent

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

st.set_page_config(
    page_title="Renan Santos AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BLACK & YELLOW THEME - Matching partidomissao.com
st.markdown("""
    <style>
    /* Global App Styling */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 100%);
        margin-top: -50px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #000000 0%, #2d2d2d 100%);
        border-right: 2px solid #FFD700;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #FFD700 !important;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
        font-weight: 800;
    }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.6);
    }
    
    /* Primary Buttons (Active) */
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFED4E 100%);
        box-shadow: 0 6px 25px rgba(255, 215, 0, 0.7);
    }
    
    /* Main Title */
    h1 {
        color: #FFD700 !important;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.6);
        font-weight: 900;
        letter-spacing: 1px;
    }
    
    h2, h3 {
        color: #FFFFFF !important;
        border-bottom: 2px solid #FFD700;
        padding-bottom: 10px;
    }
    
    /* Text Styling */
    p, .stMarkdown, .stText {
        color: #FFFFFF !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        color: #FFD700 !important;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #FFD700;
        box-shadow: 0 4px 20px rgba(255, 215, 0, 0.3);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border-left: 4px solid #FFD700;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    /* Chat Input */
 
    
    .stChatInput textarea {
        background: #1a1a1a !important;
        color: #FFFFFF !important;
        border: 2px solid #FFD700 !important;
        border-radius: 8px;
    }
    
    .stChatInput textarea:focus {
        border-color: #FFED4E !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.5);
    }
    
    /* Hide chat input on non-chat pages */
    [data-testid="stChatInput"] {
        display: block;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(0, 0, 0, 0.3);
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #FFFFFF;
        border: 2px solid #444;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000000;
        border-color: #FFD700;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.5);
    }
    
    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        background: #2d2d2d;
    }
    
    .stSlider [role="slider"] {
        background: #FFD700 !important;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #FFFFFF !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: #FFD700 !important;
        border: 1px solid #FFD700;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.4);
        border: 1px solid #FFD700;
        border-top: none;
        color: #FFFFFF;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 2px solid #FFD700;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .stDataFrame thead tr th {
        background: #FFD700 !important;
        color: #000000 !important;
        font-weight: 700;
    }
    
    .stDataFrame tbody tr {
        background: rgba(255, 255, 255, 0.05);
        color: #FFFFFF;
    }
    
    .stDataFrame tbody tr:hover {
        background: rgba(255, 215, 0, 0.1);
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: rgba(0, 0, 0, 0.6);
        border-left: 4px solid #FFD700;
        color: #FFFFFF;
    }
    
    /* Horizontal rule */
    hr {
        border-color: #FFD700 !important;
        opacity: 0.5;
    }
    
    /* Plotly charts dark mode compatibility */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #FFD700;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #FFED4E;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_data(ttl=300)
def carregar_analise(caminho="analise_topicos.json"):
    """Carrega análise com validação"""
    if not os.path.exists(caminho):
        return None, f"Arquivo não encontrado: {caminho}"
    
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        if 'temas' not in dados or 'grafo' not in dados:
            return None, "Estrutura inválida"
        
        dados['_loaded_at'] = datetime.now().isoformat()
        return dados, None
        
    except Exception as e:
        return None, f"Erro: {e}"

def formatar_numero(num):
    """Formata números"""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    return str(num)

def gerar_wordcloud(texto_dict, max_words=50):
    """Gera word cloud com tema amarelo/preto"""
    wc = WordCloud(
        width=800,
        height=400,
        background_color='#000000',
        colormap='YlOrBr',  # Yellow-Orange-Brown colormap
        max_words=max_words,
        relative_scaling=0.5,
        prefer_horizontal=0.7
    ).generate_from_frequencies(texto_dict)
    
    return wc

# ============================================================================
# NAVEGAÇÃO
# ============================================================================

if "page" not in st.session_state:
    st.session_state.page = "chat"

with st.sidebar:
    st.title("🧠 Renan Santos AI")
    st.markdown("---")
    
    if st.button("💬 Chat", use_container_width=True, 
                 type="primary" if st.session_state.page == "chat" else "secondary"):
        st.session_state.page = "chat"
        st.rerun()
    
    if st.button("📊 Dashboard", use_container_width=True, 
                 type="primary" if st.session_state.page == "analise" else "secondary"):
        st.session_state.page = "analise"
        st.rerun()
 

# ============================================================================
# PÁGINA: CHAT
# ============================================================================

if st.session_state.page == "chat":
    st.title("💬 Chat com o Renan Clone")
    chat_container = st.container()
    
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)

        if prompt := st.chat_input("Converse com o RenanAI"):
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

# ============================================================================
# PÁGINA: ANÁLISE
# ============================================================================

elif st.session_state.page == "analise":
    
    dados, erro = carregar_analise()
    
    if erro:
        st.error(f"❌ {erro}")
        st.info("Execute: `python analyzer_backend.py`")
        st.stop()
    
    # HEADER COM MÉTRICAS
    st.title("📊 Dashboard de Análise de Tópicos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📺 Lives", dados['metadados']['documentos_processados'])
    with col2:
        st.metric("🎯 Temas", len(dados['temas']['temas_finais']))
    with col3:
        total_mencoes = sum(f for _, f in dados['temas']['ranking'])
        st.metric("💬 Menções", formatar_numero(total_mencoes))
    with col4:
        G_data = json_graph.node_link_graph(dados['grafo'])
        st.metric("🔗 Conexões", G_data.number_of_edges())
    
    st.markdown("---")
    
    # TABS PRINCIPAIS
    tabs = ["📈 Ranking", "🕸️ Rede"]
    
    if dados['metadados'].get('tem_dados_temporais'):
        tabs.append("📅 Timeline")
    
    tab_refs = st.tabs(tabs)
    
    # ========================================================================
    # TAB 1: RANKING
    # ========================================================================
    
    with tab_refs[0]:
        st.subheader("🏆 Top Temas por Frequência")
        
        top_n = st.slider("Quantos temas?", 5, 50, 20, 5)
        
        ranking_data = dados['temas']['ranking'][:top_n]
        df_ranking = pd.DataFrame(ranking_data, columns=['Tema', 'Menções'])
        
        # Chart with dark theme and yellow colors
        fig = px.bar(
            df_ranking,
            x='Menções',
            y='Tema',
            orientation='h',
            color='Menções',
            color_continuous_scale=['#FFD700', '#FFA500', '#FF8C00'],
            title=f"Top {top_n} Temas"
        )
        
        fig.update_layout(
            height=max(400, top_n * 25),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            title_font=dict(color='#FFD700', size=20),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # WORD CLOUD
        st.markdown("---")
        st.subheader("☁️ Word Cloud dos Temas")
        
        freq_dict = {tema: freq for tema, freq in ranking_data}
        wc = gerar_wordcloud(freq_dict)
        
        fig_wc, ax = plt.subplots(figsize=(10, 5), facecolor='#000000')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_facecolor('#000000')
        st.pyplot(fig_wc)

    # ========================================================================
    # TAB 2: REDE
    # ========================================================================
    
        with tab_refs[1]:
            st.subheader("🕸️ Grafo de Co-ocorrências")
            
            st.info("""
            **💡 Dicas de uso:**
            - **Frequência mínima**: Mostra apenas temas com N+ menções
            - **Peso mínimo**: Mostra apenas conexões com N+ co-ocorrências
            - **Máx. nós**: Limita quantidade de nós (melhor performance)
            - Para grafos grandes, aumente os filtros para melhor visualização
            """)
            
            # Performance warning
            G_original = json_graph.node_link_graph(dados['grafo'])
            total_nodes = G_original.number_of_nodes()
            
            col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
            
            with col_ctrl1:
                min_freq = st.slider("Frequência mínima", 1, 50, 10)
            with col_ctrl2:
                min_weight = st.slider("Peso mínimo", 1, 15, 5)
            with col_ctrl3:
                max_nodes = st.slider("Máx. nós", 10, 100, 50, 5)
            with col_ctrl4:
                mostrar_comunidades = st.checkbox("Colorir por comunidade", value=True)
            
            # Filter graph with loading state
            with st.spinner("🔄 Processando grafo..."):
                G = json_graph.node_link_graph(dados['grafo'])
                
                # Get top nodes by frequency
                node_freq = [(n, data.get('frequencia', 0)) for n, data in G.nodes(data=True)]
                node_freq_sorted = sorted(node_freq, key=lambda x: x[1], reverse=True)
                top_nodes = [n for n, _ in node_freq_sorted[:max_nodes]]
                
                # Filter by frequency first
                nos_validos = [
                    n for n in top_nodes
                    if G.nodes[n].get('frequencia', 0) >= min_freq
                ]
                
                if len(nos_validos) == 0:
                    st.error("❌ Nenhum nó atende aos critérios. Reduza os filtros.")
                    st.stop()
                
                G_filtrado = G.subgraph(nos_validos).copy()
                
                # Filter edges by weight
                arestas_remover = [
                    (u, v) for u, v, data in G_filtrado.edges(data=True)
                    if data.get('weight', 0) < min_weight
                ]
                G_filtrado.remove_edges_from(arestas_remover)
                G_filtrado.remove_nodes_from(list(nx.isolates(G_filtrado)))
            
            if G_filtrado.number_of_nodes() == 0:
                st.warning("⚠️ Nenhum nó conectado. Ajuste os filtros.")
                st.stop()
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            densidade = nx.density(G_filtrado)
            componentes = nx.number_connected_components(G_filtrado)
            
            with col_m1:
                st.metric("Nós", f"{G_filtrado.number_of_nodes()}/{total_nodes}")
            with col_m2:
                st.metric("Arestas", G_filtrado.number_of_edges())
            with col_m3:
                st.metric("Densidade", f"{densidade:.3f}")
            with col_m4:
                st.metric("Componentes", componentes)
            
            # Optimize layout computation
            with st.spinner("🎨 Gerando layout..."):
                # Use simpler layout for large graphs
                if G_filtrado.number_of_nodes() > 30:
                    pos = nx.spring_layout(G_filtrado, k=2, iterations=30, seed=42)
                else:
                    pos = nx.spring_layout(G_filtrado, k=1.5, iterations=50, seed=42)
                
                centralidade = nx.degree_centrality(G_filtrado)
            
            # Color nodes by community or default
            if mostrar_comunidades and 'comunidades' in dados:
                comunidades_map = dados['comunidades'].get('mapeamento', {})
                cores = {}
                palette = ['#FFD700', '#FFA500', '#FF8C00', '#FFED4E', '#FFB347']
                
                for no in G_filtrado.nodes():
                    com_id = comunidades_map.get(no, 0)
                    cores[no] = palette[com_id % len(palette)]
            else:
                cores = {no: '#FFD700' for no in G_filtrado.nodes()}
            
            # Build graph visualization
            with st.spinner("📊 Renderizando visualização..."):
                # Create edge traces (simplified for performance)
                edge_x, edge_y = [], []
                
                for edge in G_filtrado.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(width=0.5, color='rgba(255,215,0,0.2)'),
                    hoverinfo='none',
                    showlegend=False
                )
                
                # Create node trace
                node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
                
                for node in G_filtrado.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    
                    freq = G_filtrado.nodes[node].get('frequencia', 1)
                    cent = centralidade[node]
                    
                    node_text.append(f"{node}<br>Menções: {freq}<br>Centralidade: {cent:.3f}")
                    node_size.append(10 + np.log1p(freq) * 3)
                    node_color.append(cores[node])
                
                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    text=[n[:20] + '...' if len(n) > 20 else n for n in G_filtrado.nodes()],
                    textposition="top center",
                    textfont=dict(size=8, color='#FFFFFF'),
                    hovertext=node_text,
                    hoverinfo='text',
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        line=dict(width=1.5, color='#000000')
                    ),
                    showlegend=False
                )
                
                fig_graph = go.Figure(data=[edge_trace, node_trace])
                
                fig_graph.update_layout(
                    title="Rede de Tópicos",
                    showlegend=False,
                    height=600,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF'),
                    title_font=dict(color='#FFD700', size=20),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_graph, use_container_width=True)
            
            # Communities section
            if 'comunidades' in dados and dados['comunidades']['total'] > 0:
                st.markdown("---")
                st.subheader("🎯 Comunidades Identificadas")
                
                comunidades_map = dados['comunidades']['mapeamento']
                comunidades_dict = {}
                
                for no, com_id in comunidades_map.items():
                    if no in G_filtrado.nodes():
                        if com_id not in comunidades_dict:
                            comunidades_dict[com_id] = []
                        comunidades_dict[com_id].append(no)
                
                for com_id, membros in sorted(comunidades_dict.items()):
                    with st.expander(f"🔸 Comunidade {com_id + 1} ({len(membros)} temas)"):
                        st.write(", ".join(sorted(membros)))