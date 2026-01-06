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

st.markdown("""
    <style>
    .stApp { margin-top: -50px; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
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
    """Gera word cloud a partir de dicionário {palavra: frequência}"""
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=max_words,
        relative_scaling=0.5
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
        st.metric("📁 Lives", dados['metadados']['documentos_processados'])
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
    tabs = ["📈 Ranking", "🕸️ Rede", "🧠 Semântica", "📄 Por Live"]
    
    # Adiciona tab temporal se houver dados de data
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
        
        fig = px.bar(
            df_ranking,
            x='Menções',
            y='Tema',
            orientation='h',
            color='Menções',
            color_continuous_scale='Viridis',
            title=f"Top {top_n} Temas"
        )
        
        fig.update_layout(
            height=max(400, top_n * 25),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # WORD CLOUD
        st.markdown("---")
        st.subheader("☁️ Word Cloud dos Temas")
        
        freq_dict = {tema: freq for tema, freq in ranking_data}
        wc = gerar_wordcloud(freq_dict)
        
        fig_wc, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)

    # ========================================================================
    # TAB 2: REDE (MELHORADA)
    # ========================================================================
    
    with tab_refs[1]:
        st.subheader("🕸️ Grafo de Co-ocorrências")
        
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            min_freq = st.slider("Frequência mínima", 1, 20, 5)
        with col_ctrl2:
            min_weight = st.slider("Peso mínimo", 1, 10, 3)
        with col_ctrl3:
            mostrar_comunidades = st.checkbox("Colorir por comunidade", value=True)
        
        # Reconstrói grafo
        G = json_graph.node_link_graph(dados['grafo'])
        
        # Filtros
        nos_validos = [
            n for n, data in G.nodes(data=True)
            if data.get('frequencia', 0) >= min_freq
        ]
        G_filtrado = G.subgraph(nos_validos).copy()
        
        arestas_remover = [
            (u, v) for u, v, data in G_filtrado.edges(data=True)
            if data.get('weight', 0) < min_weight
        ]
        G_filtrado.remove_edges_from(arestas_remover)
        G_filtrado.remove_nodes_from(list(nx.isolates(G_filtrado)))
        
        if G_filtrado.number_of_nodes() == 0:
            st.warning("⚠️ Ajuste os filtros")
        else:
            # Métricas
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            densidade = nx.density(G_filtrado)
            componentes = nx.number_connected_components(G_filtrado)
            
            with col_m1:
                st.metric("Nós", G_filtrado.number_of_nodes())
            with col_m2:
                st.metric("Arestas", G_filtrado.number_of_edges())
            with col_m3:
                st.metric("Densidade", f"{densidade:.3f}")
            with col_m4:
                st.metric("Componentes", componentes)
            
            # Layout
            pos = nx.spring_layout(G_filtrado, k=1, iterations=50)
            centralidade = nx.degree_centrality(G_filtrado)
            
            # Cores por comunidade
            if mostrar_comunidades and 'comunidades' in dados:
                comunidades_map = dados['comunidades'].get('mapeamento', {})
                cores = {}
                palette = px.colors.qualitative.Plotly
                
                for no in G_filtrado.nodes():
                    com_id = comunidades_map.get(no, 0)
                    cores[no] = palette[com_id % len(palette)]
            else:
                cores = {no: '#636EFA' for no in G_filtrado.nodes()}
            
            # Visualização
            edge_trace = []
            for edge in G_filtrado.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                
                edge_trace.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
            
            node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
            
            for node in G_filtrado.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                freq = G_filtrado.nodes[node].get('frequencia', 1)
                cent = centralidade[node]
                
                node_text.append(f"{node}<br>Menções: {freq}<br>Centralidade: {cent:.3f}")
                node_size.append(10+ np.log(freq))
                node_color.append(cores[node])
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[n for n in G_filtrado.nodes()],
                textposition="top center",
                textfont=dict(size=9),
                hovertext=node_text,
                hoverinfo='text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=2, color='white')
                ),
                showlegend=False
            )
            
            fig_graph = go.Figure(data=edge_trace + [node_trace])
            
            fig_graph.update_layout(
                title="Rede de Tópicos",
                showlegend=False,
                height=700,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig_graph, use_container_width=True)
            
            # Análise de comunidades
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
    
    # ========================================================================
    # TAB 3: ANÁLISE SEMÂNTICA (COMPLETA AGORA!)
    # ========================================================================
    
    with tab_refs[2]:
        st.subheader("🧠 Mapa de Similaridade Semântica")
        
        if 'analise_semantica' in dados and dados['analise_semantica']['matriz_similaridade']:
            matriz = np.array(dados['analise_semantica']['matriz_similaridade'])
            temas_ordem = dados['analise_semantica']['temas_ordem']
            
            # Limita a 20 temas para visualização
            if len(temas_ordem) > 20:
                st.info(f"📊 Mostrando top 20 de {len(temas_ordem)} temas")
                temas_ordem = temas_ordem[:20]
                matriz = matriz[:20, :20]
            
            # Heatmap
            fig_heat = go.Figure(data=go.Heatmap(
                z=matriz,
                x=temas_ordem,
                y=temas_ordem,
                colorscale='RdYlGn',
                text=np.round(matriz, 2),
                texttemplate='%{text}',
                textfont={"size": 8},
                colorbar=dict(title="Similaridade")
            ))
            
            fig_heat.update_layout(
                title="Mapa de Calor - Similaridade entre Temas",
                height=700,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Encontra pares mais similares
            st.markdown("---")
            st.subheader("🔗 Temas Mais Relacionados Semanticamente")
            
            pares_similares = []
            for i in range(len(temas_ordem)):
                for j in range(i+1, len(temas_ordem)):
                    if matriz[i][j] > 0.5:  # Threshold de similaridade
                        pares_similares.append({
                            'Tema 1': temas_ordem[i],
                            'Tema 2': temas_ordem[j],
                            'Similaridade': round(matriz[i][j], 3)
                        })
            
            if pares_similares:
                df_pares = pd.DataFrame(pares_similares).sort_values('Similaridade', ascending=False)
                st.dataframe(df_pares.head(15), use_container_width=True, hide_index=True)
            else:
                st.info("Nenhum par com similaridade > 0.5")
        else:
            st.warning("⚠️ Dados de similaridade não disponíveis")
   
