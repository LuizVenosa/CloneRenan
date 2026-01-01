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
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage
from brain import agent

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Análise de Lives - Renan Santos AI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Análise de Tópicos em Transcrições de Lives"
    }
)

# CSS melhorado
st.markdown("""
    <style>
    .stApp { margin-top: -50px; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_data(ttl=300)  # Cache por 5 minutos
def carregar_analise(caminho="analise_topicos.json"):
    """
    Carrega análise com suporte a múltiplos formatos e tratamento de erros.
    """
    if not os.path.exists(caminho):
        return None, f"Arquivo não encontrado: {caminho}"
    
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            dados = json.load(f)
        
        # Valida estrutura mínima
        if 'temas' not in dados or 'grafo' not in dados:
            return None, "Estrutura de dados inválida"
        
        # Adiciona timestamp de carregamento
        dados['_loaded_at'] = datetime.now().isoformat()
        
        return dados, None
        
    except json.JSONDecodeError as e:
        return None, f"Erro ao decodificar JSON: {e}"
    except Exception as e:
        return None, f"Erro inesperado: {e}"

def formatar_numero(num):
    """Formata números grandes de forma legível"""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    return str(num)

def calcular_metricas_grafo(G):
    """Calcula métricas de análise de rede"""
    if G.number_of_nodes() == 0:
        return {}
    
    return {
        'densidade': nx.density(G),
        'componentes': nx.number_connected_components(G),
        'diametro': nx.diameter(G) if nx.is_connected(G) else None,
        'avg_clustering': nx.average_clustering(G)
    }

# ============================================================================
# NAVEGAÇÃO
# ============================================================================

if "page" not in st.session_state:
    st.session_state.page = "analise"

# Sidebar com informações e controles
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=RS", width=150)
    st.title("🧠 Renan Santos AI")
    st.markdown("---")
    
    # Navegação
    st.subheader("Navegação")
    if st.button("💬 Chat com Clone", use_container_width=True,  type="primary" if st.session_state.page == "chat" else "secondary"):
        st.session_state.page = "chat"
        st.rerun()
    
    if st.button("📊 Dashboard de Análises", use_container_width=True, 
                 type="primary" if st.session_state.page == "analise" else "secondary"):
        st.session_state.page = "analise"
        st.rerun()
    
    st.markdown("---")
    
    # Informações do sistema
    if os.path.exists("analise_topicos.json"):
        stat = os.stat("analise_topicos.json")
        st.caption(f"📁 Última análise: {datetime.fromtimestamp(stat.st_mtime).strftime('%d/%m/%Y %H:%M')}")
        
        if st.button("🔄 Recarregar Dados", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    else:
        st.warning("⚠️ Nenhuma análise encontrada")
        st.info("Execute o script de análise primeiro:\n`python pipeline_analise.py`")

# ============================================================================
# PÁGINA: CHAT
# ============================================================================
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

# ============================================================================
# PÁGINA: ANÁLISE
# ============================================================================

elif st.session_state.page == "analise":
    
    # Carrega dados
    dados, erro = carregar_analise()
    
    if erro:
        st.error(f"❌ {erro}")
        st.stop()
    
    # Header com métricas gerais
    st.title("📊 Dashboard de Análise de Tópicos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_docs = dados['metadados']['documentos_processados']
        st.metric("📁 Lives Analisadas", total_docs)
    
    with col2:
        total_temas = len(dados['temas']['temas_finais'])
        st.metric("🎯 Temas Identificados", total_temas)
    
    with col3:
        total_mencoes = sum(freq for _, freq in dados['temas']['ranking'])
        st.metric("💬 Total de Menções", formatar_numero(total_mencoes))
    
    with col4:
        G_data = json_graph.node_link_graph(dados['grafo'])
        st.metric("🔗 Conexões no Grafo", G_data.number_of_edges())
    
    st.markdown("---")
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs([
        "📈 Ranking e Frequência",
        "🕸️ Rede de Relações",
        "🧠 Análise Semântica"
    ])
    
    # ========================================================================
    # TAB 1: RANKING E FREQUÊNCIA
    # ========================================================================
    
    with tab1:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("🏆 Top Temas por Frequência")
            
            # Controle de quantidade
            top_n = st.slider("Quantos temas exibir?", 5, 50, 20, 5)
            
            ranking_data = dados['temas']['ranking'][:top_n]
            df_ranking = pd.DataFrame(ranking_data, columns=['Tema', 'Menções'])
            
            # Gráfico de barras horizontal
            fig = px.bar(
                df_ranking,
                x='Menções',
                y='Tema',
                orientation='h',
                color='Menções',
                color_continuous_scale='Viridis',
                title=f"Top {top_n} Temas Mais Mencionados"
            )
            
            fig.update_layout(
                height=max(400, top_n * 25),
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.subheader("🎯 Temas em Destaque")
            
            temas_destaque = dados['temas']['temas_finais'][:10]
            
            for i, tema in enumerate(temas_destaque, 1):
                # Encontra frequência
                freq = next((f for t, f in ranking_data if t == tema), 0)
                st.markdown(f"**{i}.** {tema}")
                st.progress(freq / ranking_data[0][1] if ranking_data else 0)
                st.caption(f"{freq} menções")
            
            # Download do ranking completo
            st.markdown("---")
            csv = pd.DataFrame(
                dados['temas']['ranking'],
                columns=['Tema', 'Menções']
            ).to_csv(index=False)
            
            st.download_button(
                "📥 Baixar Ranking Completo (CSV)",
                csv,
                "ranking_temas.csv",
                "text/csv",
                use_container_width=True
            )
        
        # Distribuição de frequências
        st.markdown("---")
        st.subheader("📊 Distribuição de Frequências")
        
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            # Histograma
            frequencias = [freq for _, freq in dados['temas']['ranking']]
            
            fig_hist = px.histogram(
                x=frequencias,
                nbins=30,
                title="Distribuição de Menções",
                labels={'x': 'Número de Menções', 'y': 'Quantidade de Temas'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col_dist2:
            # Box plot
            fig_box = px.box(
                y=frequencias,
                title="Análise Estatística",
                labels={'y': 'Menções'}
            )
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # ========================================================================
    # TAB 2: REDE DE RELAÇÕES
    # ========================================================================
    
    with tab2:
        st.subheader("🕸️ Grafo de Co-ocorrências")
        
        # Controles
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
        
        with col_ctrl1:
            min_freq = st.slider(
                "Frequência mínima do nó",
                1, 20, 5,
                help="Exibe apenas tópicos com pelo menos N menções"
            )
        
        with col_ctrl2:
            min_weight = st.slider(
                "Peso mínimo da aresta",
                1, 10, 3,
                help="Exibe apenas conexões que ocorreram pelo menos N vezes"
            )
        
        with col_ctrl3:
            layout_type = st.selectbox(
                "Layout do grafo",
                ["spring", "kamada_kawai", "circular"],
                help="Algoritmo de posicionamento dos nós"
            )
        
        # Reconstrói grafo com filtros
        G = json_graph.node_link_graph(dados['grafo'])
        
        # Filtra nós por frequência
        nos_validos = [
            n for n, data in G.nodes(data=True)
            if data.get('frequencia', 0) >= 30
        ]
        G_filtrado = G.subgraph(nos_validos).copy()
        
        # Filtra arestas por peso
        arestas_remover = [
            (u, v) for u, v, data in G_filtrado.edges(data=True)
            if data.get('weight', 0) < 5
        ]
        G_filtrado.remove_edges_from(arestas_remover)
        
        # Remove nós isolados
        G_filtrado.remove_nodes_from(list(nx.isolates(G_filtrado)))
        
        if G_filtrado.number_of_nodes() == 0:
            st.warning("⚠️ Nenhum nó restante com os filtros atuais. Reduza os valores.")
        else:
            # Métricas do grafo
            col_m1, col_m3, col_m4 = st.columns(3)
            
            metricas = calcular_metricas_grafo(G_filtrado)
            
            with col_m1:
                st.metric("Nós", G_filtrado.number_of_nodes())
            with col_m3:
                st.metric("Densidade", f"{metricas.get('densidade', 0):.3f}")
            with col_m4:
                st.metric("Componentes", metricas.get('componentes', 0))
            
            # Calcula layout
            pos = nx.spring_layout(G_filtrado, k=1, iterations=50)

            
            # Calcula centralidades
            centralidade = nx.degree_centrality(G_filtrado)
            
            # Cria visualização com Plotly
            edge_trace = []
            
            for edge in G_filtrado.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = edge[2].get('weight', 1)
                
                edge_trace.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        showlegend=False
                    )
                )
            
            # Nós
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []
            
            for node in G_filtrado.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                freq = G_filtrado.nodes[node].get('frequencia', 1)
                cent = centralidade[node]
                
                node_text.append(f"{node}<br>Menções: {freq}<br>Centralidade: {cent:.3f}")
                node_size.append(12)  # Escala controlada
                node_color.append(cent)
            
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=[n for n in G_filtrado.nodes()],
                textposition="top center",
                textfont=dict(size=10),
                hovertext=node_text,
                hoverinfo='text',
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Centralidade"),
                    line=dict(width=2, color='white')
                )
            )
            
            fig_graph = go.Figure(data=edge_trace + [node_trace])
            
            fig_graph.update_layout(
                title="Rede de Co-ocorrências entre Tópicos",
                showlegend=False,
                hovermode='closest',
                height=700,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig_graph, use_container_width=True)
            
            # Top nós por centralidade
            st.markdown("---")
            st.subheader("🎯 Tópicos Mais Centrais na Rede")
            
            top_centrais = sorted(
                centralidade.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            df_centrais = pd.DataFrame(
                top_centrais,
                columns=['Tópico', 'Centralidade']
            )
            df_centrais['Centralidade'] = df_centrais['Centralidade'].round(3)
            
            st.dataframe(df_centrais, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 3: ANÁLISE SEMÂNTICA
    # ========================================================================
    
    with tab3:
        st.subheader("🧠 Mapa de Similaridade Semântica")
        st.info("Este heatmap mostra o quão semanticamente próximos os temas estão, baseado em embeddings BERT.")
        
        # Nota: Seu código original não salva a matriz BERT
        # Vou adicionar um exemplo de como deveria ser
        
        st.warning("⚠️ Funcionalidade em desenvolvimento - requer execução do pipeline melhorado")
        
        # Exemplo de implementação (comentado)
        st.code("""
        # No pipeline melhorado, adicione:
        def salvar_matriz_similaridade(temas_finais):
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            embeddings = model.encode(temas_finais)
            similaridades = util.cos_sim(embeddings, embeddings).cpu().numpy()
            return similaridades.tolist()
        
        # E salve no JSON final:
        resultado['analise_semantica'] = {
            'matriz_similaridade': salvar_matriz_similaridade(temas_finais),
            'temas_ordem': temas_finais
        }
        """, language='python')
    
   

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("🧠 Renan Santos AI - Análise de Tópicos | Desenvolvido com Streamlit")