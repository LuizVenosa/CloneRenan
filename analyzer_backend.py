import os
import json
import re
import spacy
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from networkx.readwrite import json_graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple, Dict

# ============================================================================
# CONFIGURAÇÃO E RECURSOS
# ============================================================================

# Lista expandida de stopwords para transcrições de lives
STOPWORDS_LIVES = {
    # Verbos genéricos
    'falar', 'querer', 'ficar', 'saber', 'achar', 'ir', 'dar', 'ter', 'fazer',
    'poder', 'dever', 'deixar', 'conseguir', 'começar', 'acabar', 'passar',
    
    # Marcadores discursivos e vícios de linguagem
    'vídeo', 'live', 'gente', 'pessoal', 'tá', 'né', 'coisa', 'aqui', 'cara',
    'tipo', 'assim', 'então', 'ai', 'galera', 'beleza', 'ok', 'certo',
    'olha', 'veja', 'entendeu', 'sabe', 'canal', 'inscrever', 'like',
    
    # Pronomes e artigos que podem vazar
    'ele', 'ela', 'isso', 'aquilo', 'esse', 'essa', 'outro', 'outra',
    
    # Números e temporais genéricos
    'hoje', 'agora', 'depois', 'antes', 'dia', 'vez', 'ano', 'hora'
}

def carregar_nlp():
    """Carrega modelo spaCy com componentes otimizados"""
    try:
        # Desabilita componentes pesados mas mantém NER e POS tagging
        nlp = spacy.load("pt_core_news_lg", disable=["parser", "attribute_ruler"])
        return nlp
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
        return spacy.load("pt_core_news_lg", disable=["parser", "attribute_ruler"])

# ============================================================================
# PROCESSAMENTO DE TEXTO
# ============================================================================

def processar_srt(caminho: str) -> str:
    """
    Lê arquivo SRT e remove metadados, mantendo apenas texto falado.
    Aplica limpeza básica de ruídos comuns em transcrições.
    """
    with open(caminho, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    # Remove numeração e timestamps
    texto = " ".join([
        l.strip() for l in linhas 
        if not l.strip().isdigit() 
        and "-->" not in l
        and l.strip()
    ])
    
    # Limpeza de ruídos comuns em transcrições automáticas
    texto = re.sub(r'\[.*?\]', '', texto)  # Remove [Música], [Aplausos]
    texto = re.sub(r'\(.*?\)', '', texto)  # Remove (risos), (tosse)
    texto = re.sub(r'\s+', ' ', texto)     # Normaliza espaços
    
    return texto.strip()

def segmentar_texto(texto: str, tamanho_janela: int = 5000) -> List[str]:
    """
    Divide texto em janelas sobrepostas para processar documentos grandes.
    SOLUÇÃO para o problema de truncamento do código original.
    """
    if len(texto) <= tamanho_janela:
        return [texto]
    
    segmentos = []
    overlap = tamanho_janela // 4  # 25% de sobreposição
    
    for i in range(0, len(texto), tamanho_janela - overlap):
        segmento = texto[i:i + tamanho_janela]
        if len(segmento) > 100:  # Ignora segmentos muito pequenos
            segmentos.append(segmento)
    
    return segmentos

def extrair_candidatos_tfidf(corpus_lematizado: List[str], 
                             min_df: int = 2, 
                             max_features: int = 150) -> Tuple[List[str], np.ndarray]:
    """
    Extrai candidatos a tópicos usando TF-IDF.
    MELHORIAS: min_df adequado, filtros de qualidade, bigramas focados.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),        # Apenas unigramas e bigramas
        max_features=max_features,
        min_df=min_df,             # Aparece em pelo menos N documentos
        max_df=0.8,                # Ignora termos muito frequentes (>80%)
        token_pattern=r'\b[a-záàâãéèêíïóôõöúçñ]{3,}\b'  # Mínimo 3 caracteres
    )
    
    try:
        matriz = vectorizer.fit_transform(corpus_lematizado)
        termos = vectorizer.get_feature_names_out()
        
        # Filtra termos de baixa qualidade
        termos_validos = [
            t.title() for t in termos
            if not t in STOPWORDS_LIVES
            and not t.isdigit()
            and len(t.split()) <= 2  # Máximo bigramas
        ]
        
        return termos_validos, matriz
        
    except ValueError as e:
        print(f"⚠️  Erro no TF-IDF: {e}")
        return [], None

def agrupar_temas_semanticamente(temas: List[str], 
                                  threshold: float = 0.75) -> List[str]:
    """
    Agrupa temas semanticamente similares usando embeddings.
    MELHORIAS: threshold ajustável, melhor modelo, deduplicação robusta.
    """
    if not temas or len(temas) < 2:
        return temas
    
    print(f"🧠 Refinando {len(temas)} candidatos com BERT...")
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(temas, convert_to_tensor=True, show_progress_bar=False)
    similaridades = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    # Clustering hierárquico simples
    ja_agrupados = set()
    temas_finais = []
    
    for i in range(len(temas)):
        if i in ja_agrupados:
            continue
            
        # Encontra similares acima do threshold
        similares_idx = [
            j for j in range(len(temas)) 
            if similaridades[i][j] > threshold and i != j and j not in ja_agrupados
        ]
        
        # Escolhe o termo mais curto e informativo do cluster
        cluster = [temas[i]] + [temas[j] for j in similares_idx]
        melhor_termo = min(cluster, key=lambda x: (len(x.split()), -len(x)))
        
        temas_finais.append(melhor_termo)
        ja_agrupados.add(i)
        ja_agrupados.update(similares_idx)
    
    return temas_finais[:20]  # Top 20 temas

# ============================================================================
# EXTRAÇÃO DE ENTIDADES E CO-OCORRÊNCIAS
# ============================================================================

def extrair_entidades_e_temas(texto: str, 
                               nlp, 
                               temas_conhecidos: List[str],
                               janela_contexto: int = 500) -> List[Tuple[str, str, int]]:
    """
    Extrai entidades nomeadas e detecta temas em janelas contextuais.
    SOLUÇÃO para o problema de matching ingênuo no código original.
    """
    co_occurrences = []
    segmentos = segmentar_texto(texto, janela_contexto)
    
    for segmento in segmentos:
        # Processa com spaCy
        doc = nlp(segmento)
        
        # Extrai entidades nomeadas relevantes
        entidades = [
            ent.text.title() 
            for ent in doc.ents 
            if ent.label_ in ['PER', 'ORG', 'LOC'] and len(ent.text) > 2
        ]
        
        # Detecta temas usando matching de tokens (não substring)
        segmento_lower = segmento.lower()
        temas_presentes = []
        
        for tema in temas_conhecidos:
            # Usa word boundaries para evitar falsos positivos
            padrao = r'\b' + re.escape(tema.lower()) + r'\b'
            if re.search(padrao, segmento_lower):
                temas_presentes.append(tema)
        
        # Combina entidades e temas
        elementos = list(set(entidades + temas_presentes))
        
        # Calcula co-ocorrências nesta janela
        if len(elementos) >= 2:
            for combo in combinations(sorted(elementos), 2):
                co_occurrences.append(combo)
    
    return co_occurrences

def construir_grafo_tematico(co_occurrences: List[Tuple[str, str]], 
                              min_co_occurrence: int = 3) -> nx.Graph:
    """
    Constrói grafo de relações entre tópicos com pesos significativos.
    """
    G = nx.Graph()
    contagem_arestas = Counter(co_occurrences)
    contagem_nos = Counter()
    
    # Conta aparições de cada nó
    for (n1, n2) in co_occurrences:
        contagem_nos[n1] += 1
        contagem_nos[n2] += 1
    
    # Adiciona arestas com peso significativo
    for (n1, n2), peso in contagem_arestas.items():
        if peso >= min_co_occurrence:
            # Normaliza peso pela frequência individual (PMI-like)
            peso_norm = peso / (np.sqrt(contagem_nos[n1] * contagem_nos[n2]))
            
            G.add_edge(n1, n2, weight=peso, weight_norm=peso_norm)
            G.nodes[n1]['frequencia'] = contagem_nos[n1]
            G.nodes[n2]['frequencia'] = contagem_nos[n2]
    
    # Calcula centralidade
    if len(G.nodes) > 0:
        centralidade = nx.degree_centrality(G)
        for no, cent in centralidade.items():
            G.nodes[no]['centralidade'] = cent
    
    return G
        
        # No pipeline melhorado, adicione:
def salvar_matriz_similaridade(temas_finais):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(temas_finais)
    similaridades = util.cos_sim(embeddings, embeddings).cpu().numpy()
    return similaridades.tolist()
# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def analisar_lives(pasta_srt: str = "./captions", 
                   min_documentos: int = 2,
                   min_co_occurrence: int = 3) -> Dict:
    """
    Pipeline completo de análise de tópicos em lives.
    """
    print("🚀 Iniciando análise de tópicos em lives...\n")
    
    # 1. Carrega recursos
    nlp = carregar_nlp()
    arquivos = [
        os.path.join(pasta_srt, f) 
        for f in os.listdir(pasta_srt) 
        if f.endswith(".srt")
    ]
    
    if not arquivos:
        print("❌ Nenhum arquivo SRT encontrado.")
        return {}
    
    print(f"📁 Encontrados {len(arquivos)} arquivos SRT\n")
    
    # 2. Processa arquivos
    corpus_lematizado = []
    corpus_bruto = []
    metadados = []
    
    for caminho in arquivos:
        try:
            print(f"⚙️  Processando: {os.path.basename(caminho)}")
            
            texto_bruto = processar_srt(caminho)
            corpus_bruto.append(texto_bruto)
            
            # Lematiza em segmentos para evitar overflow
            segmentos = segmentar_texto(texto_bruto, 10000)
            tokens_lemmatizados = []
            
            for seg in segmentos:
                doc = nlp(seg)
                tokens = [
                    token.lemma_.lower() 
                    for token in doc
                    if token.pos_ in ['NOUN', 'PROPN', 'ADJ']
                    and not token.is_stop
                    and token.lemma_.lower() not in STOPWORDS_LIVES
                    and len(token.lemma_) > 2
                    and not token.like_num
                ]
                tokens_lemmatizados.extend(tokens)
            
            texto_lematizado = " ".join(tokens_lemmatizados)
            corpus_lematizado.append(texto_lematizado)
            
            metadados.append({
                "arquivo": os.path.basename(caminho),
                "tamanho": len(texto_bruto),
                "tokens": len(tokens_lemmatizados)
            })
            
            print(f"   ✅ {len(tokens_lemmatizados)} tokens extraídos\n")
            
        except Exception as e:
            print(f"   ❌ Erro: {e}\n")
            continue
    
    if len(corpus_lematizado) < min_documentos:
        print(f"❌ Documentos insuficientes para análise (mínimo: {min_documentos})")
        return {}
    
    # 3. Extrai candidatos com TF-IDF
    print("📊 Extraindo tópicos candidatos com TF-IDF...")
    candidatos, _ = extrair_candidatos_tfidf(corpus_lematizado, min_df=min_documentos)
    print(f"   ✅ {len(candidatos)} candidatos extraídos\n")
    
    # 4. Refina com BERT
    temas_finais = agrupar_temas_semanticamente(candidatos, threshold=0.75)
    print(f"   ✅ {len(temas_finais)} temas finais após agrupamento semântico\n")
    
    # 5. Extrai co-ocorrências em janelas contextuais
    print("🔗 Analisando co-ocorrências em contextos...")
    todas_co_occurrences = []
    
    for i, texto in enumerate(corpus_bruto):
        print(f"   Documento {i+1}/{len(corpus_bruto)}")
        co_occ = extrair_entidades_e_temas(texto, nlp, temas_finais, janela_contexto=1000)
        todas_co_occurrences.extend(co_occ)
    
    print(f"   ✅ {len(todas_co_occurrences)} co-ocorrências detectadas\n")
    
    # 6. Constrói grafo
    print("🕸️  Construindo grafo de relações...")
    G = construir_grafo_tematico(todas_co_occurrences, min_co_occurrence)
    print(f"   ✅ Grafo com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas\n")
    
    # 7. Calcula estatísticas
    contagem_geral = Counter()
    for (n1, n2) in todas_co_occurrences:
        contagem_geral[n1] += 1
        contagem_geral[n2] += 1
    
    # 8. Monta resultado
    resultado = {
        "metadados": {
            "total_documentos": len(arquivos),
            "documentos_processados": len(corpus_bruto),
            "arquivos": metadados
        },
        "temas": {
            "candidatos_iniciais": candidatos[:30],
            "temas_finais": temas_finais,
            "ranking": contagem_geral.most_common(30)
        },
        "grafo": json_graph.node_link_data(G),

        'analise_semantica' : {
        'matriz_similaridade': salvar_matriz_similaridade(temas_finais),
        'temas_ordem': temas_finais
        }
    }
    
    # 9. Salva resultado
    with open("analise_topicos.json", 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    print("✅ Análise concluída! Resultado salvo em 'analise_topicos.json'")
    print(f"\n📈 Resumo:")
    print(f"   • Temas identificados: {len(temas_finais)}")
    print(f"   • Nós no grafo: {G.number_of_nodes()}")
    print(f"   • Conexões: {G.number_of_edges()}")
    print(f"\n🏆 Top 5 temas:")
    for tema, freq in contagem_geral.most_common(5):
        print(f"   {tema}: {freq} menções")
    
    return resultado

# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    analisar_lives(
        pasta_srt="./captions",
        min_documentos=2,      # Mínimo de documentos para TF-IDF
        min_co_occurrence=3    # Mínimo de co-ocorrências para criar aresta
    )