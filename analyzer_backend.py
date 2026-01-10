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
from datetime import datetime

# ============================================================================
# CONFIGURAÇÃO E RECURSOS
# ============================================================================

STOPWORDS_LIVES = {
    'falar', 'querer', 'ficar', 'saber', 'achar', 'ir', 'dar', 'ter', 'fazer',
    'poder', 'dever', 'deixar', 'conseguir', 'começar', 'acabar', 'passar',
    'vídeo', 'live', 'gente', 'pessoal', 'tá', 'né', 'coisa', 'aqui', 'cara',
    'tipo', 'assim', 'então', 'ai', 'galera', 'beleza', 'ok', 'certo',
    'olha', 'veja', 'entendeu', 'sabe', 'canal', 'inscrever', 'like',
    'ele', 'ela', 'isso', 'aquilo', 'esse', 'essa', 'outro', 'outra',
    'hoje', 'agora', 'depois', 'antes', 'dia', 'vez', 'ano', 'hora'
}

def carregar_nlp():
    """Carrega modelo spaCy com componentes otimizados"""
    try:
        nlp = spacy.load("pt_core_news_lg", disable=["parser", "attribute_ruler"])
        nlp.add_pipe('sentencizer')  # Adiciona sentencizer para frases
        return nlp
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
        nlp = spacy.load("pt_core_news_lg", disable=["parser", "attribute_ruler"])
        nlp.add_pipe('sentencizer')
        return nlp

def extrair_data_do_nome(nome_arquivo: str) -> str:
    """
    NOVO: Extrai data do nome do arquivo (formato: YYYYMMDD ou YYYY-MM-DD)
    Exemplo: "live_20240315.srt" -> "2024-03-15"
    """
    
    padroes = [
        r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})',  # YYYYMMDD ou YYYY-MM-DD
        r'(\d{2})[-_]?(\d{2})[-_]?(\d{4})',  # DDMMYYYY ou DD-MM-YYYY
    ]
    
    for padrao in padroes:
        match = re.search(padrao, nome_arquivo)
        if match:
            grupos = match.groups()
            if len(grupos[0]) == 4:  # YYYYMMDD
                return f"{grupos[0]}-{grupos[1]}-{grupos[2]}"
            else:  # DDMMYYYY
                return f"{grupos[2]}-{grupos[1]}-{grupos[0]}"
    
    return None

def processar_srt(caminho: str) -> str:
    """Lê arquivo SRT e remove metadados"""
    with open(caminho, 'r', encoding='utf-8') as f:
        linhas = f.readlines()
    
    texto = " ".join([
        l.strip() for l in linhas 
        if not l.strip().isdigit() 
        and "-->" not in l
        and l.strip()
    ])
    
    texto = re.sub(r'\[.*?\]', '', texto)
    texto = re.sub(r'\(.*?\)', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto.strip()

def segmentar_texto(texto: str, tamanho_janela: int = 5000) -> List[str]:
    
    if len(texto) <= tamanho_janela:
        return [texto]
    
    segmentos = []
    overlap = tamanho_janela // 4
    
    for i in range(0, len(texto), tamanho_janela - overlap):
        segmento = texto[i:i + tamanho_janela]
        if len(segmento) > 100:
            segmentos.append(segmento)
    
    return segmentos

def extrair_candidatos_tfidf(corpus_lematizado: List[str], 
                             min_df: int = 2, 
                             max_features: int = 150) -> Tuple[List[str], np.ndarray]:
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        max_df=0.8,
        token_pattern=r'\b[a-záàâãéèêíïóôõöúçñ]{3,}\b'
    )
    
    try:
        matriz = vectorizer.fit_transform(corpus_lematizado)
        termos = vectorizer.get_feature_names_out()
        
        termos_validos = [
            t.title() for t in termos
            if t not in STOPWORDS_LIVES
            and not t.isdigit()
            and len(t.split()) <= 2
        ]
        
        return termos_validos, matriz
        
    except ValueError as e:
        print(f"⚠️  Erro no TF-IDF: {e}")
        return [], None

def agrupar_temas_semanticamente(temas: List[str], 
                                  threshold: float = 0.75) -> List[str]:
    """Agrupa temas semanticamente similares"""
    if not temas or len(temas) < 2:
        return temas
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(temas, convert_to_tensor=True, show_progress_bar=False)
    similaridades = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    ja_agrupados = set()
    temas_finais = []
    
    for i in range(len(temas)):
        if i in ja_agrupados:
            continue
            
        similares_idx = [
            j for j in range(len(temas)) 
            if similaridades[i][j] > threshold and i != j and j not in ja_agrupados
        ]
        
        cluster = [temas[i]] + [temas[j] for j in similares_idx]
        melhor_termo = min(cluster, key=lambda x: (len(x.split()), -len(x)))
        
        temas_finais.append(melhor_termo)
        ja_agrupados.add(i)
        ja_agrupados.update(similares_idx)
    
    return temas_finais[:20]

# ============================================================================
# EXTRAÇÃO DE ENTIDADES E CO-OCORRÊNCIAS
# ============================================================================

def extrair_entidades_e_temas(texto: str, 
                               nlp, 
                               temas_conhecidos: List[str],
                               janela_contexto: int = 500) -> List[Tuple[str, str, int]]:
    """Extrai entidades nomeadas e detecta temas"""
    co_occurrences = []
    segmentos = segmentar_texto(texto, janela_contexto)
    
    for segmento in segmentos:
        doc = nlp(segmento)
        
        entidades = [
            ent.text.title() 
            for ent in doc.ents 
            if ent.label_ in ['PER', 'ORG', 'LOC'] and len(ent.text) > 2
        ]
        
        segmento_lower = segmento.lower()
        temas_presentes = []
        
        for tema in temas_conhecidos:
            padrao = r'\b' + re.escape(tema.lower()) + r'\b'
            if re.search(padrao, segmento_lower):
                temas_presentes.append(tema)
        
        elementos = list(set(entidades + temas_presentes))
        
        if len(elementos) >= 2:
            for combo in combinations(sorted(elementos), 2):
                co_occurrences.append(combo)
    
    return co_occurrences

def construir_grafo_tematico(co_occurrences: List[Tuple[str, str]], 
                              min_co_occurrence: int = 3) -> nx.Graph:
    """Constrói grafo de relações entre tópicos"""
    G = nx.Graph()
    contagem_arestas = Counter(co_occurrences)
    contagem_nos = Counter()
    
    for (n1, n2) in co_occurrences:
        contagem_nos[n1] += 1
        contagem_nos[n2] += 1
    
    for (n1, n2), peso in contagem_arestas.items():
        if peso >= min_co_occurrence:
            peso_norm = peso / (np.sqrt(contagem_nos[n1] * contagem_nos[n2]))
            
            G.add_edge(n1, n2, weight=peso, weight_norm=peso_norm)
            G.nodes[n1]['frequencia'] = contagem_nos[n1]
            G.nodes[n2]['frequencia'] = contagem_nos[n2]
    
    if len(G.nodes) > 0:
        centralidade = nx.degree_centrality(G)
        for no, cent in centralidade.items():
            G.nodes[no]['centralidade'] = cent
    
    return G

# ============================================================================
# NOVAS ANÁLISES (MELHORIAS)
# ============================================================================

def detectar_comunidades(G: nx.Graph) -> Dict:
    """
    NOVO: Detecta comunidades/clusters no grafo
    Retorna dicionário com nó -> comunidade_id
    """
    if G.number_of_nodes() < 3:
        return {}
    
    try:
        from networkx.algorithms import community
        comunidades = community.greedy_modularity_communities(G)
        
        # Mapeia nó -> id_comunidade
        mapa_comunidades = {}
        for idx, com in enumerate(comunidades):
            for no in com:
                mapa_comunidades[no] = idx
        
        return mapa_comunidades
    except Exception as e:
        print(f"⚠️ Erro ao detectar comunidades: {e}")
        return {}

def extrair_frases_chave(texto: str, nlp, n_frases: int = 10) -> List[str]:
    """
    NOVO: Extrai frases-chave representativas do texto
    Usa combinação de TF-IDF + POS tagging
    """
    doc = nlp(texto[:50000])  # Limita para performance
    
    # Extrai candidatos: substantivos + adjetivos
    candidatos = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop:
            candidatos.append(token.lemma_.lower())
    
    # Conta frequências
    freq = Counter(candidatos)
    
    # Extrai frases contendo termos frequentes
    termos_top = set([t for t, _ in freq.most_common(20)])
    frases = []
    
    for sent in doc.sents:
        if len(sent.text.split()) > 5 and len(sent.text.split()) < 20:
            tokens_sent = [t.lemma_.lower() for t in sent if not t.is_stop]
            if any(t in termos_top for t in tokens_sent):
                frases.append(sent.text.strip())
    
    return frases[:n_frases]

def analisar_por_documento(textos_brutos: List[str], 
                           temas_finais: List[str],
                           nlp,
                           metadados: List[Dict]) -> List[Dict]:
    """
    NOVO: Análise individual de cada documento
    Retorna lista com temas presentes em cada live
    """
    analise_docs = []
    
    for idx, (texto, meta) in enumerate(zip(textos_brutos, metadados)):
        # Detecta temas presentes
        temas_no_doc = []
        texto_lower = texto.lower()
        
        for tema in temas_finais:
            padrao = r'\b' + re.escape(tema.lower()) + r'\b'
            matches = len(re.findall(padrao, texto_lower))
            if matches > 0:
                temas_no_doc.append({
                    'tema': tema,
                    'mencoes': matches
                })
        
        frases = extrair_frases_chave(texto, nlp, n_frases=5)
        
        analise_docs.append({
            'arquivo': meta['arquivo'],
            'data': meta.get('data'),
            'temas_presentes': sorted(temas_no_doc, key=lambda x: x['mencoes'], reverse=True),
            'total_temas': len(temas_no_doc),
            'frases_chave': frases
        })
    
    return analise_docs

def salvar_matriz_similaridade(temas_finais: List[str]) -> List[List[float]]:
    """Calcula e retorna matriz de similaridade semântica"""
    if len(temas_finais) < 2:
        return []
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(temas_finais)
    similaridades = util.cos_sim(embeddings, embeddings).cpu().numpy()
    
    return similaridades.tolist()

# ============================================================================
# PIPELINE PRINCIPAL (MELHORADO)
# ============================================================================

def analisar_lives(pasta_srt: str = "./captions", 
                   min_documentos: int = 2,
                   min_co_occurrence: int = 3) -> Dict:
    """Pipeline completo de análise - VERSÃO MELHORADA"""
    
    print("🚀 Iniciando análise de tópicos em lives...\n")
    
    nlp = carregar_nlp()
    arquivos = [
        os.path.join(pasta_srt, f) 
        for f in os.listdir(pasta_srt) 
        if f.endswith(".srt")
    ]
    
    if not arquivos:
        print("❌ Nenhum arquivo SRT encontrado.")
        return {}
    
    print(f"Encontrados {len(arquivos)} arquivos SRT\n")
    
    # Processar arquivos COM EXTRAÇÃO DE DATA
    corpus_lematizado = []
    corpus_bruto = []
    metadados = []
    
    for caminho in arquivos:
        try:
            nome_arquivo = os.path.basename(caminho)
            print(f"⚙️  Processando: {nome_arquivo}")
            
            texto_bruto = processar_srt(caminho)
            corpus_bruto.append(texto_bruto)
            
            # NOVO: Extrai data
            data_extraida = extrair_data_do_nome(nome_arquivo)
            
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
                "arquivo": nome_arquivo,
                "data": data_extraida,  # NOVO
                "tamanho": len(texto_bruto),
                "tokens": len(tokens_lemmatizados)
            })
            
            print(f"   ✅ {len(tokens_lemmatizados)} tokens extraídos\n")
            
        except Exception as e:
            print(f"   ❌ Erro: {e}\n")
            continue
    
    if len(corpus_lematizado) < min_documentos:
        print(f"❌ Documentos insuficientes")
        return {}
    

    print("📊 Extraindo tópicos candidatos...")
    candidatos, _ = extrair_candidatos_tfidf(corpus_lematizado, min_df=min_documentos)
    print(f"   ✅ {len(candidatos)} candidatos\n")
    
    temas_finais = agrupar_temas_semanticamente(candidatos, threshold=0.75)
    print(f"   ✅ {len(temas_finais)} temas finais\n")
    
    # Co-ocorrências
    print("🔗 Analisando co-ocorrências...")
    todas_co_occurrences = []
    
    for i, texto in enumerate(corpus_bruto):
        print(f"   Documento {i+1}/{len(corpus_bruto)}")
        co_occ = extrair_entidades_e_temas(texto, nlp, temas_finais, janela_contexto=1000)
        todas_co_occurrences.extend(co_occ)
    
    print(f"   {len(todas_co_occurrences)} co-ocorrências\n")
    

    print("Construindo grafo...")
    G = construir_grafo_tematico(todas_co_occurrences, min_co_occurrence)
    print(f"    {G.number_of_nodes()} nós, {G.number_of_edges()} arestas\n")
    

    print("Detectando comunidades no grafo...")
    comunidades = detectar_comunidades(G)
    print(f"  {len(set(comunidades.values()))} comunidades identificadas\n")

    # Estatísticas
    contagem_geral = Counter()
    for (n1, n2) in todas_co_occurrences:
        contagem_geral[n1] += 1
        contagem_geral[n2] += 1
    
    # NOVO: Adiciona comunidades aos nós do grafo
    for no in G.nodes():
        if no in comunidades:
            G.nodes[no]['comunidade'] = comunidades[no]
    
    # Monta resultado
    resultado = {
        "metadados": {
            "total_documentos": len(arquivos),
            "documentos_processados": len(corpus_bruto),
            "arquivos": metadados,
            "tem_dados_temporais": any(m.get('data') for m in metadados)  # NOVO
        },
        "temas": {
            "candidatos_iniciais": candidatos[:30],
            "temas_finais": temas_finais,
            "ranking": contagem_geral.most_common(30)
        },
        "grafo": json_graph.node_link_data(G),
        "analise_semantica": {
            "matriz_similaridade": salvar_matriz_similaridade(temas_finais),
            "temas_ordem": temas_finais
        },
        "comunidades": {  # NOVO
            "mapeamento": comunidades,
            "total": len(set(comunidades.values()))
        },
        "analise_documentos": analise_docs  # NOVO
    }
    
    # Salva
    with open("analise_topicos.json", 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    print("✅ Análise concluída!\n")
    print(f"📈 Resumo:")
    print(f"   • Temas: {len(temas_finais)}")
    print(f"   • Nós: {G.number_of_nodes()}")
    print(f"   • Conexões: {G.number_of_edges()}")
    print(f"   • Comunidades: {len(set(comunidades.values()))}")
    print(f"\n🏆 Top 5 temas:")
    for tema, freq in contagem_geral.most_common(5):
        print(f"   {tema}: {freq} menções")
    
    return resultado

if __name__ == "__main__":
    analisar_lives(
        pasta_srt="./captions",
        min_documentos=2,
        min_co_occurrence=3
    )