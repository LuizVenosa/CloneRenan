import os
import json
import spacy
import networkx as nx
from collections import Counter
from itertools import combinations
from networkx.readwrite import json_graph
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nlp = spacy.load("pt_core_news_lg")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
    nlp = spacy.load("pt_core_news_lg")

# Blacklist sempre em UPPER para comparação segura
TERMOS_BANIDOS = {
    'LIKE LIVE', 'VAMOS VAMOS', 'LIKE NA LIVE', 'INSCREVA CANAL', 
    'BOA NOITE', 'VALET PLUS', 'BANCO MASTER', 'RENAN SANTOS', 
    'PESSOAL', 'GENTE', 'COISA', 'AQUI', 'TÁ', 'NÉ', 'RENAN'
}

def extrair_temas_naturais(textos, top_n=20):
    from spacy.lang.pt.stop_words import STOP_WORDS
    all_stops = STOP_WORDS.union({'então', 'falar', 'querer', 'ficar', 'olha', 'hum', 'hm'})
    
    vectorizer = TfidfVectorizer(
        ngram_range=(2, 3),
        stop_words=list(all_stops),
        max_features=200,
        min_df=3
    )
    tfidf_matrix = vectorizer.fit_transform(textos)
    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    
    ranking_contagem = Counter()
    for col, term in enumerate(feature_names):
        termo_upper = term.strip().upper()
        if termo_upper not in TERMOS_BANIDOS:
            ranking_contagem[termo_upper] += sums[0, col]
            
    # Retornamos em Title Case (Estilo Normal)
    return [termo.title() for termo, score in ranking_contagem.most_common(top_n)]

def processar_transcricoes(pasta_srt):
    all_texts = []
    contagem_geral = Counter()
    co_occurrences = Counter()
    
    caminhos = [os.path.join(pasta_srt, f) for f in os.listdir(pasta_srt) if f.endswith(".srt")]
    if not caminhos: return None

    for path in caminhos:
        with open(path, 'r', encoding='utf-8') as f:
            texto = " ".join([l.strip() for l in f.readlines() if not l.strip().isdigit() and "-->" not in l])
            all_texts.append(texto)

    temas_descobertos = extrair_temas_naturais(all_texts)
    
    for texto in all_texts:
        chunks = [texto[i:i+600] for i in range(0, len(texto), 600)]
        for chunk in chunks:
            doc = nlp(chunk)
            
            # 1. Unificar tudo para UPPER para o cálculo (Evita duplicatas)
            ents = [ent.text.strip().upper() for ent in doc.ents if ent.label_ in ['PER', 'ORG']]
            temas = [t.upper() for t in temas_descobertos if t.lower() in chunk.lower()]
            
            todos_nos_upper = list(set(ents + temas))
            
            # Filtrar banidos
            todos_nos_upper = [n for n in todos_nos_upper if n not in TERMOS_BANIDOS and len(n) > 2]
            
            for no in todos_nos_upper: 
                contagem_geral[no] += 1
            
            if len(todos_nos_upper) > 1:
                for combo in combinations(sorted(todos_nos_upper), 2):
                    co_occurrences[combo] += 1

    # 2. Criar o Grafo e converter os nomes para Title Case na hora de adicionar os nós
    G = nx.Graph()
    for (node1, node2), weight in co_occurrences.items():
        if contagem_geral[node1] >= 5 and contagem_geral[node2] >= 5:
            # Converte para estilo normal (Ex: "Lula" em vez de "LULA")
            n1_normal = node1.title()
            n2_normal = node2.title()
            
            G.add_edge(n1_normal, n2_normal, weight=weight)
            G.nodes[n1_normal]['mencoes'] = contagem_geral[node1]
            G.nodes[n2_normal]['mencoes'] = contagem_geral[node2]
            
    # 3. Formatar o ranking para estilo normal para o frontend
    ranking_normalizado = [(nome.title(), qtd) for nome, qtd in contagem_geral.most_common(30)]
            
    resultado = {
        "stats": {
            "total_arquivos": len(caminhos),
            "ranking": ranking_normalizado, 
            "temas_descobertos": [t.title() for t in temas_descobertos] 
        },
        "grafo": json_graph.node_link_data(G, edges="edges")
    }
    
    with open("analise_completa.json", 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    processar_transcricoes("./captions")
    print("✅ Dados processados e unificados com sucesso!")