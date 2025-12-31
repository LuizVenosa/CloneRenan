import os
import json
import spacy
import networkx as nx
from collections import Counter
from itertools import combinations
from networkx.readwrite import json_graph
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregar modelo SpaCy
try:
    nlp = spacy.load("pt_core_news_lg")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "pt_core_news_lg"])
    nlp = spacy.load("pt_core_news_lg")

STOP_WORDS_CUSTOM = {'gente', 'então', 'né', 'aqui', 'tá', 'coisa', 'falar', 'querer', 'ir', 'ficar', 'saber', 'achar', 'olha', 'bom', 'bem', 'muito'}

def extrair_temas_naturais(textos, top_n=15):
    from spacy.lang.pt.stop_words import STOP_WORDS
    all_stops = STOP_WORDS.union(STOP_WORDS_CUSTOM)
    vectorizer = TfidfVectorizer(ngram_range=(2, 3), stop_words=list(all_stops), max_features=150, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(textos)
    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.sum(axis=0)
    data = [(term, sums[0, col]) for col, term in enumerate(feature_names)]
    return [termo for termo, score in sorted(data, key=lambda x: x[1], reverse=True)[:top_n]]

def processar_transcricoes(pasta_srt):
    all_texts = []
    co_occurrences = Counter()
    contagem_geral = Counter()
    
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
            entidades = list(set([ent.text.title() for ent in doc.ents if ent.label_ in ['PER', 'ORG']]))
            temas_presentes = [t for t in temas_descobertos if t.lower() in chunk.lower()]
            
            todos_nos = entidades + temas_presentes
            for no in todos_nos: contagem_geral[no] += 1
            
            if len(todos_nos) > 1:
                for combo in combinations(sorted(todos_nos), 2):
                    co_occurrences[combo] += 1

    G = nx.Graph()
    for (node1, node2), weight in co_occurrences.items():
        if contagem_geral[node1] >= 5 and contagem_geral[node2] >= 5:
            G.add_edge(node1, node2, weight=weight)
            G.nodes[node1]['mencoes'] = contagem_geral[node1]
            G.nodes[node2]['mencoes'] = contagem_geral[node2]
            
    resultado = {
        "stats": {
            "total_arquivos": len(caminhos),
            "total_conexoes": len(G.edges),
            "ranking": contagem_geral.most_common(20), # A CHAVE QUE FALTAVA
            "temas_descobertos": temas_descobertos 
        },
        "grafo": json_graph.node_link_data(G, edges="edges")
    }
    
    with open("analise_completa.json", 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)
    return resultado

if __name__ == "__main__":
    processar_transcricoes("./captions")