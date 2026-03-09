import pandas as pd
import pickle
from pathlib import Path
import os


def recover_metadata_only(
    laws_path="data/raw/laws_de.csv",
    courts_path="data/raw/court_considerations.csv",
    out_pkl="data/processed/corpus_dense.pkl",
):
    """
    Iniciando reconstrução de metadados (Bypass de GPU).
    Garante que a ordem das citações bata 1:1 com os vetores do FAISS.
    """
    print("Iniciando reconstrução de metadados...")

    # 1. Carrega apenas as citações, mantendo a ordem estrita da leitura original
    print(f"Lendo leis de {laws_path}...")
    if os.path.exists(laws_path):
        df_laws = pd.read_csv(laws_path, usecols=["citation"])
        citations_laws = df_laws["citation"].tolist()
    else:
        print(f"Aviso: {laws_path} não encontrado.")
        citations_laws = []

    print(f"Lendo acórdãos de {courts_path}...")
    if os.path.exists(courts_path):
        df_courts = pd.read_csv(courts_path, usecols=["citation"])
        citations_courts = df_courts["citation"].tolist()
    else:
        print(f"Aviso: {courts_path} não encontrado.")
        citations_courts = []

    # 2. Concatena na ordem exata em que o FAISS os ingeriu (Laws -> Courts)
    all_citations = citations_laws + citations_courts

    total = len(all_citations)
    print(f"Total de citações extraídas: {total}")

    # 3. Salva no formato esperado pelo DenseIndex (dicionário com chave 'citations')
    # O DenseIndex.load foi atualizado para ler este formato.
    metadata = {
        "citations": all_citations,
        "model_name": "intfloat/multilingual-e5-small",  # Padrão
        "text_field": "text",
        "citation_field": "citation",
    }

    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[V] Arquivo {out_pkl} reconstruído com sucesso.")


if __name__ == "__main__":
    # Ajuste os caminhos se necessário para o ambiente local ou Kaggle
    recover_metadata_only()
