import faiss
import pickle
from pathlib import Path


def validate_indexing_artifacts(
    bm25_path="/kaggle/working/corpus_bm25.pkl",
    faiss_path="/kaggle/working/corpus_dense.index",
    dense_meta_path="/kaggle/working/corpus_dense.pkl",
    expected_dim=768,  # 768 para MPNet
):
    print("Iniciando Validação de Artefatos de First-Stage Retrieval...\n")

    # 1. Validação do Índice Denso (FAISS)
    try:
        print(f"Carregando FAISS Index de: {faiss_path}")
        faiss_index = faiss.read_index(str(faiss_path))
        ntotal = faiss_index.ntotal
        dim = faiss_index.d

        print(" [V] FAISS Carregado com Sucesso.")
        print(f"  -> Total de Vetores (ntotal): {ntotal:,}")
        print(f"  -> Dimensionalidade (d): {dim}")

        if dim != expected_dim:
            print(
                f" [!] ALERTA: Dimensão {dim} não bate com o esperado {expected_dim}."
            )
    except Exception as e:
        print(f" [X] Erro Crítico ao carregar FAISS: {e}")

    # 2. Validação do Mapeamento de Metadados (Dense Meta)
    try:
        print(f"\nCarregando Dense Metadata de: {dense_meta_path}")
        with open(dense_meta_path, "rb") as f:
            dense_meta = pickle.load(f)

        meta_len = len(dense_meta)
        print(f" [V] Metadata Carregado. Total de Registros: {meta_len:,}")

        # Teste de Invariante Estrutural
        assert (
            ntotal == meta_len
        ), "Data Leakage / Corruption Risk: FAISS ntotal não corresponde ao tamanho do Metadata!"
        print(" [V] Invariante de Alinhamento (FAISS <-> Meta) Validado.")

        # Amostra Estrutural
        print(f"  -> Amostra do Metadata (Top 3): {dense_meta[:3]}")
    except AssertionError as ae:
        print(f" [X] Erro de Assimetria: {ae}")
    except Exception as e:
        print(f" [X] Erro Crítico ao carregar Dense Meta: {e}")

    # 3. Validação do Índice Esparso (BM25)
    try:
        print(f"\nCarregando BM25 Index de: {bm25_path}")
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        print(f" [V] BM25 Carregado. Tipo do Objeto: {type(bm25_index)}")

        # Dependendo da implementação interna (se usou rank_bm25 ou TfidfVectorizer):
        if hasattr(bm25_index, "corpus_size"):
            print(f"  -> Tamanho do Corpus BM25: {bm25_index.corpus_size:,}")
        elif hasattr(bm25_index, "vocabulary_"):
            print(
                f"  -> Tamanho do Vocabulário (TF-IDF): {len(bm25_index.vocabulary_):,}"
            )

    except Exception as e:
        print(f" [X] Erro Crítico ao carregar BM25: {e}")


validate_indexing_artifacts()
