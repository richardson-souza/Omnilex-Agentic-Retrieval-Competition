import argparse
import sys
import os
import gc
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.dense_index import DenseIndex


def load_corpus_in_chunks(
    file_path: Path, chunksize: int = 100000
) -> Tuple[List[str], List[str]]:
    """Load texts and citations from large CSV in chunks."""
    if not file_path.exists():
        return [], []

    print(f"  Streaming corpus from {file_path}...")
    texts = []
    citations = []

    if file_path.suffix == ".csv":
        for chunk in pd.read_csv(
            file_path, chunksize=chunksize, usecols=["citation", "text"]
        ):
            texts.extend(chunk["text"].fillna("").astype(str).tolist())
            citations.extend(chunk["citation"].fillna("Unknown").astype(str).tolist())
            print(f"    Loaded {len(citations)} rows...")
    else:
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item.get("text", ""))
                citations.append(item.get("citation", "Unknown"))

    return texts, citations


def build_hybrid_index(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    limit: int = None,
    multi_gpu: bool = False,
) -> None:
    """Build both BM25 and Dense indices using memory-efficient streaming."""
    print(f"Building hybrid index for LARGE-SCALE corpus (Multi-GPU={multi_gpu})...")

    all_texts = []
    all_citations = []

    # 1. Load Laws
    for name in ["laws_de.csv", "federal_laws.jsonl"]:
        path = input_dir / name
        t, c = load_corpus_in_chunks(path)
        if t:
            print(f"  Loaded {len(t)} laws.")
            all_texts.extend(t)
            all_citations.extend(c)
            break

    # 2. Load Courts
    for name in ["court_considerations.csv", "court_decisions.jsonl"]:
        path = input_dir / name
        t, c = load_corpus_in_chunks(path)
        if t:
            print(f"  Loaded {len(t)} court considerations.")
            all_texts.extend(t)
            all_citations.extend(c)
            break

    if not all_texts:
        print("  Error: No corpus files found.")
        return

    if limit:
        all_texts = all_texts[:limit]
        all_citations = all_citations[:limit]

    total_docs = len(all_texts)
    print(f"  Total documents to index: {total_docs}")

    # 1. Build BM25 Index (Sparse Matrix)
    print("  Step 1: Building BM25 Index (Sparse Matrix)...")
    bm25_idx = BM25Index()
    bm25_idx.build(all_texts, all_citations)

    bm25_path = output_dir / "corpus_bm25.pkl"
    bm25_idx.save(bm25_path)
    print(f"  BM25 Index saved to {bm25_path}")

    # Clear BM25 from RAM
    del bm25_idx
    gc.collect()

    # 2. Build Dense Index (FAISS)
    print("  Step 2: Building Dense Index (FAISS)...")
    dense_idx = DenseIndex(model_name=model_name)

    # Pass full lists and let DenseIndex handle multi-GPU and metadata accumulation
    dense_idx.build_from_lists(
        all_texts, all_citations, batch_size=128, multi_gpu=multi_gpu
    )

    dense_prefix = output_dir / "corpus_dense"
    dense_idx.save(dense_prefix)

    # Final Validation
    print("-" * 30)
    print(f"Indexation Densa Concluída.")
    print(f"Total de vetores (FAISS): {dense_idx.index.ntotal}")
    print(f"Total de metadados (PKL): {len(dense_idx.citations)}")

    if dense_idx.index.ntotal == len(dense_idx.citations):
        print("[V] Invariante de Alinhamento (FAISS <-> Meta) Validado.")
    else:
        print("[X] Erro de Assimetria detectado durante a criação!")
    print("-" * 30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--model-name", type=str, default="intfloat/multilingual-e5-small"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--multi-gpu", action="store_true", help="Enable multi-GPU encoding"
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    build_hybrid_index(
        args.input_dir, args.output_dir, args.model_name, args.limit, args.multi_gpu
    )
    print("\nHybrid index building complete!")


if __name__ == "__main__":
    main()
