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


def load_corpus_in_chunks(file_path: Path, chunksize: int = 100000) -> Tuple[List[str], List[str]]:
    """Load texts and citations from large CSV in chunks."""
    if not file_path.exists():
        return [], []

    print(f"  Streaming corpus from {file_path}...")
    texts = []
    citations = []

    # Check if it's CSV or JSONL
    if file_path.suffix == ".csv":
        for chunk in pd.read_csv(file_path, chunksize=chunksize, usecols=["citation", "text"]):
            texts.extend(chunk["text"].fillna("").astype(str).tolist())
            citations.extend(chunk["citation"].fillna("Unknown").astype(str).tolist())
            print(f"    Loaded {len(citations)} rows...")
    else:
        # JSONL fallback (less memory efficient but manageable)
        import json

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                texts.append(item.get("text", ""))
                citations.append(item.get("citation", "Unknown"))

    return texts, citations


def build_hybrid_index(
    input_dir: Path, output_dir: Path, model_name: str, limit: int = None
) -> None:
    """Build both BM25 and Dense indices using memory-efficient streaming."""
    print("Building hybrid index for LARGE-SCALE corpus...")

    all_texts = []
    all_citations = []

    # 1. Load Laws
    for name in ["laws_de.csv", "federal_laws.jsonl"]:
        path = input_dir / name
        t, c = load_corpus_in_chunks(path)
        if t:
            all_texts.extend(t)
            all_citations.extend(c)
            break

    # 2. Load Courts
    for name in ["court_considerations.csv", "court_decisions.jsonl"]:
        path = input_dir / name
        t, c = load_corpus_in_chunks(path)
        if t:
            all_texts.extend(t)
            all_citations.extend(c)
            break

    if not all_texts:
        print("  Error: No corpus files found.")
        return

    if limit:
        all_texts = all_texts[:limit]
        all_citations = all_citations[:limit]

    print(f"  Total documents indexed: {len(all_texts)}")

    # 1. Build BM25 Index (Sparse)
    print("  Step 1: Building BM25 Index (Sparse Matrix)...")
    bm25_idx = BM25Index()
    bm25_idx.build(all_texts, all_citations)

    bm25_path = output_dir / "corpus_bm25.pkl"
    bm25_idx.save(bm25_path)
    print(f"  BM25 Index saved to {bm25_path}")

    # 2. Clear memory for Dense Indexing
    # Keep only all_texts and all_citations temporarily
    del bm25_idx
    gc.collect()

    # 3. Build Dense Index (FAISS)
    print(f"  Step 2: Building Dense Index (FAISS + {model_name})...")
    # Wrap in a format DenseIndex expects (list of dicts)
    # But to save memory, we rebuild this list inside DenseIndex.build or similar
    # For now, let's pass documents as citations only to save RAM
    docs_metadata = [{"citation": c} for c in all_citations]

    dense_idx = DenseIndex(model_name=model_name)
    # Modified DenseIndex build to take texts and metadata separately to save RAM
    # Here I'll use a hack: create a minimal documents list
    dense_idx.documents = docs_metadata

    # Batch size 128 for T4 speed
    dense_idx.build_from_lists(all_texts, docs_metadata, batch_size=128)

    dense_prefix = output_dir / "corpus_dense"
    dense_idx.save(dense_prefix)
    print(f"  Dense Index saved to {dense_prefix}.index and .pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-name", type=str, default="intfloat/multilingual-e5-small")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    build_hybrid_index(args.input_dir, args.output_dir, args.model_name, args.limit)
    print("\nLarge-scale index building complete!")


if __name__ == "__main__":
    main()
