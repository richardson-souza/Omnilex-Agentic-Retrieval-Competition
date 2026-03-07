import argparse
import sys
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import BM25Index, load_jsonl_corpus
from omnilex.retrieval.dense_index import DenseIndex


def load_corpus(file_path: Path) -> List[Dict[str, Any]]:
    """Load corpus from either JSONL or CSV."""
    if not file_path.exists():
        return []

    print(f"  Loading corpus from {file_path}...")
    if file_path.suffix == ".jsonl":
        return load_jsonl_corpus(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        # Ensure required columns exist
        if "citation" not in df.columns or "text" not in df.columns:
            print(f"  Warning: CSV {file_path} missing 'citation' or 'text' columns.")
            return []
        return df.to_dict("records")
    return []


def build_hybrid_index(
    input_dir: Path, output_dir: Path, model_name: str, limit: int = None
) -> None:
    """Build both BM25 and Dense indices for the full corpus."""
    print("Building hybrid index for COMPLETE corpus...")

    documents = []

    # 1. Load Laws (Prioritize full corpus over samples)
    laws_found = False
    for name in ["laws_de.csv", "federal_laws.jsonl", "samples/federal_laws.jsonl"]:
        path = input_dir / name
        docs = load_corpus(path)
        if docs:
            print(f"  Loaded {len(docs)} documents from {name}")
            documents.extend(docs)
            laws_found = True
            break

    # 2. Load Courts (Prioritize full corpus over samples)
    courts_found = False
    for name in [
        "court_considerations.csv",
        "court_decisions.jsonl",
        "samples/court_decisions.jsonl",
    ]:
        path = input_dir / name
        docs = load_corpus(path)
        if docs:
            print(f"  Loaded {len(docs)} documents from {name}")
            documents.extend(docs)
            courts_found = True
            break

    if not documents:
        print("  Error: No corpus files found. Skipping index build.")
        return

    if limit:
        print(f"  Limiting to first {limit} documents for local testing.")
        documents = documents[:limit]

    print(f"  Total documents to index: {len(documents)}")

    # 1. Build BM25 Index
    print("  Building BM25 Index (Sparse)...")
    bm25_idx = BM25Index(documents=documents, text_field="text", citation_field="citation")
    bm25_path = output_dir / "corpus_bm25.pkl"
    bm25_idx.save(bm25_path)
    print(f"  BM25 Index saved to {bm25_path}")

    # 2. Build Dense Index
    print(f"  Building Dense Index (FAISS + {model_name})...")
    # Note: On local machine with 2GB VRAM, this might need small batch_size
    dense_idx = DenseIndex(
        model_name=model_name, documents=documents, text_field="text", citation_field="citation"
    )
    dense_prefix = output_dir / "corpus_dense"
    dense_idx.save(dense_prefix)
    print(f"  Dense Index saved to {dense_prefix}.index and .pkl")


def main():
    parser = argparse.ArgumentParser(description="Build hybrid indices for legal corpora")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Input directory with corpus files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for indices",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="Sentence transformer model for dense indexing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents for testing",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    build_hybrid_index(args.input_dir, args.output_dir, args.model_name, args.limit)
    print("\nHybrid index building complete!")


if __name__ == "__main__":
    main()
