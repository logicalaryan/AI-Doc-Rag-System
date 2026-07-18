"""
scripts/ingest_docs.py — CLI to ingest documents into ChromaDB.

Usage:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --data-dir ./data --chunk-size 1000 --chunk-overlap 200
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingest import ingest


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vectorstore"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory containing documents to ingest (default: ./data)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Size of each text chunk in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks (default: 200)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./vectorstore",
        help="Directory to store the ChromaDB index (default: ./vectorstore)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="recursive",
        choices=["recursive", "character"],
        help="Chunking strategy to use: 'recursive' or 'character' (default: recursive)",
    )
    args = parser.parse_args()

    print(f"Ingesting documents from: {args.data_dir}")
    print(f"  Chunk size:    {args.chunk_size}")
    print(f"  Chunk overlap: {args.chunk_overlap}")
    print(f"  Strategy:      {args.strategy}")
    print(f"  Persist dir:   {args.persist_dir}")
    print()

    result = ingest(
        data_dir=args.data_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist_dir=args.persist_dir,
        strategy=args.strategy,
    )

    if result is not None:
        print("\nIngestion complete! You can now run the API or UI.")
    else:
        print("\nNo documents ingested. Add files to the data/ directory and try again.")


if __name__ == "__main__":
    main()
