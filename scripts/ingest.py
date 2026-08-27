#!/usr/bin/env python3
"""
Ingest SF listings CSV into Pinecone vector database.

Usage:
  python scripts/ingest.py
  python scripts/ingest.py --source data/listings.csv
  python scripts/ingest.py --source data/listings.csv --recreate
  python scripts/ingest.py --photos-only
  python scripts/ingest.py --photos-only --recreate
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.indexer import run_ingest, run_photos_ingest


def main():
    parser = argparse.ArgumentParser(description="Ingest SF listings into Pinecone")
    parser.add_argument("--source", default="data/listings.csv", help="Path to listings CSV")
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the index")
    parser.add_argument("--photos-only", action="store_true", help="Re-index photos only (skip BM25 fit and listings upsert)")
    args = parser.parse_args()

    if args.photos_only:
        run_photos_ingest(recreate=args.recreate)
    else:
        if not Path(args.source).exists():
            print(f"Error: {args.source} not found.")
            sys.exit(1)
        run_ingest(args.source, recreate=args.recreate)


if __name__ == "__main__":
    main()
