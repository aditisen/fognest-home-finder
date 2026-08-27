from functools import lru_cache
from pathlib import Path

import pandas as pd
from pinecone_text.sparse import BM25Encoder

from ..config import settings


def build_corpus_text(row: dict | pd.Series) -> str:
    """Concatenate structured listing fields into a BM25 document string."""
    fields = [
        "neighborhood_name", "property_type", "architecture_style",
        "views", "school_district_name", "elementary_school_name",
        "middle_school_name", "high_school_name",
    ]
    parts = [str(row.get(f, "") or "").strip() for f in fields]
    return " ".join(p for p in parts if p and p.lower() not in ("none", "nan", ""))


def fit_bm25(csv_path: str) -> BM25Encoder:
    """Fit BM25 on structured fields from listings CSV and save model."""
    df = pd.read_csv(csv_path, dtype={"mls_listing_number": str})
    corpus = [build_corpus_text(row) for _, row in df.iterrows()]
    encoder = BM25Encoder()
    encoder.fit(corpus)
    Path(settings.bm25_model_path).parent.mkdir(parents=True, exist_ok=True)
    encoder.dump(settings.bm25_model_path)
    print(f"  BM25 encoder fitted on {len(corpus)} docs → {settings.bm25_model_path}")
    return encoder


@lru_cache(maxsize=1)
def get_bm25_encoder() -> BM25Encoder:
    """Load saved BM25 encoder. Must call fit_bm25() during ingest first."""
    path = Path(settings.bm25_model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 model not found at {path}. Run `python scripts/ingest.py` first."
        )
    encoder = BM25Encoder()
    encoder.load(str(path))
    return encoder


def encode_document(text: str) -> dict:
    return get_bm25_encoder().encode_documents(text)


def encode_query(text: str) -> dict:
    return get_bm25_encoder().encode_queries(text)