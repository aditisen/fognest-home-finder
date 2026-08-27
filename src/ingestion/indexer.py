import time
from pathlib import Path

import pandas as pd
from pinecone import Pinecone, ServerlessSpec

from ..config import settings
from ..retrieval.embeddings import get_embeddings
from ..retrieval.bm25_encoder import fit_bm25, build_corpus_text, encode_document
from ..retrieval.clip_embeddings import embed_image


# ---------------------------------------------------------------------------
# Pinecone client
# ---------------------------------------------------------------------------

def _pinecone() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def _wait_ready(pc: Pinecone, name: str) -> None:
    while not pc.describe_index(name).status.get("ready", False):
        time.sleep(1)


def _create_index(pc: Pinecone, name: str, dimension: int, metric: str, recreate: bool) -> None:
    existing = [idx.name for idx in pc.list_indexes()]
    if recreate and name in existing:
        print(f"  Deleting '{name}'...")
        pc.delete_index(name)
        existing = []
    if name not in existing:
        print(f"  Creating '{name}' (dim={dimension}, metric={metric})...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        _wait_ready(pc, name)
        print(f"  '{name}' ready.")
    else:
        print(f"  '{name}' already exists.")


# ---------------------------------------------------------------------------
# Listings index  (hybrid: dotproduct, 3072-dim)
# ---------------------------------------------------------------------------

def upsert_listings(pc: Pinecone, listings_csv: str, batch_size: int = 50) -> int:
    """
    Embed listing_description (OpenAI dense) + structured fields (BM25 sparse).
    Upserts to the listings index.  Requires dotproduct metric for hybrid search.
    """
    from ..ingestion.loader import load_listings
    from ..ingestion.document_builder import build_documents

    listings = load_listings(listings_csv)
    docs = build_documents(listings)
    embeddings = get_embeddings()
    index = pc.Index(settings.pinecone_index_name)

    # Pre-build BM25 corpus texts aligned with docs
    df = pd.read_csv(listings_csv, dtype={"mls_listing_number": str})
    bm25_text_map = {
        row["mls_listing_number"]: build_corpus_text(row)
        for _, row in df.iterrows()
    }

    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d.page_content for d in batch]
        dense_vecs = embeddings.embed_documents(texts)

        vectors = []
        for doc, dense in zip(batch, dense_vecs):
            mls = doc.metadata["mls_listing_number"]
            bm25_text = bm25_text_map.get(mls, "")
            sparse = encode_document(bm25_text)
            vectors.append({
                "id": mls,
                "values": dense,
                "sparse_values": {
                    "indices": sparse["indices"],
                    "values": sparse["values"],
                },
                "metadata": doc.metadata,
            })

        index.upsert(vectors=vectors)
        total += len(batch)
        print(f"  Listings upserted: {total}/{len(docs)}")

    return total


# ---------------------------------------------------------------------------
# Photos index  (dense CLIP, 512-dim, cosine)
# ---------------------------------------------------------------------------

def upsert_photos(pc: Pinecone, photos_csv: str, photos_dir: str, batch_size: int = 50) -> int:
    """
    Embed every photo in listing_photos.csv with CLIP.
    Vector ID = {mls_listing_number}_{photo_filename_stem}
    Cache CLIP embeddings by filename so each unique image is embedded once.
    """
    photos_path = Path(photos_csv)
    photos_dir_path = Path(photos_dir)

    if not photos_path.exists():
        print(f"  Photos CSV not found at {photos_path}, skipping photo index.")
        return 0

    df = pd.read_csv(photos_path, dtype={"mls_listing_number": str, "photo": str})
    index = pc.Index(settings.pinecone_photos_index_name)

    # Cache: filename → clip embedding (each unique file embedded once)
    embedding_cache: dict[str, list[float]] = {}
    vectors = []
    skipped = 0

    for _, row in df.iterrows():
        mls = str(row["mls_listing_number"])
        filename = str(row["photo"])
        is_main = str(row.get("main_photo", "False")).lower() == "true"
        photo_path = photos_dir_path / filename

        if not photo_path.exists():
            skipped += 1
            continue

        if filename not in embedding_cache:
            try:
                embedding_cache[filename] = embed_image(photo_path)
            except Exception as e:
                print(f"  Warning: could not embed {filename}: {e}")
                skipped += 1
                continue

        vector_id = f"{mls}_{Path(filename).stem}"
        vectors.append({
            "id": vector_id,
            "values": embedding_cache[filename],
            "metadata": {
                "mls_listing_number": mls,
                "photo_filename": filename,
                "is_main_photo": is_main,
            },
        })

    print(f"  {len(vectors)} photo vectors prepared ({skipped} skipped — file missing).")
    print(f"  {len(embedding_cache)} unique images embedded via CLIP.")

    total = 0
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])
        total += len(vectors[i : i + batch_size])
        print(f"  Photos upserted: {total}/{len(vectors)}")

    return total


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_ingest(listings_csv: str, recreate: bool = False) -> None:
    pc = _pinecone()

    # 1. Fit BM25 on structured fields (saves model to data/bm25_model.json)
    print("\n[1/4] Fitting BM25 encoder on structured listing fields...")
    fit_bm25(listings_csv)

    # 2. Listings index — hybrid (dotproduct, 3072-dim)
    print("\n[2/4] Setting up listings index (hybrid: dense + BM25)...")
    _create_index(pc, settings.pinecone_index_name, settings.embedding_dimension, "dotproduct", recreate)
    print("Upserting listings...")
    n_listings = upsert_listings(pc, listings_csv)
    print(f"  Done: {n_listings} listing vectors.")

    # 3. Photos index — CLIP dense (cosine, 512-dim)
    print("\n[3/4] Setting up photos index (CLIP dense, 512-dim)...")
    _create_index(pc, settings.pinecone_photos_index_name, settings.clip_dimension, "cosine", recreate)
    print("Embedding and upserting all photos with CLIP...")
    print("  Note: CLIP model (~400 MB) downloads on first run.")
    n_photos = upsert_photos(
        pc,
        settings.photos_csv_path,
        settings.photos_dir,
    )
    print(f"  Done: {n_photos} photo vectors.")

    print("\n[4/4] Ingest complete.")
    print(f"  Listings index : '{settings.pinecone_index_name}' — {n_listings} vectors")
    print(f"  Photos index   : '{settings.pinecone_photos_index_name}' — {n_photos} vectors")


def run_photos_ingest(recreate: bool = False) -> None:
    """Re-index photos only — skips BM25 fit and listings upsert."""
    pc = _pinecone()

    print("\n[1/2] Setting up photos index (CLIP dense, 512-dim)...")
    _create_index(pc, settings.pinecone_photos_index_name, settings.clip_dimension, "cosine", recreate)

    print("\n[2/2] Embedding and upserting all photos with CLIP...")
    print("  Note: CLIP model (~400 MB) downloads on first run.")
    n_photos = upsert_photos(pc, settings.photos_csv_path, settings.photos_dir)
    print(f"\nDone: {n_photos} photo vectors in '{settings.pinecone_photos_index_name}'.")