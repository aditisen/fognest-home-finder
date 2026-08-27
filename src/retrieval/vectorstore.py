import logging
import logging.handlers
from functools import lru_cache
from pathlib import Path

from pinecone import Pinecone

from ..config import settings


# ---------------------------------------------------------------------------
# Search logger — writes to logs/search.log, rotates at 10 MB (5 backups)
# ---------------------------------------------------------------------------

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("vectorstore.search")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_path = Path("logs/search.log")
    log_path.parent.mkdir(exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Pinecone client & index handles
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _pinecone() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def _listings_index():
    return _pinecone().Index(settings.pinecone_index_name)


def _photos_index():
    return _pinecone().Index(settings.pinecone_photos_index_name)


# ---------------------------------------------------------------------------
# Hybrid scaling
# ---------------------------------------------------------------------------

def _hybrid_scale(dense: list[float], sparse: dict, alpha: float):
    """
    Scale dense and sparse vectors by alpha for weighted hybrid query.
    alpha=1.0 → pure dense,  alpha=0.0 → pure BM25 sparse.
    """
    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse["indices"],
        "values": [v * (1.0 - alpha) for v in sparse["values"]],
    }
    return scaled_dense, scaled_sparse


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def search_listings(
    dense_vec: list[float],
    sparse_vec: dict,
    candidate_ids: list[str] | None = None,
    k: int = 10,
    alpha: float = None,
) -> list[dict]:
    """
    Hybrid search on the listings index.
    Combines OpenAI dense (listing_description) + BM25 sparse (structured fields).
    Optionally scoped to candidate_ids from the CSV hard-constraint filter.
    Returns list of match dicts with id, score, metadata.
    """
    alpha = alpha if alpha is not None else settings.hybrid_alpha
    scaled_dense, scaled_sparse = _hybrid_scale(dense_vec, sparse_vec, alpha)

    query_kwargs = dict(
        vector=scaled_dense,
        sparse_vector=scaled_sparse,
        top_k=k,
        include_metadata=True,
    )
    if candidate_ids:
        query_kwargs["filter"] = {"mls_listing_number": {"$in": candidate_ids}}

    response = _listings_index().query(**query_kwargs)
    return [
        {"id": m.id, "score": m.score, "metadata": m.metadata}
        for m in response.matches
    ]


def search_photos(clip_vec: list[float], k: int = 30) -> list[dict]:
    """
    Dense CLIP search on the photos index.
    Returns list of match dicts with id, score, metadata (incl. mls_listing_number).
    """
    response = _photos_index().query(
        vector=clip_vec,
        top_k=k,
        include_metadata=True,
    )
    return [
        {"id": m.id, "score": m.score, "metadata": m.metadata}
        for m in response.matches
    ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def rrf_merge(rankings: list[list[str]], k_rrf: int = 60) -> list[str]:
    """
    Merge multiple ranked listing-ID lists with Reciprocal Rank Fusion.
    Higher score = better combined rank.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, listing_id in enumerate(ranking):
            scores[listing_id] = scores.get(listing_id, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ---------------------------------------------------------------------------
# Full retrieval pipeline (called from the LangGraph retrieve node)
# ---------------------------------------------------------------------------

def retrieve(
    semantic_query: str,
    candidate_ids: list[str] | None,
    k_listings: int = 10,
    k_photos: int = 30,
    alpha: float = None,
) -> list[dict]:
    """
    Three-signal hybrid retrieval:
      1. OpenAI dense + BM25 sparse on listing_description / structured fields
      2. CLIP dense on all photos (text-to-image cross-modal)
    Merges with RRF, returns top-k metadata dicts.
    """
    from .embeddings import get_embeddings
    from .bm25_encoder import encode_query
    from .clip_embeddings import embed_text

    log = _get_logger()
    effective_alpha = alpha if alpha is not None else settings.hybrid_alpha
    log.info(
        "QUERY | query=%r | candidates=%d | k_listings=%d | k_photos=%d"
        " | index:listings=%r (dense=%.0f%% BM25=%.0f%%)"
        " | index:photos=%r (CLIP 512-dim cosine)",
        semantic_query,
        len(candidate_ids) if candidate_ids else 0,
        k_listings, k_photos,
        settings.pinecone_index_name,
        effective_alpha * 100, (1 - effective_alpha) * 100,
        settings.pinecone_photos_index_name,
    )

    # Encode query three ways
    dense_vec = get_embeddings().embed_query(semantic_query)
    sparse_vec = encode_query(semantic_query)
    clip_vec = embed_text(semantic_query)

    # Run searches
    listing_hits = search_listings(dense_vec, sparse_vec, candidate_ids, k=k_listings, alpha=alpha)
    photo_hits = search_photos(clip_vec, k=k_photos)

    log.info("HITS  | index:listings=%d hits | index:photos=%d hits",
             len(listing_hits), len(photo_hits))

    for i, h in enumerate(listing_hits):
        log.info("  [listings-hybrid] rank=%d id=%s blended_score=%.4f"
                 " (dense %.0f%% + BM25 %.0f%%)",
                 i, h["id"], h["score"],
                 effective_alpha * 100, (1 - effective_alpha) * 100)

    for i, h in enumerate(photo_hits[:10]):
        log.info("  [photos-clip]     rank=%d id=%s mls=%s clip_score=%.4f",
                 i, h["id"], h["metadata"].get("mls_listing_number", "?"), h["score"])

    # Extract ranked listing ID lists
    text_ranking = [h["id"] for h in listing_hits]
    photo_ranking = list(dict.fromkeys(
        h["metadata"].get("mls_listing_number", "")
        for h in photo_hits
        if h["metadata"].get("mls_listing_number")
    ))

    text_set = set(text_ranking)
    photo_set = set(photo_ranking)
    merged_ids = rrf_merge([text_ranking, photo_ranking])

    log.info("RRF   | merged=%d unique listings | signals per top result:", len(merged_ids))
    for i, lid in enumerate(merged_ids[:8]):
        signals = []
        if lid in text_set:
            signals.append("listings-hybrid(dense+BM25)")
        if lid in photo_set:
            signals.append("photos-clip")
        log.info("  rrf[%d] id=%s signals=[%s]", i, lid, ", ".join(signals))

    # Build output dicts from metadata we already have (avoid a second Pinecone fetch)
    meta_by_id = {h["id"]: h["metadata"] for h in listing_hits}
    results = []
    for listing_id in merged_ids:
        meta = meta_by_id.get(listing_id)
        if meta:
            results.append({"page_content": meta.get("listing_description", ""), "metadata": meta})

    log.info("RESULT| returned=%d docs to LangGraph", len(results))
    return results