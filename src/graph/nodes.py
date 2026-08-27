from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ..config import settings
from ..models import SearchFilters, GradeResult, GenerationOutput
from ..llm import get_llm
from ..retrieval.csv_filter import filter_listings, relax_filters
from ..retrieval.vectorstore import retrieve
from ..retrieval.self_query import parse_filters
from .state import SearchState

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

GRADE_SYSTEM = """You are a real estate listing relevance grader.
Given a buyer's question and a property listing description, decide if the listing is relevant.
Return JSON: {{"relevant": true/false, "reason": "one sentence"}}
Be generous — mark relevant if the listing could plausibly interest the buyer."""

GENERATE_SYSTEM = """You are a knowledgeable real estate assistant helping buyers and agents \
find homes in San Francisco, CA.

Use ONLY the listing information provided below. Do not invent or mention any listing not in \
the context. For every listing you reference, cite its MLS number like [MLS: {{id}}].

If the context contains no relevant listings, clearly say you couldn't find matching listings \
and suggest how the buyer might broaden their search.

Format your response in clear markdown. Lead with a brief summary, then present up to 5 \
recommended listings with key highlights from their descriptions.

--- LISTING CONTEXT ---
{context}
-----------------------"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(docs: list[dict]) -> str:
    parts = []
    for doc in docs:
        meta = doc["metadata"]
        parts.append(
            f"[MLS: {meta.get('mls_listing_number')}]\n"
            f"Address: {meta.get('address')}\n"
            f"Price: ${meta.get('list_price', 0):,.0f} | "
            f"{meta.get('total_bedrooms')}BD/{meta.get('total_bathrooms')}BA | "
            f"{meta.get('total_square_feet', 0):,} sqft\n"
            f"Type: {meta.get('property_type')} ({meta.get('architecture_style')})\n"
            f"Neighborhood: {meta.get('neighborhood_name')} | Views: {meta.get('views')}\n"
            f"Schools: {meta.get('elementary_school_name')} / "
            f"{meta.get('middle_school_name')} / {meta.get('high_school_name')}\n"
            f"Description: {doc['page_content']}"
        )
    return "\n\n---\n\n".join(parts)

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_node(state: SearchState) -> dict:
    """Extract structured SearchFilters from the user question."""
    question = state["question"]
    history = state.get("messages", [])
    filters = parse_filters(question, history)
    return {
        "filters": filters,
        "retry_count": 0,
        "messages": [HumanMessage(content=question)],
    }


def retrieve_node(state: SearchState) -> dict:
    """
    Three-signal hybrid retrieval:
      - CSV hard-constraint filter  → candidate listing IDs
      - Pinecone hybrid (OpenAI dense + BM25 sparse) on listing_description / structured fields
      - Pinecone CLIP dense search on ALL listing photos (text-to-image cross-modal)
    Results merged with Reciprocal Rank Fusion.
    """
    filters: SearchFilters = state["filters"]
    retry_count = state.get("retry_count", 0)

    # Relax hard-constraint filters on retry
    active_filters = relax_filters(filters) if retry_count > 0 else filters

    # CSV filter — enforces exact numeric constraints (price, beds, baths, sqft)
    candidate_ids = filter_listings(active_filters)

    if not candidate_ids:
        return {"documents": [], "candidate_ids": []}

    # Three-signal Pinecone retrieval (scoped to CSV candidates)
    docs = retrieve(
        semantic_query=active_filters.semantic_query,
        candidate_ids=candidate_ids,
        k_listings=settings.retrieval_k,
        k_photos=settings.retrieval_k * 3,  # cast wider net on photos
    )

    return {
        "documents": docs,
        "candidate_ids": candidate_ids,
    }


def grade_node(state: SearchState) -> dict:
    """Grade each retrieved document for relevance. Route to retry if hits are weak."""
    docs = state.get("documents", [])
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    if not docs:
        return {
            "relevant_docs": [],
            "grade": "weak",
            "retry_count": retry_count + 1,
        }

    llm = get_llm("gpt-4o-mini", temperature=0.0)
    relevant_docs = []

    for doc in docs:
        try:
            result = llm.with_structured_output(GradeResult).invoke([
                SystemMessage(content=GRADE_SYSTEM),
                HumanMessage(content=f"Buyer question: {question}\n\nListing:\n{doc['page_content'][:800]}"),
            ])
            if result.relevant:
                relevant_docs.append(doc)
        except Exception:
            relevant_docs.append(doc)  # include on parse failure to avoid data loss

    grade = "relevant" if len(relevant_docs) >= settings.relevance_threshold else "weak"

    return {
        "relevant_docs": relevant_docs,
        "grade": grade,
        "retry_count": retry_count + 1,
    }


def generate_node(state: SearchState) -> dict:
    """Generate a grounded answer from relevant listing documents."""
    question = state["question"]
    docs = state.get("relevant_docs") or state.get("documents", [])

    if not docs:
        no_match = (
            "I couldn't find San Francisco listings matching your criteria. "
            "Try broadening your search — for example, widen the price range, "
            "consider adjacent neighborhoods, or relax a specific requirement."
        )
        return {
            "answer": no_match,
            "cited_listing_ids": [],
            "messages": [AIMessage(content=no_match)],
        }

    context = _format_context(docs)
    llm = get_llm("gpt-4o", temperature=0.1)

    result = llm.with_structured_output(GenerationOutput).invoke([
        SystemMessage(content=GENERATE_SYSTEM.format(context=context)),
        HumanMessage(content=question),
    ])

    return {
        "answer": result.answer,
        "cited_listing_ids": result.cited_listing_ids,
        "messages": [AIMessage(content=result.answer)],
    }
