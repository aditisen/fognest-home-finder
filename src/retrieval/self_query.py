from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from ..models import SearchFilters
from ..llm import get_llm

SF_NEIGHBORHOODS = (
    "Mission District, Noe Valley, Pacific Heights, SOMA, Castro, Haight-Ashbury, "
    "Inner Richmond, Outer Richmond, Inner Sunset, Outer Sunset, Marina, North Beach, "
    "Bernal Heights, Potrero Hill, Hayes Valley, Dogpatch, Glen Park, West Portal, "
    "Russian Hill, Nob Hill, Excelsior, Portola, Bayview, Cole Valley, Duboce Triangle, "
    "Twin Peaks, Financial District, Lower Haight, Forest Hill"
)

PARSE_SYSTEM_PROMPT = f"""You are a real estate search parser for San Francisco, CA listings.

Extract structured search criteria from the buyer's question. Return a SearchFilters object.

Rules:
- semantic_query: distill the qualitative intent (style, feel, features, location character) into a short phrase suitable for embedding similarity search. Do NOT include numeric constraints here.
- Set numeric fields (price, beds, baths, sqft) only when the buyer explicitly states a constraint.
- property_type must be one of: Single Family, Condo, Townhouse, Multi Family — or null if unspecified.
- neighborhood_name must be an SF neighborhood. Known neighborhoods: {SF_NEIGHBORHOODS}.
- If the buyer mentions a landmark (Dolores Park, Golden Gate Park, etc.) infer the nearest neighborhood.
- views: only set if buyer explicitly mentions a view type.
- When the buyer says "under $X" set price_max. "at least X beds" sets bedrooms_min. "around $X" sets price_max = X * 1.1.
"""


def parse_filters(question: str, history: list[BaseMessage] = None) -> SearchFilters:
    """Extract SearchFilters from a buyer question using GPT-4o structured output."""
    llm = get_llm("gpt-4o", temperature=0.0)
    structured_llm = llm.with_structured_output(SearchFilters)

    messages = [SystemMessage(content=PARSE_SYSTEM_PROMPT)]
    if history:
        messages.extend(history[-6:])
    messages.append(HumanMessage(content=question))

    return structured_llm.invoke(messages)
