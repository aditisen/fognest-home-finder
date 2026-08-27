from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..models import SearchFilters


class SearchState(TypedDict):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    filters: Optional[SearchFilters]
    candidate_ids: list[str]
    documents: list[dict]       # serialized LangChain Documents {page_content, metadata}
    relevant_docs: list[dict]   # subset graded as relevant
    grade: str                  # "relevant" | "weak"
    answer: str
    cited_listing_ids: list[str]
    retry_count: int
