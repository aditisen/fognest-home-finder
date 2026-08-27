from functools import lru_cache
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..config import settings
from .state import SearchState
from .nodes import parse_node, retrieve_node, grade_node, generate_node


def _should_retry(state: SearchState) -> str:
    if state.get("grade") == "weak" and state.get("retry_count", 0) <= settings.max_retries:
        return "retrieve"
    return "generate"


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(SearchState)

    graph.add_node("parse", parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "parse")
    graph.add_edge("parse", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _should_retry,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=MemorySaver())
