from functools import lru_cache
from langchain_openai import ChatOpenAI
from .config import settings


@lru_cache(maxsize=4)
def get_llm(model: str = None, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or settings.openai_llm_model,
        temperature=temperature,
        openai_api_key=settings.openai_api_key,
    )
