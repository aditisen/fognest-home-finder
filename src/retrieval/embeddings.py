from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from ..config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key,
    )
