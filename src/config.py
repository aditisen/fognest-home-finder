from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str
    openai_llm_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072

    pinecone_api_key: str
    pinecone_index_name: str = "sf-real-estate-listings"
    pinecone_photos_index_name: str = "sf-real-estate-photos"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_dimension: int = 512
    bm25_model_path: str = "data/bm25_model.json"
    hybrid_alpha: float = 0.7   # 0 = pure BM25, 1 = pure dense

    listings_csv_path: str = "data/listings.csv"
    photos_csv_path: str = "data/listing_photos.csv"
    photos_dir: str = "data/photos"

    retrieval_k: int = 8
    relevance_threshold: int = 3
    max_retries: int = 1

    class Config:
        env_file = ".env"


settings = Settings()
