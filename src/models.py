from typing import Optional
from pydantic import BaseModel, Field


class Listing(BaseModel):
    mls_listing_number: str
    address: str
    list_price: float
    total_bedrooms: int
    total_bathrooms: float
    total_square_feet: int
    property_type: str
    architecture_style: str
    neighborhood_name: str
    school_district_name: str
    elementary_school_name: str
    middle_school_name: str
    high_school_name: str
    listing_description: str
    views: str


class SearchFilters(BaseModel):
    """Structured search criteria extracted from a buyer's natural-language question."""
    semantic_query: str = Field(
        description="Distilled intent phrase for semantic similarity search against listing descriptions. "
                    "Capture qualitative attributes (style, feel, features) NOT covered by the structured filters below."
    )
    price_min: Optional[float] = Field(None, description="Minimum list price in USD")
    price_max: Optional[float] = Field(None, description="Maximum list price in USD")
    bedrooms_min: Optional[int] = Field(None, description="Minimum number of bedrooms")
    bedrooms_max: Optional[int] = Field(None, description="Maximum number of bedrooms")
    bathrooms_min: Optional[float] = Field(None, description="Minimum number of bathrooms")
    sqft_min: Optional[int] = Field(None, description="Minimum square footage")
    sqft_max: Optional[int] = Field(None, description="Maximum square footage")
    property_type: Optional[str] = Field(
        None,
        description="One of: Single Family, Condo, Townhouse, Multi Family"
    )
    architecture_style: Optional[str] = Field(
        None,
        description="Architecture style e.g. Victorian, Edwardian, Craftsman, Contemporary, Mid-Century Modern"
    )
    neighborhood_name: Optional[str] = Field(
        None,
        description="SF neighborhood name e.g. Noe Valley, Mission District, Pacific Heights"
    )
    views: Optional[str] = Field(
        None,
        description="View type e.g. Bay View, City View, Ocean View, Park View, Garden View"
    )


class GradeResult(BaseModel):
    """Relevance grade for a single listing document."""
    relevant: bool
    reason: str


class GenerationOutput(BaseModel):
    """Structured output from the generate node."""
    answer: str = Field(description="Markdown-formatted answer citing listings by mls_listing_number")
    cited_listing_ids: list[str] = Field(description="mls_listing_numbers referenced in the answer")
