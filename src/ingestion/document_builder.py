from langchain_core.documents import Document
from ..models import Listing


def listing_to_document(listing: Listing) -> Document:
    """
    Convert a Listing to a LangChain Document.
    page_content = listing_description (what gets embedded).
    metadata     = all structured fields (used for Pinecone filtering and display).
    """
    return Document(
        page_content=listing.listing_description,
        metadata={
            "mls_listing_number": listing.mls_listing_number,
            "address": listing.address,
            "list_price": listing.list_price,
            "total_bedrooms": listing.total_bedrooms,
            "total_bathrooms": listing.total_bathrooms,
            "total_square_feet": listing.total_square_feet,
            "property_type": listing.property_type,
            "architecture_style": listing.architecture_style,
            "neighborhood_name": listing.neighborhood_name,
            "school_district_name": listing.school_district_name,
            "elementary_school_name": listing.elementary_school_name,
            "middle_school_name": listing.middle_school_name,
            "high_school_name": listing.high_school_name,
            "views": listing.views,
            "listing_description": listing.listing_description,
        },
    )


def build_documents(listings: list[Listing]) -> list[Document]:
    return [listing_to_document(l) for l in listings]
