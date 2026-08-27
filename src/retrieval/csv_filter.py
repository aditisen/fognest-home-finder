from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import settings
from ..models import SearchFilters


@lru_cache(maxsize=1)
def _load_dataframe() -> pd.DataFrame:
    path = Path(settings.listings_csv_path)
    df = pd.read_csv(path, dtype={"mls_listing_number": str})
    df["list_price"] = pd.to_numeric(df["list_price"], errors="coerce")
    df["total_bedrooms"] = pd.to_numeric(df["total_bedrooms"], errors="coerce")
    df["total_bathrooms"] = pd.to_numeric(df["total_bathrooms"], errors="coerce")
    df["total_square_feet"] = pd.to_numeric(df["total_square_feet"], errors="coerce")
    return df


def get_dataframe() -> pd.DataFrame:
    return _load_dataframe()


def filter_listings(filters: SearchFilters) -> list[str]:
    """
    Apply structured filters to the listings DataFrame.
    Returns a list of mls_listing_numbers that pass all filters.
    """
    df = get_dataframe().copy()
    mask = pd.Series(True, index=df.index)

    if filters.price_min is not None:
        mask &= df["list_price"] >= filters.price_min
    if filters.price_max is not None:
        mask &= df["list_price"] <= filters.price_max
    if filters.bedrooms_min is not None:
        mask &= df["total_bedrooms"] >= filters.bedrooms_min
    if filters.bedrooms_max is not None:
        mask &= df["total_bedrooms"] <= filters.bedrooms_max
    if filters.bathrooms_min is not None:
        mask &= df["total_bathrooms"] >= filters.bathrooms_min
    if filters.sqft_min is not None:
        mask &= df["total_square_feet"] >= filters.sqft_min
    if filters.sqft_max is not None:
        mask &= df["total_square_feet"] <= filters.sqft_max
    if filters.property_type:
        mask &= df["property_type"].str.lower() == filters.property_type.lower()
    if filters.neighborhood_name:
        mask &= df["neighborhood_name"].str.lower().str.contains(
            filters.neighborhood_name.lower(), na=False
        )
    if filters.architecture_style:
        mask &= df["architecture_style"].str.lower().str.contains(
            filters.architecture_style.lower(), na=False
        )
    if filters.views and filters.views.lower() != "none":
        mask &= df["views"].str.lower().str.contains(
            filters.views.lower(), na=False
        )

    return df.loc[mask, "mls_listing_number"].tolist()


def relax_filters(filters: SearchFilters) -> SearchFilters:
    """Loosen filters for a retry: widen price/sqft, drop style and view constraints."""
    data = filters.model_dump()
    if data.get("price_max"):
        data["price_max"] = int(data["price_max"] * 1.20)
    if data.get("price_min"):
        data["price_min"] = int(data["price_min"] * 0.80)
    if data.get("sqft_min"):
        data["sqft_min"] = int(data["sqft_min"] * 0.85)
    data["views"] = None
    data["architecture_style"] = None
    return SearchFilters(**data)
