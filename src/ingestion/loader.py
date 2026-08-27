import csv
from pathlib import Path
from ..models import Listing


def load_listings(csv_path: str | Path) -> list[Listing]:
    """Load and validate listings from CSV. Skips rows with missing description."""
    listings = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("listing_description", "").strip():
                continue
            try:
                listings.append(Listing(
                    mls_listing_number=row["mls_listing_number"],
                    address=row["address"],
                    list_price=float(row["list_price"]),
                    total_bedrooms=int(row["total_bedrooms"]),
                    total_bathrooms=float(row["total_bathrooms"]),
                    total_square_feet=int(row["total_square_feet"]),
                    property_type=row["property_type"],
                    architecture_style=row.get("architecture_style", ""),
                    neighborhood_name=row["neighborhood_name"],
                    school_district_name=row["school_district_name"],
                    elementary_school_name=row["elementary_school_name"],
                    middle_school_name=row["middle_school_name"],
                    high_school_name=row["high_school_name"],
                    listing_description=row["listing_description"].strip(),
                    views=row.get("views", "None"),
                ))
            except (KeyError, ValueError):
                continue
    return listings
