#!/usr/bin/env python3
"""
Fetch listing photos from Unsplash and assign them to synthetic SF listings.

Strategy:
  - Run ~14 Unsplash search queries grouped by property_type + architecture_style
  - Download each result photo to data/photos/ (CDN downloads don't count toward API limits)
  - Assign a pool of 5-10 photos per listing (cycling through the type pool)
  - Write data/listing_photos.csv with mls_listing_number, main_photo, photo columns

Usage:
  python scripts/fetch_photos.py
  python scripts/fetch_photos.py --listings data/listings.csv --photos-dir data/photos
"""

import argparse
import csv
import time
import sys
from collections import defaultdict
from pathlib import Path

import requests

UNSPLASH_ACCESS_KEY = "HsxR8Om9VqYOH1nnnI7a5FHcpXz7WRe0sop8vIaX7n0"
UNSPLASH_SEARCH_URL = "https://api.unsplash.com/search/photos"
HEADERS = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}

# ---------------------------------------------------------------------------
# Search queries per property_type (and arch style where it helps)
# More queries = more variety in the photo pool
# ---------------------------------------------------------------------------

QUERIES = {
    "Single Family": [
        ("sfh", "Victorian house San Francisco exterior"),
        ("sfh", "craftsman bungalow exterior California"),
        ("sfh", "Edwardian home exterior San Francisco"),
        ("sfh", "single family home exterior suburban California"),
    ],
    "Condo": [
        ("condo", "modern condo interior living room"),
        ("condo", "luxury apartment interior kitchen"),
        ("condo", "contemporary apartment bedroom interior"),
        ("condo", "high rise apartment San Francisco interior"),
    ],
    "Townhouse": [
        ("ths", "modern townhouse exterior urban"),
        ("ths", "townhouse interior living room contemporary"),
        ("ths", "row house exterior city"),
    ],
    "Multi Family": [
        ("mfh", "duplex house exterior California"),
        ("mfh", "apartment building exterior urban"),
        ("mfh", "Victorian flat exterior San Francisco"),
    ],
    "Luxury Bathroom": [
        ("bath", "luxury bathroom renovation marble tile"),
        ("bath", "spa bathroom freestanding bathtub modern"),
        ("bath", "master bathroom walk-in shower glass"),
        ("bath", "designer bathroom double vanity white marble"),
        ("bath", "luxury ensuite bathroom soaking tub"),
        ("bath", "renovated bathroom herringbone tile"),
        ("bath", "contemporary bathroom rainfall shower head"),
    ],
    "Luxury Kitchen": [
        ("kit", "luxury kitchen renovation quartz countertop"),
        ("kit", "gourmet kitchen island stainless steel appliances"),
        ("kit", "modern open concept kitchen white cabinets"),
        ("kit", "chef kitchen professional range marble backsplash"),
        ("kit", "designer kitchen renovation shaker cabinets"),
        ("kit", "luxury kitchen waterfall island pendant lights"),
        ("kit", "renovated kitchen farmhouse sink subway tile"),
    ],
}

# Map filename prefix → pool key
PREFIX_TO_POOL = {
    "sfh":   "Single Family",
    "condo": "Condo",
    "ths":   "Townhouse",
    "mfh":   "Multi Family",
    "bath":  "Luxury Bathroom",
    "kit":   "Luxury Kitchen",
}

# Normalise CSV property_type values to pool keys
PT_NORMALIZE = {
    "single-family home": "Single Family",
    "single family home": "Single Family",
    "single family":      "Single Family",
    "condo":              "Condo",
    "condominium":        "Condo",
    "townhouse":          "Townhouse",
    "multi-family":       "Multi Family",
    "multi family":       "Multi Family",
    "multifamily":        "Multi Family",
}

# ---------------------------------------------------------------------------
# Unsplash helpers
# ---------------------------------------------------------------------------

def search_unsplash(query: str, per_page: int = 20, page: int = 1) -> list[dict]:
    """Return list of Unsplash photo objects for a query."""
    params = {"query": query, "per_page": per_page, "page": page, "orientation": "landscape"}
    resp = requests.get(UNSPLASH_SEARCH_URL, headers=HEADERS, params=params, timeout=15)
    if resp.status_code == 429:
        print("  Rate limited — waiting 60s...")
        time.sleep(60)
        resp = requests.get(UNSPLASH_SEARCH_URL, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("results", [])


def download_photo(url: str, dest: Path) -> bool:
    """Download a photo from a CDN URL to dest. Returns True on success."""
    if dest.exists():
        return True
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"    Warning: failed to download {url}: {e}")
        return False

# ---------------------------------------------------------------------------
# Build photo pools
# ---------------------------------------------------------------------------

def build_photo_pool(photos_dir: Path) -> dict[str, list[str]]:
    """
    Run Unsplash searches, download photos, return pool dict:
      { property_type -> [filename, filename, ...] }
    """
    photos_dir.mkdir(parents=True, exist_ok=True)
    pool: dict[str, list[str]] = defaultdict(list)

    total_queries = sum(len(v) for v in QUERIES.values())
    done = 0

    for prop_type, query_list in QUERIES.items():
        print(f"\n[{prop_type}] fetching photo pool...")
        for prefix, query in query_list:
            done += 1
            print(f"  ({done}/{total_queries}) Searching: \"{query}\"")
            try:
                photos = search_unsplash(query, per_page=20)
            except Exception as e:
                print(f"    Search failed: {e}")
                time.sleep(2)
                continue

            print(f"    Found {len(photos)} results — downloading...")
            for i, photo in enumerate(photos):
                url = photo.get("urls", {}).get("regular")
                if not url:
                    continue
                # Stable filename based on Unsplash photo ID
                photo_id = photo.get("id", f"{prefix}_{done}_{i:03d}")
                filename = f"{prefix}_{photo_id}.jpg"
                dest = photos_dir / filename
                if download_photo(url, dest):
                    pool[prop_type].append(filename)
                    print(f"    ✓ {filename}")
                else:
                    print(f"    ✗ skipped {filename}")
                time.sleep(0.1)  # gentle CDN pacing

            # Small pause between API search calls
            time.sleep(1.5)

    print(f"\nPhoto pool summary:")
    for pt, files in pool.items():
        print(f"  {pt}: {len(files)} photos")

    return dict(pool)

# ---------------------------------------------------------------------------
# Build pool from already-downloaded files (no API calls)
# ---------------------------------------------------------------------------

def build_pool_from_disk(photos_dir: Path) -> dict[str, list[str]]:
    """
    Scan photos_dir for *.jpg files and group them by filename prefix.
    Returns the same pool dict shape as build_photo_pool() but without
    hitting the Unsplash API.
    """
    pool: dict[str, list[str]] = defaultdict(list)
    for f in sorted(photos_dir.glob("*.jpg")):
        prefix = f.stem.split("_")[0]
        ptype = PREFIX_TO_POOL.get(prefix)
        if ptype:
            pool[ptype].append(f.name)

    print("Photo pool from disk:")
    for pt, files in pool.items():
        print(f"  {pt}: {len(files)} photos")
    return dict(pool)


# ---------------------------------------------------------------------------
# Assign photos to listings
# ---------------------------------------------------------------------------

def assign_photos(listings: list[dict], pool: dict[str, list[str]]) -> list[dict]:
    """
    Assignment rules — one style per category per listing:
      1. ONE exterior/property photo from the property-type pool  → main photo
      2. ONE bathroom photo from Luxury Bathroom pool
      3. ONE kitchen photo from Luxury Kitchen pool

    Each pool is cycled independently so photos are reused across listings,
    but no single listing ever receives more than one exterior style,
    one bathroom style, or one kitchen style.
    """
    bath_pool = pool.get("Luxury Bathroom", [])
    kit_pool  = pool.get("Luxury Kitchen",  [])
    pool_idx: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    skipped = 0

    for listing in listings:
        mls   = listing["mls_listing_number"]
        raw_pt = listing.get("property_type", "")
        pt    = PT_NORMALIZE.get(raw_pt.lower().strip(), raw_pt)
        prop_pool = pool.get(pt, [])

        photos: list[tuple[str, bool]] = []  # (filename, is_main)

        # 1 — exterior / property lifestyle photo (main photo)
        if prop_pool:
            idx = pool_idx[pt] % len(prop_pool)
            pool_idx[pt] += 1
            photos.append((prop_pool[idx], True))
        else:
            skipped += 1
            print(f"  Warning: no property pool for {raw_pt!r}, skipping {mls}")
            continue

        # 2 — one bathroom photo
        if bath_pool:
            idx = pool_idx["bath"] % len(bath_pool)
            pool_idx["bath"] += 1
            photos.append((bath_pool[idx], False))

        # 3 — one kitchen photo
        if kit_pool:
            idx = pool_idx["kit"] % len(kit_pool)
            pool_idx["kit"] += 1
            photos.append((kit_pool[idx], False))

        for filename, is_main in photos:
            rows.append({
                "mls_listing_number": mls,
                "main_photo": is_main,
                "photo": filename,
            })

    if skipped:
        print(f"  Skipped {skipped} listings with no matching property pool.")
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch Unsplash photos for SF listings")
    parser.add_argument("--listings",      default="data/listings.csv",       help="Path to listings CSV")
    parser.add_argument("--photos-dir",    default="data/photos",             help="Directory to save photos")
    parser.add_argument("--output",        default="data/listing_photos.csv", help="Output photos CSV path")
    parser.add_argument("--download-only", action="store_true",
                        help="Download photos to photos-dir only — skip assignment and CSV update")
    parser.add_argument("--assign-only", action="store_true",
                        help="Reassign existing on-disk photos to listings — no Unsplash API calls")
    args = parser.parse_args()

    photos_dir  = Path(args.photos_dir)
    output_path = Path(args.output)

    if args.assign_only:
        print(f"Building pool from existing photos in {photos_dir}...")
        pool = build_pool_from_disk(photos_dir)
    else:
        pool = build_photo_pool(photos_dir)

    if not any(pool.values()):
        print("No photos found. Run without --assign-only first to download them.")
        sys.exit(1)

    if args.download_only:
        total = sum(len(v) for v in pool.values())
        print(f"\nDownload-only mode — {total} photos in {photos_dir}. listing_photos.csv not changed.")
        return

    listings_path = Path(args.listings)
    if not listings_path.exists():
        print(f"Error: {listings_path} not found. Run generate_synthetic_data.py first.")
        sys.exit(1)

    with open(listings_path, newline="", encoding="utf-8") as f:
        listings = list(csv.DictReader(f))
    print(f"Loaded {len(listings)} listings from {listings_path}")

    # Assign photos to listings
    print("\nAssigning photos to listings...")
    photo_rows = assign_photos(listings, pool)
    print(f"  Generated {len(photo_rows)} photo assignments")

    # Write listing_photos.csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mls_listing_number", "main_photo", "photo"])
        writer.writeheader()
        writer.writerows(photo_rows)
    print(f"  Saved → {output_path}")

    main_count = sum(1 for r in photo_rows if r["main_photo"])
    print(f"\nDone. {len(photo_rows)} total photo rows, {main_count} main photos.")


if __name__ == "__main__":
    main()