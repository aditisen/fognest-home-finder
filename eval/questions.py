"""
Evaluation question dataset for the SF HomeFinder RAG pipeline.
Each question is tagged with the search dimensions it exercises so
results can be sliced by category.
"""

QUESTIONS = [
    # ── Neighborhood + property type ──────────────────────────────────────
    {
        "id": "Q01",
        "question": "I'm looking for a 3-bedroom Victorian home in Noe Valley. What's available?",
        "tags": ["neighborhood", "architecture_style", "bedrooms"],
        "expect_results": True,
    },
    {
        "id": "Q02",
        "question": "Show me condos in SOMA with city views under $1.2 million.",
        "tags": ["neighborhood", "property_type", "views", "price"],
        "expect_results": True,
    },
    {
        "id": "Q03",
        "question": "Are there any townhouses in Pacific Heights with at least 2 bathrooms?",
        "tags": ["neighborhood", "property_type", "bathrooms"],
        "expect_results": True,
    },
    {
        "id": "Q04",
        "question": "Find me a single family home in Bernal Heights with a backyard.",
        "tags": ["neighborhood", "property_type", "amenities"],
        "expect_results": True,
    },
    {
        "id": "Q05",
        "question": "What listings are available in the Mission District under $1.5 million?",
        "tags": ["neighborhood", "price"],
        "expect_results": True,
    },

    # ── Price range focused ────────────────────────────────────────────────
    {
        "id": "Q06",
        "question": "Show me homes under $800,000 in San Francisco. I'm a first-time buyer.",
        "tags": ["price", "first_time_buyer"],
        "expect_results": True,
    },
    {
        "id": "Q07",
        "question": "I have a budget of $3 million and want a luxury home in Pacific Heights or Russian Hill.",
        "tags": ["price", "neighborhood", "luxury"],
        "expect_results": True,
    },
    {
        "id": "Q08",
        "question": "What 2-bedroom condos are available between $900k and $1.4 million?",
        "tags": ["price", "bedrooms", "property_type"],
        "expect_results": True,
    },

    # ── Views and amenities ───────────────────────────────────────────────
    {
        "id": "Q09",
        "question": "I want a home with bay views and a parking space.",
        "tags": ["views", "amenities"],
        "expect_results": True,
    },
    {
        "id": "Q10",
        "question": "Find listings with ocean views in the Outer Sunset or Richmond district.",
        "tags": ["views", "neighborhood"],
        "expect_results": True,
    },
    {
        "id": "Q11",
        "question": "Are there any homes with a rooftop deck and city views in Hayes Valley?",
        "tags": ["neighborhood", "amenities", "views"],
        "expect_results": True,
    },

    # ── Architecture style ─────────────────────────────────────────────────
    {
        "id": "Q12",
        "question": "I love Craftsman-style homes. What's available in San Francisco?",
        "tags": ["architecture_style"],
        "expect_results": True,
    },
    {
        "id": "Q13",
        "question": "Show me modern contemporary homes with open floor plans.",
        "tags": ["architecture_style", "amenities"],
        "expect_results": True,
    },
    {
        "id": "Q14",
        "question": "Are there any Edwardian homes available in the Marina or Cow Hollow area?",
        "tags": ["architecture_style", "neighborhood"],
        "expect_results": True,
    },

    # ── Schools ────────────────────────────────────────────────────────────
    {
        "id": "Q15",
        "question": "I have two school-age kids. Show me family-friendly homes near good elementary schools.",
        "tags": ["schools", "family"],
        "expect_results": True,
    },
    {
        "id": "Q16",
        "question": "What homes are in the Noe Valley area near good middle schools?",
        "tags": ["schools", "neighborhood"],
        "expect_results": True,
    },

    # ── Square footage / size ─────────────────────────────────────────────
    {
        "id": "Q17",
        "question": "I need at least 2,500 square feet. Show me spacious homes in San Francisco.",
        "tags": ["sqft"],
        "expect_results": True,
    },
    {
        "id": "Q18",
        "question": "What are the largest homes available in Pacific Heights or Sea Cliff?",
        "tags": ["sqft", "neighborhood", "luxury"],
        "expect_results": True,
    },

    # ── Multi-dimensional / complex ───────────────────────────────────────
    {
        "id": "Q19",
        "question": "Looking for a 4-bedroom Victorian or Edwardian with a garage and city views under $3 million in Noe Valley or Castro.",
        "tags": ["bedrooms", "architecture_style", "amenities", "views", "price", "neighborhood"],
        "expect_results": True,
    },
    {
        "id": "Q20",
        "question": "I want a renovated kitchen and spa-like bathroom. What luxury condos are available in SOMA or Mission Bay?",
        "tags": ["amenities", "property_type", "neighborhood", "luxury"],
        "expect_results": True,
    },

    # ── Edge cases ────────────────────────────────────────────────────────
    {
        "id": "Q21",
        "question": "Are there any penthouses available in San Francisco?",
        "tags": ["luxury", "edge_case"],
        "expect_results": False,  # may not exist in synthetic data
    },
    {
        "id": "Q22",
        "question": "I'm looking for a home with a swimming pool and tennis court under $1 million.",
        "tags": ["amenities", "price", "edge_case"],
        "expect_results": False,  # unlikely match
    },
]
