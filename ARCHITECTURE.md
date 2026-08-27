# FogNest — Architecture & Design

## 1. System Overview

FogNest is a retrieval-augmented generation application scoped to **San Francisco, CA**
real estate listings. Listing data lives in a CSV file. The `listing_description` column —
free-text written by listing agents — is the primary semantic search surface because agents
encode detail there (neighborhood character, finishes, light, micro-location) that never
appears in structured fields alone.

The retrieval pipeline is **hybrid with three signals**:
1. A pandas-based structured filter on CSV columns narrows to hard-constraint candidates.
2. A Pinecone listings index (dense `text-embedding-3-large` + BM25, `dotproduct`) ranks by description match.
3. A Pinecone photos index (CLIP `openai/clip-vit-base-patch32`, `cosine`) ranks by visual similarity.

Signals 2 and 3 are merged via **Reciprocal Rank Fusion (RRF)** before grading and generation.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                    Streamlit UI                        │
                    │  ┌──────────────────────┐  ┌────────────────────────┐ │
                    │  │ Find your dream homes │  │   Agent Dashboard      │ │
                    │  │  (chat + carousel)    │  │  (filters + table)     │ │
                    │  └──────────┬────────────┘  └──────────┬─────────────┘ │
                    └────────────┼───────────────────────────┼───────────────┘
                                 │ user question              │
                                 ▼                            ▼
                    ┌────────────────────────────────────────────────────────┐
                    │              LangGraph State Machine                   │
                    │                                                        │
                    │  [parse] ──► [retrieve] ──► [grade] ──► [generate]    │
                    │                  │              │                      │
                    │         ┌────────┴──────┐   weak? retry once          │
                    │         │               │                              │
                    │    CSV filter     ┌─────┴──────────────────┐          │
                    │    (pandas)       │  Pinecone (two indexes) │          │
                    │                   │  listings: dense + BM25 │          │
                    │                   │  photos:   CLIP cosine  │          │
                    │                   │  merged via RRF         │          │
                    │                   └────────────────────────-┘          │
                    └────────────────────────────────────────────────────────┘
                          │                        │
          ┌───────────────┘                        └────────────────────┐
          ▼                                                              ▼
┌──────────────────┐                                        ┌───────────────────────┐
│  OpenAI GPT-4o   │                                        │  listings.csv (pandas)│
│  - filter parse  │                                        │  structured columns   │
│  - generation    │                                        │  → candidate IDs      │
│  GPT-4o-mini     │                                        └───────────┬───────────┘
│  - relevance     │                                                    │ listing_ids
│    grading       │                            ┌───────────────────────┴───────────────────────┐
└──────────────────┘                            │                                               │
                                    ┌───────────▼───────────┐               ┌───────────────────▼──┐
                                    │ Pinecone: listings     │               │ Pinecone: photos      │
                                    │ dim=3072, dotproduct   │               │ dim=512, cosine       │
                                    │ dense + BM25 hybrid    │               │ CLIP visual search    │
                                    │ scoped to candidate IDs│               │ → listing_ids         │
                                    └───────────────────────┘               └──────────────────────┘
                                                    └──────────────┬─────────────────┘
                                                                   ▼
                                                          RRF merge → ranked listings
```

---

## 2. Geographic Scope

All listings are within **San Francisco, CA city limits**. This is enforced at:

- **Ingest time:** rows outside SF are rejected by `loader.py` (filter on `city == "San Francisco"`).
- **UI copy:** searches, placeholders, and help text reference SF neighborhoods explicitly
  (Mission, Noe Valley, Pacific Heights, SOMA, etc.).
- **Self-query parser:** the neighborhood extraction list is seeded with SF neighborhoods.

---

## 3. Data Model

### 3.1 CSV Source — Listings

The listings CSV (`data/listings.csv`) is the system of record.

> **Data source — synthetic:** Listings data was **synthetically generated by Claude** to
> approximate plausible real estate inventory across San Francisco neighborhoods. Addresses,
> MLS numbers, prices, square footage, and listing descriptions are fabricated. They are
> loosely modeled on the character of real SF properties and neighborhoods, but **property
> details, features, and prices should be considered fictional and are likely incorrect**.
> This dataset exists solely to demonstrate the RAG pipeline — it is not sourced from any
> real MLS feed and must not be used for actual real estate decisions.

| Column | Type | Role |
|---|---|---|
| `mls_listing_number` | string | Primary key / vector ID in Pinecone |
| `listing_description` | string | Agent free-text. **Embedded into Pinecone listings index.** |
| `address` | string | Display |
| `list_price` | float | Hard filter (price_max) |
| `total_bedrooms` | int | Hard filter (bedrooms_min) |
| `total_bathrooms` | float | Hard filter (bathrooms_min) |
| `total_square_feet` | int | Hard filter (sqft_min) |
| `neighborhood_name` | string | Hard filter + display |
| `property_type` | string | Hard filter (condo, SFH, townhouse, etc.) |
| `views` | string | Semantic + display |
| `architecture_style` | string | Semantic + display |
| `parking` | string | Semantic + display |

### 3.2 CSV Source — Photos

The photos CSV (`data/listing_photos.csv`) maps downloaded Unsplash photos to listings.

> **Data source — Unsplash:** Photos are downloaded from the [Unsplash API](https://unsplash.com/developers)
> using search queries for exterior property styles, luxury bathrooms, and luxury kitchens.
> They are stock images and are **not photos of the synthetically generated listings** — they
> are assigned to listings to illustrate the carousel UI, not to represent actual properties.

| Column | Type | Role |
|---|---|---|
| `mls_listing_number` | string | FK to listings |
| `photo` | string | Filename in `data/photos/` |
| `main_photo` | bool | If true, shown first in carousel |
| `photo_type` | string | `ext` (exterior), `bath`, `kit` (kitchen) |

Assignment rule: max 1 exterior + 1 bathroom + 1 kitchen photo per listing.

### 3.3 What Gets Embedded

| Index | Input | Model | Dimension | Metric |
|---|---|---|---|---|
| `sf-real-estate-listings` | `listing_description` (text) | `text-embedding-3-large` | 3072 | `dotproduct` |
| `sf-real-estate-photos` | Photo file (image) | CLIP `openai/clip-vit-base-patch32` | 512 | `cosine` |

At query time, the listings index receives a dense text query + BM25 sparse query.
The photos index receives a CLIP text embedding of the same semantic query.

### 3.4 Pinecone Record Structure

**Listings index:**
```
vector_id   = mls_listing_number
values      = text-embedding-3-large(listing_description)   # 3072-dim dense
sparse      = BM25(listing_description)                     # sparse for hybrid
metadata    = {
    "mls_listing_number": "...",
    "listing_description": "...",
    "address": "...", "list_price": ..., "total_bedrooms": ...,
    "neighborhood_name": "...", "property_type": "...", ...
}
```

**Photos index:**
```
vector_id   = "{mls_listing_number}_{photo_filename}"
values      = CLIP.vision_model(image)   # 512-dim
metadata    = {
    "mls_listing_number": "...",
    "photo": "...",
    "main_photo": true/false
}
```

---

## 4. Hybrid Search Design

The retrieve step runs three signals and merges them via RRF:

```
User question: "3+ bedrooms under $2M with white kitchen cabinets and marble countertops"
    │
    ▼
[parse node — GPT-4o]
    │  structured filters  →  {bedrooms_min: 3, price_max: 2000000}
    │  semantic query      →  "white kitchen cabinets marble countertops"
    │
    ├──────────────────────────────────────────────────────────────┐
    │                                                              │
    ▼                                                             ▼
[CSV filter leg — pandas]                          [Pinecone legs — parallel]
listings.csv                                            │
→ apply hard filters                                    ├─ Listings index (dotproduct, hybrid)
→ return candidate listing_ids                          │   dense embed (text-embedding-3-large)
  (e.g. 2000 total → ~380 candidates)                  │   + BM25 sparse
    │                                                   │   scoped to candidate listing_ids
    │                                                   │   → top-k hits [listings-hybrid]
    │                                                   │
    │                                                   └─ Photos index (cosine, CLIP)
    │                                                       CLIP text embed of semantic query
    │                                                       → top matching photo vectors
    │                                                       → resolve to listing_ids
    │                                                       → top-k hits [photos-clip]
    │                                                              │
    └──────────────────────────────────┬───────────────────────────┘
                                       │
                                       ▼
                              [RRF merge]
                              Reciprocal Rank Fusion combines
                              [listings-hybrid] + [photos-clip] rankings
                              → unified listing ranking
                                       │
                                       ▼
                              [grade node] → [generate node]
```

**Why RRF over simple score blending?**
RRF is rank-based, so it's robust to score scale differences between cosine (photos) and
dotproduct (listings). A listing that appears in both signals gets a strong combined rank
even if neither individual score was the highest.

**Why `dotproduct` for listings?**
Pinecone requires `dotproduct` metric for hybrid (dense + sparse BM25) search. Cosine
would reject the sparse component.

**Why `cosine` for photos?**
CLIP embeddings are unit-normalized — cosine similarity is the standard metric and gives
meaningful cross-modal alignment between text queries and image vectors.

---

## 5. Component Breakdown

### 5.1 Streamlit UI (`app.py`)

| Mode | Audience | Key widgets |
|---|---|---|
| **Find your dream homes** | Home buyers | Chat input, listing cards with photo carousel (‹/› navigation, dot indicators), neighborhood badge |
| **Agent Dashboard** | Real estate agents | Comparison table, filter sidebar, NL search, export CSV |

`st.session_state` holds the LangGraph thread ID, message history, and per-card photo index
(`photo_idx_{mls}`) for independent carousel navigation across listing cards.

Photos are loaded at startup from `listing_photos.csv` into a dict keyed by MLS number,
main photo first.

### 5.2 LangGraph State Machine (`src/graph/`)

```
parse ──► retrieve ──► grade ──┬─► generate
                               └─► retrieve  (relaxed filters, max 1 retry)
```

**State schema** (`src/graph/state.py`):

```python
class SearchState(TypedDict):
    question: str
    filters: SearchFilters          # structured output from parse node
    candidate_ids: list[str]        # listing_ids from CSV filter leg
    documents: list[Document]       # merged hits from RRF (listings + photos)
    grade: str                      # "relevant" | "weak"
    answer: str
    listing_ids: list[str]          # cited in generated answer
    retry_count: int
    messages: list[BaseMessage]
```

**Nodes** (`src/graph/nodes.py`):

| Node | Model | What it does |
|---|---|---|
| `parse` | GPT-4o (structured output) | Extracts `SearchFilters` from user question |
| `retrieve` | pandas + Pinecone (listings hybrid + photos CLIP) | CSV filter → candidate IDs; hybrid + CLIP retrieval; RRF merge |
| `grade` | GPT-4o-mini | Scores each document: relevant / not relevant |
| `generate` | GPT-4o | Writes grounded answer citing MLS IDs found in retrieval |

### 5.3 Self-Query Parser (`src/retrieval/self_query.py`)

GPT-4o with structured output extracts `SearchFilters` from the user question.

```python
class SearchFilters(BaseModel):
    price_min: float | None
    price_max: float | None
    bedrooms_min: int | None
    bathrooms_min: float | None
    sqft_min: int | None
    neighborhood: str | None
    property_type: str | None
    semantic_query: str
```

SF neighborhood list seeded in the system prompt:
*Mission, Noe Valley, Pacific Heights, SOMA, Castro, Haight-Ashbury, Richmond, Sunset,
Marina, North Beach, Chinatown, Tenderloin, Potrero Hill, Bernal Heights, Glen Park,
Excelsior, Bayview, Inner Sunset, Outer Sunset, Cole Valley, Duboce Triangle,
Lower Haight, Hayes Valley, Dogpatch, Portola, West Portal, Forest Hill, Twin Peaks,
Russian Hill, Nob Hill, Financial District, Embarcadero.*

### 5.4 CSV Filter (`src/retrieval/csv_filter.py`)

```python
def filter_csv(df: pd.DataFrame, filters: SearchFilters) -> list[str]:
    mask = pd.Series(True, index=df.index)
    if filters.price_max:
        mask &= df["list_price"] <= filters.price_max
    if filters.bedrooms_min:
        mask &= df["total_bedrooms"] >= filters.bedrooms_min
    if filters.bathrooms_min:
        mask &= df["total_bathrooms"] >= filters.bathrooms_min
    if filters.sqft_min:
        mask &= df["total_square_feet"] >= filters.sqft_min
    if filters.neighborhood:
        mask &= df["neighborhood_name"].str.contains(filters.neighborhood, case=False)
    if filters.property_type:
        mask &= df["property_type"].str.contains(filters.property_type, case=False)
    return df.loc[mask, "mls_listing_number"].tolist()
```

The DataFrame is loaded once at app startup into `st.session_state`.

### 5.5 Hybrid + CLIP Retrieve (`src/retrieval/vectorstore.py`)

```python
def retrieve(semantic_query: str, candidate_ids: list[str], k: int = 8) -> list[Document]:
    # 1. Listings index — hybrid dense + BM25
    listings_hits = listings_index.query(
        vector=embed_text(semantic_query),         # dense: text-embedding-3-large
        sparse_vector=bm25_encode(semantic_query), # sparse: BM25
        filter={"mls_listing_number": {"$in": candidate_ids}},
        top_k=k, include_metadata=True,
    )

    # 2. Photos index — CLIP text-to-image
    photos_hits = photos_index.query(
        vector=clip_embed_text(semantic_query),    # CLIP 512-dim text embedding
        top_k=k, include_metadata=True,
    )
    # resolve photo hits → listing_ids (intersect with candidate_ids)

    # 3. RRF merge
    return rrf_merge(listings_hits, photos_hits, k=k)
```

Each query is logged to `logs/search.log` with signal attribution:
`[listings-hybrid]` for listings index hits, `[photos-clip]` for photo index hits.

### 5.6 CLIP Embeddings (`src/retrieval/clip_embeddings.py`)

```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path: Path) -> list[float]:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
    features = model.visual_projection(vision_out.pooler_output)
    return F.normalize(features, p=2, dim=-1).squeeze().tolist()

def embed_text(text: str) -> list[float]:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    text_out = model.text_model(input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"])
    features = model.text_projection(text_out.pooler_output)
    return F.normalize(features, p=2, dim=-1).squeeze().tolist()
```

Sub-models are used directly (`.vision_model()` + `.visual_projection()`) rather than
`get_image_features()` for cross-version compatibility with newer `transformers` releases.

### 5.7 Text Embeddings (`src/retrieval/embeddings.py`)

```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",  # 3072-dim
    openai_api_key=settings.openai_api_key,
)
```

At ingest: embed `listing_description` for every row.
At query time: embed the `semantic_query` string for the dense component of hybrid search.

### 5.8 LLM (`src/llm.py`)

```python
def get_llm(model: str = "gpt-4o", temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature,
                      openai_api_key=settings.openai_api_key)
```

`parse` and `generate` use `gpt-4o`. `grade` uses `gpt-4o-mini`.

---

## 6. Ingestion Pipeline

### 6.1 Listings Ingest

```
data/listings.csv
      │
      ▼
loader.py
      │  read CSV with pandas
      │  validate SF city scope
      │  validate listing_description is non-null
      │
      ▼
document_builder.py
      │  page_content = row["listing_description"]
      │  metadata     = {all structured columns} + {"mls_listing_number": ...}
      │
      ▼
OpenAIEmbeddings.embed_documents()     # dense 3072-dim
BM25Encoder.encode_documents()         # sparse BM25
      │
      ▼
indexer.py → Pinecone upsert (batches of 100)
      │  index: sf-real-estate-listings
      │  metric: dotproduct  (required for hybrid)
      │  dimension: 3072
      ▼
Pinecone index: sf-real-estate-listings
```

```bash
python scripts/ingest.py --source data/listings.csv [--recreate]
```

### 6.2 Photos Ingest

```
data/listing_photos.csv + data/photos/*.jpg
      │
      ▼
indexer.py (run_photos_ingest)
      │  for each row in listing_photos.csv:
      │    load image from data/photos/{photo}
      │    CLIP.embed_image() → 512-dim vector
      │    upsert to photos index
      │      vector_id = "{mls}_{filename}"
      │      metadata  = {mls_listing_number, photo, main_photo}
      │
      ▼
Pinecone index: sf-real-estate-photos
      │  metric: cosine
      │  dimension: 512
```

```bash
python scripts/ingest.py --photos-only [--recreate]
```

### 6.3 Photo Download & Assignment

```bash
python scripts/fetch_photos.py
```

Downloads Unsplash photos across three categories (exterior/property, luxury bathroom,
luxury kitchen) and assigns them to 2000 listings. Assignment rule: max 1 exterior style +
1 bathroom style + 1 kitchen style per listing — no two photos of different styles from the
same category are assigned to the same listing.

---

## 7. Prompt Templates

### parse node

```
System: You are a real estate search parser for San Francisco, CA listings.
        Extract structured search criteria from the user's question.
        SF neighborhoods include: Mission, Noe Valley, Pacific Heights, SOMA, [full list].
        Return ONLY the SearchFilters JSON — no other text.

User: {question}
Conversation history: {history_summary}
```

### grade node (per document)

```
System: You are a relevance grader for real estate listings.
        Given a buyer's question and a listing description, return JSON:
        {"relevant": true/false, "reason": "one sentence"}.

User question: {question}
Listing description: {document.page_content}
```

### generate node

```
System: You are a helpful real estate assistant for San Francisco home buyers and agents.
        Use ONLY the listing descriptions provided below. Do not invent or mention
        any listing not in the context. For each listing you reference, cite its
        MLS number like [MLS: {mls_listing_number}]. If no relevant listings are in
        the context, say clearly that no matching listings were found.

Listing context:
{formatted_documents}

User: {question}
```

---

## 8. Streamlit UI Design

### Find your dream homes

```
┌──────────────────────────────────────────────────────────┐
│  🌁 FogNest                             [New Search]     │
│  Dream nests in fog...                                   │
│  ○ Find your dream homes  ○ Agent Dashboard              │
├──────────────────────────────────────────────────────────┤
│  Find your home in San Francisco                         │
│                                                          │
│  Assistant: Hi! Describe your ideal SF home and I'll     │
│  find listings that match.                               │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 3+ bedrooms under $2M with white kitchen cabinets  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  3 listings found                                        │
│  ┌──────────────────┐ ┌──────────────────┐              │
│  │  [photo]  ‹ ●○○ ›│ │  [photo]  ‹ ○●○ ›│             │
│  │  $1,895,000      │ │  $1,750,000      │              │
│  │  123 Sanchez St  │ │  456 Liberty St  │              │
│  │  3bd / 2ba       │ │  4bd / 2.5ba     │              │
│  │  Noe Valley · SFH│ │  Castro · SFH    │              │
│  │  [description ▼] │ │  [description ▼] │              │
│  │  MLS# SF281614   │ │  MLS# SF624282   │              │
│  └──────────────────┘ └──────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

### Agent Dashboard

```
┌──────────────────────────────────────────────────────────┐
│  Agent Dashboard — San Francisco Listings                │
├───────────────┬──────────────────────────────────────────┤
│  Filters      │  Results                                 │
│               │                                          │
│  Price        │  MLS        Address      Beds Baths Price│
│  [$][max]     │  SF281614   123 Sanchez   3    2   $1.9M │
│               │  SF624282   456 Liberty   4    2.5 $1.75M│
│  Beds   [≥]   │                                          │
│  Baths  [≥]   │  [Export CSV]                            │
│  Type   [▾]   │                                          │
│  Neighbor[▾]  │  Natural language search:                │
│               │  ┌──────────────────────────────────┐   │
│  [Apply]      │  │ white kitchen cabinets marble...  │   │
│               │  └──────────────────────────────────┘   │
└───────────────┴──────────────────────────────────────────┘
```

---

## 9. Cost & Latency

| Step | Model | Approx cost/query |
|---|---|---|
| Parse (self-query) | gpt-4o | ~$0.002 |
| CSV filter | pandas (local) | $0 |
| Text embedding (semantic query) | text-embedding-3-large | ~$0.0001 |
| CLIP text embedding (photos query) | CLIP (local, CPU) | $0 |
| Grade (8 docs) | gpt-4o-mini | ~$0.001 |
| Generate | gpt-4o | ~$0.004 |
| **Total per query** | | **~$0.007** |

**Ingest cost:**
- `text-embedding-3-large`: $0.13 / 1M tokens. ~250 tokens/listing → ~$0.000033/listing. 2000 listings ≈ $0.07.
- CLIP photo embedding: free (runs locally on CPU). 6000 photos in ~minutes.

**Avg query latency (from eval):** ~18s — dominated by GPT-4o parse + generate calls.

---

## 10. Configuration (`src/config.py`)

```python
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

    clip_dimension: int = 512
    listings_csv_path: str = "data/listings.csv"
    photos_csv_path: str = "data/listing_photos.csv"
    photos_dir: str = "data/photos"

    retrieval_k: int = 8
    relevance_threshold: int = 3
    max_retries: int = 1

    class Config:
        env_file = ".env"
```

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| `listing_description` is the primary embedded field | Agents encode qualitative intent-matching detail here (architecture, light, character) that no structured field captures |
| `dotproduct` metric for listings index | Required by Pinecone for hybrid (dense + BM25 sparse) search — cosine rejects the sparse component |
| `cosine` metric for photos index | CLIP embeddings are unit-normalized; cosine is the standard metric for cross-modal text-image similarity |
| CLIP sub-models used directly | `model.vision_model()` + `model.visual_projection()` instead of `get_image_features()` — cross-version compatible with newer `transformers` releases |
| RRF to merge listing and photo signals | Rank-based fusion is robust to score scale differences between dotproduct (listings) and cosine (photos) — no manual weight tuning needed |
| 1 exterior + 1 bathroom + 1 kitchen photo per listing | Prevents listings from showing multiple conflicting styles; a property has one exterior look, one kitchen, one bathroom aesthetic |
| Per-card `photo_idx_{mls}` session state | Independent carousel navigation across listing cards — one card advancing doesn't reset another |
| CSV as structured authority, Pinecone as semantic index | Hard constraints (price, beds) are never violated by soft semantic scores; the CSV is the source of record for attributes |
| `gpt-4o-mini` for grading | Binary relevance classification; mini is 15x cheaper and accurate enough — confirmed by eval (4.27/5 groundedness) |
| DataFrame loaded once at app startup | Avoids re-reading CSV on every query; pandas in-memory filter on thousands of rows is sub-millisecond |
| Listing MLS number as Pinecone vector ID | Upsert is idempotent; updating a listing re-embeds only that row |
