"""
FogNest — Streamlit App
Run: streamlit run app.py
"""
import logging
import logging.handlers
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st


def _get_app_logger() -> logging.Logger:
    logger = logging.getLogger("app.dashboard")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_path = Path("logs/search.log")
    log_path.parent.mkdir(exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FogNest",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy imports (avoid loading heavy deps until env is confirmed)
# ---------------------------------------------------------------------------
def _check_env():
    try:
        from src.config import settings  # noqa: F401
        return True, ""
    except Exception as e:
        return False, str(e)

env_ok, env_err = _check_env()
if not env_ok:
    st.error(f"**Configuration error:** {env_err}\n\nCopy `.env.example` → `.env` and fill in your API keys.")
    st.stop()

from src.config import settings
from src.graph.build import build_graph
from src.retrieval.csv_filter import get_dataframe

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {role, content, listings}
if "graph" not in st.session_state:
    with st.spinner("Loading AI pipeline..."):
        st.session_state.graph = build_graph()
if "df" not in st.session_state:
    st.session_state.df = get_dataframe()
if "photos" not in st.session_state:
    photos_path = Path(settings.photos_csv_path)
    if photos_path.exists():
        photos_df = pd.read_csv(photos_path, dtype={"mls_listing_number": str})
        photos_df["is_main"] = photos_df["main_photo"].astype(str).str.lower() == "true"
        # Build dict: mls -> [main_photo, ...other photos] in order
        grouped: dict[str, list[str]] = {}
        for _, row in photos_df.iterrows():
            mls = row["mls_listing_number"]
            if mls not in grouped:
                grouped[mls] = []
            if row["is_main"]:
                grouped[mls].insert(0, row["photo"])
            else:
                grouped[mls].append(row["photo"])
        st.session_state.photos = grouped
    else:
        st.session_state.photos = {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PHOTOS_DIR = Path(settings.photos_dir)

def get_all_photos(mls: str) -> list[Path]:
    """Return all existing photo paths for a listing, main photo first."""
    filenames = st.session_state.photos.get(mls, [])
    return [PHOTOS_DIR / f for f in filenames if (PHOTOS_DIR / f).exists()]

def lookup_listings(mls_ids: list[str]) -> list[dict]:
    df = st.session_state.df
    rows = df[df["mls_listing_number"].isin(mls_ids)]
    return rows.to_dict(orient="records")

def _carousel_prev(idx_key: str, current: int) -> None:
    st.session_state[idx_key] = max(current - 1, 0)

def _carousel_next(idx_key: str, current: int, limit: int) -> None:
    st.session_state[idx_key] = min(current + 1, limit)

def render_listing_card(listing: dict):
    mls = str(listing.get("mls_listing_number", ""))
    photos = get_all_photos(mls)

    idx_key = f"photo_idx_{mls}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    idx = min(st.session_state[idx_key], max(len(photos) - 1, 0))

    with st.container(border=True):
        if photos:
            st.image(str(photos[idx]), use_container_width=True)
            if len(photos) > 1:
                prev_col, dots_col, next_col = st.columns([1, 5, 1])
                with prev_col:
                    st.button(
                        "‹", key=f"prev_{mls}",
                        disabled=idx == 0,
                        use_container_width=True,
                        on_click=_carousel_prev,
                        args=(idx_key, idx),
                    )
                with next_col:
                    st.button(
                        "›", key=f"next_{mls}",
                        disabled=idx == len(photos) - 1,
                        use_container_width=True,
                        on_click=_carousel_next,
                        args=(idx_key, idx, len(photos) - 1),
                    )
                with dots_col:
                    dots = "".join("●" if i == idx else "○" for i in range(len(photos)))
                    st.markdown(
                        f"<div style='text-align:center;font-size:11px;"
                        f"color:#888;letter-spacing:3px;padding-top:6px'>{dots}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("🏠")

        price = listing.get("list_price", 0)
        st.markdown(f"**${price:,.0f}**")
        st.caption(listing.get("address", ""))

        col1, col2, col3 = st.columns(3)
        col1.metric("Beds", listing.get("total_bedrooms", "—"))
        col2.metric("Baths", listing.get("total_bathrooms", "—"))
        col3.metric("Sqft", f"{int(listing.get('total_square_feet', 0)):,}")

        st.caption(
            f"🏘 {listing.get('neighborhood_name', '')}  •  "
            f"{listing.get('property_type', '')}  •  {listing.get('views', '')}"
        )

        desc = str(listing.get("listing_description", ""))
        with st.expander("Listing description"):
            st.write(desc)

        st.caption(f"MLS# {mls}")

def invoke_graph(question: str) -> dict:
    graph = st.session_state.graph
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = graph.invoke(
        {"question": question, "retry_count": 0},
        config=config,
    )
    return result

# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🌁 FogNest")
    st.caption("Dream nests in fog — An AI RAG powered San Francisco home finder app.")
    st.divider()
    mode = st.radio("Mode", ["Find your dream homes", "Agent Dashboard"], label_visibility="collapsed")
    st.divider()
    if st.button("New search / clear chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.markdown("---")
    st.caption("Powered by OpenAI · LangGraph · Pinecone")

# ===========================================================================
# BUYER CHAT MODE
# ===========================================================================
if mode == "Find your dream homes":
    st.header("Find your dream home in San Francisco")

    # Quick-filter neighborhood chips
    st.markdown(
        "**Popular neighborhoods:** "
        "`Mission` · `Noe Valley` · `Pacific Heights` · `Marina` · `Castro` · "
        "`Haight-Ashbury` · `Bernal Heights` · `Hayes Valley` · `SOMA` · `Russian Hill`"
    )
    st.divider()

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("listings"):
                cols = st.columns(min(len(msg["listings"]), 3))
                for col, listing in zip(cols, msg["listings"][:3]):
                    with col:
                        render_listing_card(listing)
                if len(msg["listings"]) > 3:
                    with st.expander(f"Show {len(msg['listings']) - 3} more listings"):
                        more_cols = st.columns(min(len(msg["listings"]) - 3, 3))
                        for col, listing in zip(more_cols, msg["listings"][3:]):
                            with col:
                                render_listing_card(listing)

    # Chat input
    if prompt := st.chat_input("Describe your ideal SF home…"):
        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Invoke graph
        with st.chat_message("assistant"):
            with st.spinner("Searching listings…"):
                try:
                    result = invoke_graph(prompt)
                    answer = result.get("answer", "Sorry, something went wrong.")
                    cited_ids = result.get("cited_listing_ids", [])
                    listings = lookup_listings(cited_ids) if cited_ids else []
                except Exception as e:
                    answer = f"Error: {e}"
                    listings = []

            st.markdown(answer)
            if listings:
                cols = st.columns(min(len(listings), 3))
                for col, listing in zip(cols, listings[:3]):
                    with col:
                        render_listing_card(listing)
                if len(listings) > 3:
                    with st.expander(f"Show {len(listings) - 3} more"):
                        more_cols = st.columns(min(len(listings) - 3, 3))
                        for col, listing in zip(more_cols, listings[3:]):
                            with col:
                                render_listing_card(listing)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "listings": listings,
        })

# ===========================================================================
# AGENT DASHBOARD MODE
# ===========================================================================
else:
    st.header("Agent Dashboard — San Francisco Listings")

    df = st.session_state.df

    # --- Filter sidebar ---
    with st.sidebar:
        st.subheader("Filters")

        price_min, price_max = st.slider(
            "Price range",
            min_value=500_000,
            max_value=3_000_000,
            value=(500_000, 3_000_000),
            step=50_000,
            format="$%d",
        )

        beds_min = st.selectbox("Min bedrooms", [0, 1, 2, 3, 4, 5], index=0)
        baths_min = st.selectbox("Min bathrooms", [0, 1, 1.5, 2, 2.5, 3], index=0)

        prop_types = ["All"] + sorted(df["property_type"].dropna().unique().tolist())
        selected_type = st.selectbox("Property type", prop_types)

        neighborhoods = ["All"] + sorted(df["neighborhood_name"].dropna().unique().tolist())
        selected_nbhd = st.selectbox("Neighborhood", neighborhoods)

        views_opts = ["All"] + sorted(df["views"].dropna().unique().tolist())
        selected_view = st.selectbox("View", views_opts)

        apply = st.button("Apply filters", type="primary")

    # --- Filtered results table ---
    filtered = df.copy()
    filtered = filtered[
        (filtered["list_price"] >= price_min) &
        (filtered["list_price"] <= price_max) &
        (filtered["total_bedrooms"] >= beds_min) &
        (filtered["total_bathrooms"] >= baths_min)
    ]
    if selected_type != "All":
        filtered = filtered[filtered["property_type"] == selected_type]
    if selected_nbhd != "All":
        filtered = filtered[filtered["neighborhood_name"] == selected_nbhd]
    if selected_view != "All":
        filtered = filtered[filtered["views"] == selected_view]

    if apply:
        _get_app_logger().info(
            "DASHBOARD FILTER | price=$%s–$%s | beds>=%s | baths>=%s"
            " | type=%r | neighborhood=%r | view=%r | results=%d",
            f"{price_min:,}", f"{price_max:,}",
            beds_min, baths_min,
            selected_type, selected_nbhd, selected_view,
            len(filtered),
        )

    st.metric("Listings matching filters", len(filtered))

    # --- NL search within filtered results ---
    st.subheader("Natural language search")
    agent_q = st.text_input("Describe what you're looking for (searches within filtered results)")

    if agent_q:
        _get_app_logger().info(
            "DASHBOARD NL SEARCH | query=%r | active_filters: type=%r neighborhood=%r view=%r"
            " | csv_candidates=%d",
            agent_q, selected_type, selected_nbhd, selected_view, len(filtered),
        )
        with st.spinner("Searching…"):
            try:
                result = invoke_graph(agent_q)
                answer = result.get("answer", "")
                cited_ids = result.get("cited_listing_ids", [])
                nl_listings = lookup_listings(cited_ids)
            except Exception as e:
                answer = f"Error: {e}"
                nl_listings = []
                _get_app_logger().error("DASHBOARD NL SEARCH ERROR | %s", e)

        st.markdown(answer)
        if nl_listings:
            nl_cols = st.columns(min(len(nl_listings), 3))
            for col, listing in zip(nl_cols, nl_listings[:3]):
                with col:
                    render_listing_card(listing)

    st.divider()

    # --- Results table ---
    display_cols = [
        "mls_listing_number", "address", "list_price",
        "total_bedrooms", "total_bathrooms", "total_square_feet",
        "property_type", "neighborhood_name", "views",
    ]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "mls_listing_number": "MLS#",
            "address": "Address",
            "list_price": "Price",
            "total_bedrooms": "Beds",
            "total_bathrooms": "Baths",
            "total_square_feet": "Sqft",
            "property_type": "Type",
            "neighborhood_name": "Neighborhood",
            "views": "Views",
        }).style.format({"Price": "${:,.0f}"}),
        use_container_width=True,
        height=450,
    )

    # Export
    csv_export = filtered[display_cols].to_csv(index=False)
    st.download_button(
        "Export filtered results (CSV)",
        data=csv_export,
        file_name="sf_listings_filtered.csv",
        mime="text/csv",
    )
