import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(
    page_title="OpenFlights Semantic Explorer ✈",
    page_icon="✈",
    layout="wide",
)
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
RAW_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001").strip()
EMB_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))
def normalize_model(name: str) -> str:
    name = (name or "").strip()
    aliases = {
        "gemini-embedding-001": "models/embedding-001",
        "embedding-001": "models/embedding-001",
        "text-embedding-004": "models/text-embedding-004",
    }
    if name.startswith("models/") or name.startswith("tunedModels/"):
        return name
    return aliases.get(name, f"models/{name}")

MODEL = normalize_model(RAW_MODEL)
@st.cache_resource(show_spinner=False)
def _genai():
    try:
        import google.generativeai as genai
    except Exception as e:
        st.error(f"Failed to import google.generativeai: {e}")
        raise
    if not GOOGLE_API_KEY:
        st.error("Missing GOOGLE_API_KEY in environment or .env")
        raise RuntimeError("No GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai

def vec_to_text(v: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

@st.cache_data(show_spinner=False)
def embed_cached(text: str, task_type: str, model: str, dim: int) -> List[float]:
    genai = _genai()
    tt = task_type if task_type in {"RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"} else "RETRIEVAL_QUERY"
    last_err = None
    for attempt in range(4):
        try:
            r = genai.embed_content(
                model=model,
                content=text,
                task_type=tt,
                output_dimensionality=dim,
            )
            emb = r["embedding"]
            if len(emb) != dim:
                raise RuntimeError(f"Model returned {len(emb)} dims; expected {dim}")
            return emb
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (2 ** attempt))
    raise RuntimeError(f"Embedding failed: {last_err}")

def embed(text: str, task_type: str = "RETRIEVAL_QUERY") -> List[float]:
    return embed_cached(text.strip(), task_type, MODEL, EMB_DIM)


def _needs_multilingual(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _filter_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None and v != ""}


def _run_vector_search(
    endpoint: str,
    prompt: str,
    base_params: Dict[str, Any],
    task_type: str = "RETRIEVAL_QUERY",
    timeout: int = 90,
) -> Tuple[List[Dict[str, Any]], bool]:

    params = dict(base_params)
    used_server_embedding = False

    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt must be non-empty")

    if GOOGLE_API_KEY:
        try:
            vec = embed(prompt, task_type=task_type)
            params["query_vec"] = vec_to_text(vec)
        except Exception as exc:  
            st.info(
                f"Local embedding failed ({exc}); falling back to API-side embeddings.",
                icon="ℹ️",
            )
            params.pop("query_vec", None)
            used_server_embedding = True
    else:
        used_server_embedding = True

    if used_server_embedding:
        params["query_text"] = prompt
        if _needs_multilingual(prompt):
            params["use_multilingual"] = "true"

    params = _filter_params(params)
    with st.spinner("Searching…"):
        resp = requests.get(
            f"{API_BASE}/{endpoint}", params=params, timeout=timeout
        )
        resp.raise_for_status()
    return resp.json(), used_server_embedding

st.markdown("### OpenFlights Semantic Explorer ✈")
st.caption(f"Embedding model: **{MODEL}** (from `{RAW_MODEL}`) · dim: **{EMB_DIM}**")
st.divider()
health_col1, health_col2 = st.columns([1, 8])
with health_col1:
    if st.button("Check API health", use_container_width=True):
        try:
            r = requests.get(f"{API_BASE}/health", timeout=10)
            ok = r.ok and r.json().get("ok")
            if ok:
                st.success("API healthy ")
            else:
                st.warning(f"API responded but not healthy: {r.text[:200]}")
        except Exception as e:
            st.error(f"Healthcheck failed: {e}")

with health_col2:
    st.info("Tip: If results look empty, try removing filters (e.g., Timezone) to widen search.", icon="💡")
tab_air, tab_routes, tab_airlines = st.tabs(["Airports", "Routes", "Airlines"])
with tab_air:
    st.subheader("Similar Airports")
    c1, c2 = st.columns([3, 2])

    with c1:
        aq = st.text_input(
            "Describe the airport you want",
            placeholder="e.g., major South India hub, international, long runway, gateway to US",
            key="air_q",
        )

    tz_col, _ = st.columns([2, 3])
    with tz_col:
        tz_continent = st.selectbox(
            "Timezone prefix (IANA)",
            ["(any)", "Africa/", "America/", "Antarctica/", "Asia/", "Atlantic/", "Australia/", "Europe/", "Indian/", "Pacific/"],
            index=3,  
        )
        exact_tz = st.text_input("Exact timezone (optional, e.g., Asia/Kolkata)", value="", key="exact_tz")

    k = st.slider("Top K", min_value=5, max_value=100, value=25, step=5, key="air_k")

    run = st.button("Search similar airports", type="primary", use_container_width=True)
    if run:
        if not aq.strip():
            st.warning("Please enter a short description."); st.stop()
        try:
            tz_prefix = None
            if exact_tz.strip():
                tz_prefix = exact_tz.strip()
            elif tz_continent != "(any)":
                tz_prefix = tz_continent

            params = {"tz_prefix": tz_prefix, "k": k}
            data, used_server_embedding = _run_vector_search(
                "similar-airports",
                aq,
                params,
                timeout=60,
            )
            if not data:
                st.warning("No matches. Try removing Timezone filter or broadening the description.")
            else:
                st.dataframe(data, use_container_width=True, height=480)
            if used_server_embedding and not GOOGLE_API_KEY:
                st.caption("Embedding handled server-side (no local GOOGLE_API_KEY).")
        except Exception as e:
            st.error(f"Airport search failed: {e}")

    with st.expander("Example queries"):
        st.markdown(
            "- “BLR-like international hub in Asia, long runway”\n"
            "- “Busy secondary airport near major metro, limited long-haul”\n"
            "- “Popular island leisure airport in Europe”"
        )

with tab_routes:
    st.subheader("Similar Routes (hybrid: vector + SQL filters)")
    rq = st.text_input(
        "Describe the route (used for semantic similarity)",
        placeholder="e.g., BLR to US West Coast, long-haul",
        key="route_q",
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        src = st.text_input("From (IATA)", placeholder="BLR").upper().strip() or None
    with c2:
        dst = st.text_input("To (IATA)", placeholder="SFO").upper().strip() or None
    with c3:
        avoid = st.text_input("Avoid airline (IATA/ICAO)", placeholder="AI").upper().strip() or None
    with c4:
        max_stops = st.number_input("Max stops", min_value=0, max_value=3, value=1, step=1)

    k_rt = st.slider("Top K", min_value=5, max_value=100, value=25, step=5, key="route_k")
    if st.button("Search routes", type="primary", use_container_width=True):
        if not rq.strip():
            st.warning("Please enter a short route description."); st.stop()
        try:
            params = {
                "src": src,
                "dst": dst,
                "avoid_airline": avoid,
                "stops_max": int(max_stops),
                "k": int(k_rt),
            }
            data, used_server_embedding = _run_vector_search(
                "similar-routes",
                rq,
                params,
                timeout=90,
            )
            if not data:
                st.warning("No route matches. Try removing filters or widening the description.")
            else:
                st.dataframe(data, use_container_width=True, height=480)
            if used_server_embedding and not GOOGLE_API_KEY:
                st.caption("Embedding handled server-side (no local GOOGLE_API_KEY).")
        except Exception as e:
            st.error(f"Route search failed: {e}")

    with st.expander("Example queries"):
        st.markdown(
            "- “BLR to SFO long-haul, avoid AI, ≤1 stop”\n"
            "- “SEA to Japan, prefer direct or 1 stop”\n"
            "- “SYD to Europe leisure-heavy routes”"
        )

with tab_airlines:
    st.subheader("Similar Airlines")
    alq = st.text_input(
        "Describe the airline",
        placeholder="e.g., premium Asian long-haul carrier",
        key="airline_q",
    )
    c1, c2 = st.columns([2, 1])
    with c1:
        country = st.text_input("Country filter (optional)", placeholder="India")
    with c2:
        k_al = st.slider("Top K", min_value=5, max_value=100, value=25, step=5, key="airline_k")

    if st.button("Search airlines", type="primary", use_container_width=True):
        if not alq.strip():
            st.warning("Please enter a short airline description."); st.stop()
        try:
            params = {
                "country": country.strip() or None,
                "k": int(k_al),
            }
            data, used_server_embedding = _run_vector_search(
                "similar-airlines",
                alq,
                params,
                timeout=60,
            )
            if not data:
                st.warning("No airline matches. Try clearing the country filter or adjusting your description.")
            else:
                st.dataframe(data, use_container_width=True, height=480)
            if used_server_embedding and not GOOGLE_API_KEY:
                st.caption("Embedding handled server-side (no local GOOGLE_API_KEY).")
        except Exception as e:
            st.error(f"Airline search failed: {e}")

    with st.expander("Example queries"):
        st.markdown(
            "- “Gulf premium long-haul carrier”\n"
            "- “Low-cost European leisure airline”\n"
            "- “Regional Indian carrier with turboprops”"
        )

st.divider()
st.caption(
    "Tip: Timezone accepts prefixes like **Asia/** or exact zones like **Asia/Kolkata**. "
    "Hybrid search = SQL filters (src/dst/stops/avoid) + vector similarity on descriptions."
)
