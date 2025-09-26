# backend/app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from .db import get_conn
from .embedding import embed_text, sanitize_vector_string, vector_to_text

app = FastAPI(title="OpenFlights Semantic Explorer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def _fetch_dicts(cur) -> List[Dict[str, Any]]:
    """Fetch rows from a DB-API cursor into list[dict]."""
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]

def _as_int(n: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(n)
    except Exception:
        n = default
    return max(lo, min(hi, n))


class AirportResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    airport_id: int
    name: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    iata: Optional[str] = None
    icao: Optional[str] = None
    tz: Optional[str] = None
    score: float = Field(..., description="Cosine distance (lower is more similar)")


class RouteResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    airline: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    stops: Optional[int] = None
    score: float = Field(..., description="Cosine distance (lower is more similar)")


class AirlineResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    airline_id: int
    name: Optional[str] = None
    country: Optional[str] = None
    iata: Optional[str] = None
    icao: Optional[str] = None
    active: Optional[str] = None
    score: float = Field(..., description="Cosine distance (lower is more similar)")


def _resolve_query_vector(
    *,
    query_vec: Optional[str],
    query_text: Optional[str],
    use_multilingual: bool,
    task_type: str,
) -> str:
    if query_vec and query_vec.strip():
        try:
            return sanitize_vector_string(query_vec)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    if query_text and query_text.strip():
        try:
            vector = embed_text(
                query_text,
                task_type=task_type,
                force_multilingual=use_multilingual,
            )
            return vector_to_text(vector)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    raise HTTPException(
        status_code=422,
        detail="Provide either query_vec (JSON array string) or query_text",
    )
@app.get("/health")
def health():
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchall()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
@app.get("/similar-airports", response_model=List[AirportResult])
def similar_airports(
    query_vec: Optional[str] = Query(
        None, description="Embedding as JSON-like text '[...]'"
    ),
    query_text: Optional[str] = Query(
        None,
        description="Free-form description to embed server-side if query_vec is omitted",
    ),
    use_multilingual: bool = Query(
        False,
        description="Force the multilingual embedding model when query_text is provided",
    ),
    tz_prefix: Optional[str] = Query(
        None,
        description="IANA timezone prefix or exact (e.g., 'Asia/' or 'Asia/Kolkata'). If provided, we use LIKE prefix%",
    ),
    k: int = Query(25, ge=1, le=200),
):
    """
    Vector similarity over airports with optional timezone narrowing.
    Uses the vector index first, then applies an optional tz filter.
    """
    k = _as_int(k, 25, 1, 200)
    vector_text = _resolve_query_vector(
        query_vec=query_vec,
        query_text=query_text,
        use_multilingual=use_multilingual,
        task_type="RETRIEVAL_QUERY",
    )

    base_sql = f"""
      SELECT a.airport_id, a.name, a.city, a.country, a.iata, a.icao, a.tz,
             VEC_DISTANCE_COSINE(e.emb, VEC_FromText(%s)) AS score
      FROM airports a
      JOIN airports_emb e ON e.airport_id = a.airport_id
    """

    where = []
    params: List[Any] = [vector_text]

    if tz_prefix and tz_prefix.strip():
        like_val = tz_prefix.strip()
        if not like_val.endswith("%"):
            like_val = like_val + "%"
        where.append("a.tz LIKE %s")
        params.append(like_val)

    sql = base_sql
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY score ASC LIMIT {k}"

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = _fetch_dicts(cur)
            return [AirportResult(**row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/similar-airports error: {e}")

# ---------- Similar Routes (Hybrid: vector + filters) ----------
@app.get("/similar-routes", response_model=List[RouteResult])
def similar_routes(
    query_vec: Optional[str] = Query(
        None, description="Embedding as JSON-like text '[...]'"
    ),
    query_text: Optional[str] = Query(
        None,
        description="Route description to embed server-side when query_vec is omitted",
    ),
    use_multilingual: bool = Query(
        False,
        description="Force multilingual embedding for query_text",
    ),
    src: Optional[str] = Query(None, description="IATA source (e.g., BLR)"),
    dst: Optional[str] = Query(None, description="IATA destination (e.g., SFO)"),
    avoid_airline: Optional[str] = Query(None, description="IATA/ICAO code to exclude (e.g., AI)"),
    stops_max: Optional[int] = Query(None, ge=0, le=3, description="Max stops to allow"),
    k: int = Query(25, ge=1, le=200),
):
    """
    Two-step pattern:
      1) Use vector index to pull top-N candidates (fast).
      2) Apply SQL filters on the candidate set, then return best K.
    """
    src = (src or "").strip().upper() or None
    dst = (dst or "").strip().upper() or None
    avoid_airline = (avoid_airline or "").strip().upper() or None
    k = _as_int(k, 25, 1, 200)
    vector_text = _resolve_query_vector(
        query_vec=query_vec,
        query_text=query_text,
        use_multilingual=use_multilingual,
        task_type="RETRIEVAL_QUERY",
    )
    topN = 2000  # candidate pool size before filtering

    sub_sql = f"""
      SELECT r.id, r.airline, r.src, r.dst, r.stops,
             VEC_DISTANCE_COSINE(re.emb, VEC_FromText(%s)) AS score
      FROM routes r
      JOIN routes_emb re ON re.route_id = r.id
      ORDER BY score ASC
      LIMIT {topN}
    """

    where = []
    params: List[Any] = [vector_text]  # for the subquery vec

    if src:
        where.append("t.src = %s")
        params.append(src)
    if dst:
        where.append("t.dst = %s")
        params.append(dst)
    if stops_max is not None:
        where.append("t.stops <= %s")
        params.append(int(stops_max))
    if avoid_airline:
        where.append("t.airline <> %s")
        params.append(avoid_airline)

    sql = "SELECT * FROM (" + sub_sql + ") t"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY t.score ASC LIMIT {k}"

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = _fetch_dicts(cur)
            return [RouteResult(**row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/similar-routes error: {e}")
@app.get("/similar-airlines", response_model=List[AirlineResult])
def similar_airlines(
    query_vec: Optional[str] = Query(
        None, description="Embedding as JSON-like text '[...]'"
    ),
    query_text: Optional[str] = Query(
        None,
        description="Airline description to embed server-side if query_vec is omitted",
    ),
    use_multilingual: bool = Query(
        False,
        description="Force multilingual embedding for query_text",
    ),
    country: Optional[str] = Query(None, description="Exact country match, optional"),
    k: int = Query(25, ge=1, le=200),
):
    """
    Vector similarity over airlines with optional country filter.
    """
    country = (country or "").strip()
    k = _as_int(k, 25, 1, 200)
    vector_text = _resolve_query_vector(
        query_vec=query_vec,
        query_text=query_text,
        use_multilingual=use_multilingual,
        task_type="RETRIEVAL_QUERY",
    )

    sql = f"""
      SELECT a.airline_id, a.name, a.country, a.iata, a.icao, a.active,
             VEC_DISTANCE_COSINE(e.emb, VEC_FromText(%s)) AS score
      FROM airlines a
      JOIN airlines_emb e ON e.airline_id = a.airline_id
      {"WHERE a.country = %s" if country else ""}
      ORDER BY score ASC
      LIMIT {k}
    """

    params: List[Any] = [vector_text] + ([country] if country else [])

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = _fetch_dicts(cur)
            return [AirlineResult(**row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/similar-airlines error: {e}")

# ---------- Optional: root ----------
@app.get("/")
def root():
    return {
        "name": "OpenFlights Semantic Explorer API",
        "endpoints": ["/health", "/similar-airports", "/similar-routes", "/similar-airlines"],
        "db_driver": os.getenv("DB_DRIVER", "auto"),
    }
