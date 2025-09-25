# backend/app/main.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Our connection helper (supports mariadb or pymysql based on env)
from .db import get_conn

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
@app.get("/similar-airports")
def similar_airports(
    query_vec: str = Query(..., description="Embedding as JSON-like text '[...]'"),
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

    base_sql = f"""
      SELECT a.airport_id, a.name, a.city, a.country, a.iata, a.icao, a.tz,
             VEC_DISTANCE_COSINE(e.emb, VEC_FromText(%s)) AS score
      FROM airports a
      JOIN airports_emb e ON e.airport_id = a.airport_id
    """

    where = []
    params: List[Any] = [query_vec]

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
            return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/similar-airports error: {e}")

# ---------- Similar Routes (Hybrid: vector + filters) ----------
@app.get("/similar-routes")
def similar_routes(
    query_vec: str = Query(..., description="Embedding as JSON-like text '[...]'"),
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
    params: List[Any] = [query_vec]  # for the subquery vec

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
            return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/similar-routes error: {e}")

# ---------- Similar Airlines ----------
@app.get("/similar-airlines")
def similar_airlines(
    query_vec: str = Query(..., description="Embedding as JSON-like text '[...]'"),
    country: Optional[str] = Query(None, description="Exact country match, optional"),
    k: int = Query(25, ge=1, le=200),
):
    """
    Vector similarity over airlines with optional country filter.
    """
    country = (country or "").strip()
    k = _as_int(k, 25, 1, 200)

    sql = f"""
      SELECT a.airline_id, a.name, a.country, a.iata, a.icao, a.active,
             VEC_DISTANCE_COSINE(e.emb, VEC_FromText(%s)) AS score
      FROM airlines a
      JOIN airlines_emb e ON e.airline_id = a.airline_id
      {"WHERE a.country = %s" if country else ""}
      ORDER BY score ASC
      LIMIT {k}
    """

    params: List[Any] = [query_vec] + ([country] if country else [])

    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = _fetch_dicts(cur)
            return rows
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
