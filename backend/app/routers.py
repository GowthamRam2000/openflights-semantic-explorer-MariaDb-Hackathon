from fastapi import APIRouter, Query
from .db import get_conn

router = APIRouter()

@router.get("/similar-airports")
def similar_airports(query_vec: str = Query(..., description="Vector as text: [x1,x2,...]"),
                     tz_prefix: str = "Asia/"):
    """
    filter by timezone prefix + vector order.
    Pass 'query_vec' as a string like "[0.1,0.2,...]".
    """
    sql = """
    SELECT a.iata, a.name, a.city, a.country,
           VEC_DISTANCE_COSINE(e.emb, VEC_FromText(%s)) AS dist
    FROM airports a
    JOIN airports_emb e USING (airport_id)
    WHERE a.tz LIKE CONCAT(%s, '%%')
    ORDER BY dist ASC
    LIMIT 20
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, (query_vec, tz_prefix))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]