import os, time, random, argparse, math
from typing import List, Iterable, Tuple, Optional
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from backend.app.db import get_conn
from etl.build_texts import airport_text, airline_text, route_text

load_dotenv()
API_KEY=os.getenv("GOOGLE_API_KEY")
RAW_MODEL=os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004").strip()
DIM= int(os.getenv("GEMINI_EMBED_DIM", "768"))

def normalize_model(name: str) -> str:
    name = name.strip()
    aliases = {
        "gemini-embedding-001": "models/embedding-001",
        "embedding-001": "models/embedding-001",
        "text-embedding-004": "models/text-embedding-004",
    }
    if name.startswith("models/") or name.startswith("tunedModels/"):
        return name
    return aliases.get(name, f"models/{name}")
MODEL = normalize_model(RAW_MODEL)
if not API_KEY:
    raise SystemExit("Set GOOGLE_API_KEY in .env before running embeddings.")
genai.configure(api_key=API_KEY)
HAS_BATCH = hasattr(genai, "batch_embed_contents")
def backoff(attempt: int, base: float = 0.5, cap: float = 20.0) -> float:
    return min(cap, base * (2 ** attempt) + random.uniform(0, 0.25))

def embed_single(txt: str) -> List[float]:
    r = genai.embed_content(
        model=MODEL,
        content=txt,
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=DIM,
    )
    emb = r["embedding"]
    if len(emb) != DIM:
        raise RuntimeError(f"Model returned {len(emb)} dims; DB expects {DIM}.")
    return emb

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Use batch API when available, else loop single."""
    if not texts:
        return []
    if HAS_BATCH:
        # robust retry for whole batch
        for attempt in range(8):
            try:
                reqs = [{"content": t,
                         "task_type": "RETRIEVAL_DOCUMENT",
                         "output_dimensionality": DIM} for t in texts]
                resp = genai.batch_embed_contents(model=MODEL, requests=reqs)
                embs = []
                for e in resp.get("embeddings", []):
                    v = e.get("values") or e.get("embedding") or e
                    if len(v) != DIM:
                        raise RuntimeError(f"Batch item has {len(v)} dims; DB expects {DIM}.")
                    embs.append(v)
                if len(embs) != len(texts):
                    raise RuntimeError(f"Batch size mismatch: sent {len(texts)}, got {len(embs)}")
                return embs
            except Exception as e:
                if attempt == 7:
                    return [embed_with_retry(t) for t in texts]
                sleep_s = backoff(attempt)
                print(f"[batch retry {attempt+1}/8] {type(e).__name__}: {e} -> sleeping {sleep_s:.2f}s")
                time.sleep(sleep_s)
    return [embed_with_retry(t) for t in texts]

def embed_with_retry(txt: str) -> List[float]:
    for attempt in range(8):
        try:
            return embed_single(txt)
        except Exception as e:
            if attempt == 7:
                raise
            sleep_s = backoff(attempt)
            print(f"[single retry {attempt+1}/8] {type(e).__name__}: {e} -> sleeping {sleep_s:.2f}s")
            time.sleep(sleep_s)

def to_vec_text(v: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

def chunked(seq: List, size: int) -> Iterable[List]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]
def stream_airports(missing_only: bool = True,
                    tz_prefix: Optional[str] = None,
                    limit: Optional[int] = None,
                    page: int = 1000) -> Iterable[pd.DataFrame]:
    total = limit if limit else math.inf
    fetched = 0
    params = []
    where = []
    if missing_only:
        where.append("e.airport_id IS NULL")
    if tz_prefix:
        where.append("a.tz LIKE %s")
        params.append(f"{tz_prefix}%")
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    while fetched < total:
        take = page if limit is None else min(page, total - fetched)
        sql = f"""
          SELECT a.* FROM airports a
          LEFT JOIN airports_emb e ON e.airport_id=a.airport_id
          {where_clause}
          LIMIT {int(take)}
        """
        df = pd.read_sql(sql, get_conn(), params=params)
        if df.empty:
            break
        fetched += len(df)
        yield df

def stream_airlines(missing_only: bool = True,
                    limit: Optional[int] = None,
                    page: int = 1000) -> Iterable[pd.DataFrame]:
    total = limit if limit else math.inf
    fetched = 0
    while fetched < total:
        take = page if limit is None else min(page, total - fetched)
        sql = f"""
          SELECT a.* FROM airlines a
          LEFT JOIN airlines_emb e ON e.airline_id=a.airline_id
          {"WHERE e.airline_id IS NULL" if missing_only else ""}
          LIMIT {int(take)}
        """
        df = pd.read_sql(sql, get_conn())
        if df.empty:
            break
        fetched += len(df)
        yield df

def stream_routes(missing_only: bool = True,
                  limit: Optional[int] = None,
                  page: int = 2000) -> Iterable[pd.DataFrame]:
    total = limit if limit else math.inf
    fetched = 0
    while fetched < total:
        take = page if limit is None else min(page, total - fetched)
        sql = f"""
          SELECT r.* FROM routes r
          LEFT JOIN routes_emb e ON e.route_id=r.id
          {"WHERE e.route_id IS NULL" if missing_only else ""}
          LIMIT {int(take)}
        """
        df = pd.read_sql(sql, get_conn())
        if df.empty:
            break
        fetched += len(df)
        yield df

def write_airports(limit: Optional[int], tz_prefix: Optional[str],
                   batch_size: int, page: int):
    total_written = 0
    for df in stream_airports(missing_only=True, tz_prefix=tz_prefix, limit=limit, page=page):
        df = df.copy()
        df["desc_text"] = df.apply(airport_text, axis=1)
        descs = df["desc_text"].tolist()
        # embed in batches
        with get_conn() as conn:
            cur = conn.cursor()
            done = 0
            for texts in chunked(descs, batch_size):
                embs = embed_batch(texts)
                rows = []
                base = done
                for j, emb in enumerate(embs):
                    row = df.iloc[base + j]
                    rows.append((int(row["airport_id"]), row["desc_text"], to_vec_text(emb)))
                cur.executemany(
                    "INSERT INTO airports_emb (airport_id, desc_text, emb) VALUES (%s,%s,VEC_FromText(%s)) "
                    "ON DUPLICATE KEY UPDATE desc_text=VALUES(desc_text), emb=VALUES(emb)",
                    rows
                )
                done += len(texts)
                total_written += len(texts)
                if total_written % 500 == 0:
                    print(f"airports: {total_written} embedded")
    print(f"airports: total embedded this run = {total_written}")

def write_airlines(limit: Optional[int], batch_size: int, page: int):
    total_written = 0
    for df in stream_airlines(missing_only=True, limit=limit, page=page):
        df = df.copy()
        df["desc_text"] = df.apply(airline_text, axis=1)
        descs = df["desc_text"].tolist()
        with get_conn() as conn:
            cur = conn.cursor()
            done = 0
            for texts in chunked(descs, batch_size):
                embs = embed_batch(texts)
                rows = []
                base = done
                for j, emb in enumerate(embs):
                    row = df.iloc[base + j]
                    rows.append((int(row["airline_id"]), row["desc_text"], to_vec_text(emb)))
                cur.executemany(
                    "INSERT INTO airlines_emb (airline_id, desc_text, emb) VALUES (%s,%s,VEC_FromText(%s)) "
                    "ON DUPLICATE KEY UPDATE desc_text=VALUES(desc_text), emb=VALUES(emb)",
                    rows
                )
                done += len(texts)
                total_written += len(texts)
                if total_written % 500 == 0:
                    print(f"airlines: {total_written} embedded")
    print(f"airlines: total embedded this run = {total_written}")

def write_routes(limit: Optional[int], batch_size: int, page: int):
    total_written = 0
    for df in stream_routes(missing_only=True, limit=limit, page=page):
        df = df.copy()
        df["desc_text"] = df.apply(route_text, axis=1)
        descs = df["desc_text"].tolist()
        with get_conn() as conn:
            cur = conn.cursor()
            done = 0
            for texts in chunked(descs, batch_size):
                embs = embed_batch(texts)
                rows = []
                base = done
                for j, emb in enumerate(embs):
                    row = df.iloc[base + j]
                    rows.append((int(row["id"]), row["desc_text"], to_vec_text(emb)))
                cur.executemany(
                    "INSERT INTO routes_emb (route_id, desc_text, emb) VALUES (%s,%s,VEC_FromText(%s)) "
                    "ON DUPLICATE KEY UPDATE desc_text=VALUES(desc_text), emb=VALUES(emb)",
                    rows
                )
                done += len(texts)
                total_written += len(texts)
                if total_written % 1000 == 0:
                    print(f"routes: {total_written} embedded")
    print(f"routes: total embedded this run = {total_written}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["airports","airlines","routes","all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="Max rows per table for this run")
    p.add_argument("--tz", type=str, default=None, help="Airports only, e.g. 'Asia/'")
    p.add_argument("--batch", type=int, default=128, help="Embedding batch size (50-256 good)")
    p.add_argument("--page", type=int, default=4000, help="DB fetch page size")
    args = p.parse_args()

    # clamp batch to reasonable bounds
    batch = max(16, min(args.batch, 512))
    page  = max(batch, args.page)

    if args.only in ("airports","all"):
        write_airports(limit=args.limit, tz_prefix=args.tz, batch_size=batch, page=page)
    if args.only in ("airlines","all"):
        write_airlines(limit=args.limit, batch_size=batch, page=page)
    if args.only in ("routes","all"):
        write_routes(limit=args.limit, batch_size=batch, page=page)
