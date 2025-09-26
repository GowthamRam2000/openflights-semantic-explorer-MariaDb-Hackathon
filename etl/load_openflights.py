import os
import sys
import pandas as pd
from dotenv import load_dotenv
from backend.app.db import get_conn
from etl.build_texts import airport_text, airline_text, route_text

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

DATA_DIR = os.path.join("data", "openflights", "data")

FILES = {
    "airports": ("airports.dat", [
        "airport_id","name","city","country","iata","icao","latitude","longitude",
        "altitude","timezone","dst","tz","type","source"
    ]),
    "airlines": ("airlines.dat", [
        "airline_id","name","alias","iata","icao","callsign","country","active"
    ]),
    "routes": ("routes.dat", [
        "airline","airline_id","src","src_id","dst","dst_id","codeshare","stops","equipment"
    ]),
}

def load_table(table: str):
    file, cols = FILES[table]
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path, header=None, names=cols, na_values="\\N")
    if "timezone" in df.columns:
        df["timezone"] = df["timezone"].astype(str).str.slice(0, 255)
    if "tz" in df.columns:
        df["tz"] = df["tz"].astype(str).str.slice(0, 255)
    for c in ["airport_id", "airline_id", "src_id", "dst_id", "stops", "altitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def upsert_df(df: pd.DataFrame, table: str, cols: list[str], pk: str | None):
    placeholders = ",".join(["%s"] * len(cols))
    collist = ",".join(cols)
    updates = ",".join([f"{c}=VALUES({c})" for c in cols if c != pk])
    sql = f"INSERT INTO {table} ({collist}) VALUES ({placeholders})"
    if pk:
        sql += f" ON DUPLICATE KEY UPDATE {updates}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(sql, df[cols].where(pd.notnull(df), None).values.tolist())
        print(f"Upserted {cur.rowcount} rows into {table}")

def ensure_routes_unique_key():
    """Ensure the routes table has the generated route_key and no duplicates."""
    db_name = os.getenv("DB_NAME", "openflights")
    alter_sql = """
      ALTER TABLE routes
        ADD COLUMN route_key VARCHAR(191)
          GENERATED ALWAYS AS (
            CONCAT_WS(
              '|',
              UPPER(COALESCE(airline, '')),
              UPPER(COALESCE(src, '')),
              UPPER(COALESCE(dst, '')),
              COALESCE(codeshare, ''),
              COALESCE(CAST(stops AS CHAR), ''),
              COALESCE(equipment, '')
            )
          ) STORED,
        ADD UNIQUE KEY uniq_route_key (route_key)
    """

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.COLUMNS
            WHERE table_schema = %s
              AND table_name = 'routes'
              AND column_name = 'route_key'
            """,
            (db_name,),
        )
        has_column = cur.fetchone()[0] > 0
        if has_column:
            return

        print("Preparing routes table for idempotent loads (dedup + route_key)...")
        cur.execute(
            """
            DELETE r1 FROM routes r1
            JOIN routes r2
              ON r1.id > r2.id
             AND COALESCE(r1.airline, '') = COALESCE(r2.airline, '')
             AND COALESCE(r1.src, '') = COALESCE(r2.src, '')
             AND COALESCE(r1.dst, '') = COALESCE(r2.dst, '')
             AND COALESCE(r1.codeshare, '') = COALESCE(r2.codeshare, '')
             AND COALESCE(r1.stops, -1) = COALESCE(r2.stops, -1)
             AND COALESCE(r1.equipment, '') = COALESCE(r2.equipment, '')
            """
        )
        conn.commit()
        cur.execute(alter_sql)
        conn.commit()
        print("route_key column and uniqueness constraint added to routes table")

def build_route_dedup_key(df: pd.DataFrame) -> pd.Series:
    """Create a stable key mirroring the generated route_key expression."""
    stops_as_text = df["stops"].apply(lambda x: "" if pd.isna(x) else str(int(x)))
    return (
        df["airline"].fillna("").str.upper()
        + "|"
        + df["src"].fillna("").str.upper()
        + "|"
        + df["dst"].fillna("").str.upper()
        + "|"
        + df["codeshare"].fillna("")
        + "|"
        + stops_as_text
        + "|"
        + df["equipment"].fillna("")
    )

def main():
    airports = load_table("airports")
    airlines = load_table("airlines")
    routes = load_table("routes")

    # relational
    upsert_df(airports, "airports", airports.columns.tolist(), "airport_id")
    upsert_df(airlines, "airlines", airlines.columns.tolist(), "airline_id")

    # routes -> idempotent upsert preserving auto ids
    ensure_routes_unique_key()

    dedup_keys = build_route_dedup_key(routes)
    before = len(routes)
    routes = routes.loc[~dedup_keys.duplicated()].copy()
    after = len(routes)
    if after != before:
        print(f"Removed {before - after} duplicate route rows before upsert")

    route_cols = routes.columns.tolist()
    placeholders = ",".join(["%s"] * len(route_cols))
    update_cols = [c for c in route_cols if c not in ("id",)]
    updates = ",".join([f"{c}=VALUES({c})" for c in update_cols])

    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            (
                f"INSERT INTO routes ({','.join(route_cols)}) VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {updates}"
            ),
            routes[route_cols].where(pd.notnull(routes), None).values.tolist(),
        )
        print(f"Upserted {cur.rowcount} route rows")

    # optional debug dumps
    os.makedirs("tmp", exist_ok=True)
    airports.assign(desc_text=airports.apply(airport_text, axis=1))[["airport_id","desc_text"]].to_csv("tmp/airports_desc.csv", index=False)
    airlines.assign(desc_text=airlines.apply(airline_text, axis=1))[["airline_id","desc_text"]].to_csv("tmp/airlines_desc.csv", index=False)
    # routes desc built during embeddings

if __name__ == "__main__":
    main()
