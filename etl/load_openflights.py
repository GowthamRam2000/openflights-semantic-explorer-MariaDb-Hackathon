import os, pandas as pd,sys
from dotenv import load_dotenv
from backend.app.db import get_conn
from etl.build_texts import airport_text, airline_text, route_text
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
load_dotenv()
DATA_DIR = os.path.join("data","openflights","data")

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
    for c in ["airport_id","airline_id","src_id","dst_id","stops","altitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def upsert_df(df: pd.DataFrame, table: str, cols: list[str], pk: str|None):
    placeholders = ",".join(["%s"]*len(cols))
    collist = ",".join(cols)
    updates = ",".join([f"{c}=VALUES({c})" for c in cols if c != pk])
    sql = f"INSERT INTO {table} ({collist}) VALUES ({placeholders})"
    if pk:
        sql += f" ON DUPLICATE KEY UPDATE {updates}"
    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(sql, df[cols].where(pd.notnull(df), None).values.tolist())
        print(f"Upserted {cur.rowcount} rows into {table}")

def main():
    airports = load_table("airports")
    airlines = load_table("airlines")
    routes   = load_table("routes")

    # relational
    upsert_df(airports, "airports", airports.columns.tolist(), "airport_id")
    upsert_df(airlines, "airlines", airlines.columns.tolist(), "airline_id")

    # routes -> autoinc id
    with get_conn() as conn:
        cur = conn.cursor()
        placeholders = ",".join(["%s"]*len(routes.columns))
        cur.executemany(
            f"INSERT INTO routes ({','.join(routes.columns)}) VALUES ({placeholders})",
            routes.where(pd.notnull(routes), None).values.tolist()
        )
        print(f"Inserted {cur.rowcount} route rows")

    # optional debug dumps
    os.makedirs("tmp", exist_ok=True)
    airports.assign(desc_text=airports.apply(airport_text, axis=1))[["airport_id","desc_text"]].to_csv("tmp/airports_desc.csv", index=False)
    airlines.assign(desc_text=airlines.apply(airline_text, axis=1))[["airline_id","desc_text"]].to_csv("tmp/airlines_desc.csv", index=False)
    # routes desc built during embeddings

if __name__ == "__main__":
    main()