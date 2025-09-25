import pandas as pd

def airport_text(row: pd.Series) -> str:
    code = row.get("iata") or row.get("icao") or ""
    flags = []
    if isinstance(row.get("name"), str) and "International" in row["name"]:
        flags.append("international")
    if isinstance(row.get("type"), str) and row["type"]:
        flags.append(row["type"])
    tz = row.get("tz") or row.get("timezone") or ""
    bits = [code, row.get("name",""), row.get("city",""), row.get("country",""), *flags, tz]
    return " • ".join([str(b).strip() for b in bits if b and str(b).strip()])

def airline_text(row: pd.Series) -> str:
    code = row.get("iata") or row.get("icao") or ""
    bits = [code, row.get("name",""), row.get("callsign",""), row.get("country",""), f"active={row.get('active','')}"]
    return " • ".join([str(b).strip() for b in bits if b and str(b).strip()])

def route_text(row: pd.Series) -> str:
    legs = f"{row.get('src','')} → {row.get('dst','')}"
    stops = int(row.get("stops") or 0)
    stop_txt = "nonstop" if stops==0 else f"{stops} stops"
    airline = row.get("airline","")
    return f"{legs} • {stop_txt} • airline={airline}"