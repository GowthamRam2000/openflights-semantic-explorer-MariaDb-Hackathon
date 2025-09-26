# OpenFlights Semantic Explorer

This project pairs the public OpenFlights dataset with modern vector search to build an interactive explorer for airports, airlines, and routes. A FastAPI backend exposes vector-powered endpoints backed by MariaDB 11.8’s native `VECTOR` type, while a Streamlit frontend lets you prompt for “similar” entities using natural language. ETL utilities load the relational data, generate descriptive texts, and upload Gemini embeddings that drive the similarity lookups.

---

## Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Running MariaDB](#running-mariadb)
- [Bootstrap the Database](#bootstrap-the-database)
- [Generate Embeddings](#generate-embeddings)
- [Run the API](#run-the-api)
- [Run the Streamlit App](#run-the-streamlit-app)
- [API Reference](#api-reference)
- [Cross-Platform Notes](#cross-platform-notes)
- [Verification on Apple M2 Mac](#verification-on-apple-m2-mac)
- [Troubleshooting](#troubleshooting)

---

## Architecture

- **MariaDB 11.8** stores the OpenFlights relational tables and three vector tables (`airports_emb`, `airlines_emb`, `routes_emb`). We rely on the `VECTOR` datatype and `VEC_DISTANCE_COSINE()` function for similarity search.
- **ETL (`etl/`)** scripts ingest the raw `.dat` files, normalize fields, and write them into MariaDB. They also generate descriptive text for each entity and call Google Gemini embeddings to populate the vector tables.
- **FastAPI backend (`backend/app/`)** exposes routes for health checks and similarity search. SQL queries compute cosine distance inside MariaDB, returning already filtered results.
- **Streamlit frontend (`frontend/streamlit_app.py`)** provides a lightweight dashboard that embeds user prompts, calls the API, and displays ranked entities.

---

## Project Structure

```
backend/          FastAPI application (DB connector, routers, main app)
data/openflights/ Raw OpenFlights CSV-equivalent data and helper docs
docker-compose.yml MariaDB 11.8 container with schema autoload mounts
etl/              Data loading and embedding generation utilities
frontend/         Streamlit UI for interactive exploration
sql/              Schema and vector table definitions for MariaDB
```

---

## Prerequisites

- **Python 3.10 or newer** (tested with 3.12)
- **pip** for dependency installation
- **Docker 24+ and Docker Compose v2** (or compatible) for the bundled MariaDB service
- **MariaDB Connector/C headers** (macOS: `brew install mariadb-connector-c`) when installing the `mariadb` Python package locally
- **Google Gemini API key** (required for embedding generation and the Streamlit prompt embedding helper)

Optional:
- `pymysql` only if you set `DB_DRIVER=pymysql` (MariaDB Connector/Python is the default and recommended)

---

## Environment Configuration

Create a `.env` file in the project root (or update the existing one) with at least the following keys:

```
DB_DRIVER=mariadb
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=yourchoice
DB_PASSWORD=yourchoice
DB_NAME=openflights

# Embedding configuration (Gemini)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_EMBED_MODEL=models/text-embedding-004
GEMINI_EMBED_DIM=768
GEMINI_MULTI_EMBED_MODEL=models/text-embedding-004  # optional multilingual fallback

# Frontend → Backend base URL
API_BASE=http://127.0.0.1:8000
```

The backend, ETL, and Streamlit load `.env` automatically via `python-dotenv`. Set `GEMINI_MULTI_EMBED_MODEL` if you want server-side embedding to choose a multilingual-capable model automatically when non-Latin input is detected.

---

## Running MariaDB

We ship a ready-to-use Docker Compose configuration that:

- Starts MariaDB 11.8 (the LTS build with vector support)
- Creates the `openflights` schema, user, and grants
- Mounts `sql/schema.sql` and `sql/vector_schema.sql` into `/docker-entrypoint-initdb.d` so tables are created on first boot

Start the database:

```bash
docker compose up -d mariadb
```

Check that the container is healthy:

```bash
docker compose logs -f mariadb
```

The database is exposed on `127.0.0.1:3306` with credentials `ofx/ofxpw` (root password is `rootpw123`).

If you prefer a self-managed MariaDB instance, ensure it runs 11.4+ with the Vector plugin enabled and apply the schema files manually.

---

## Bootstrap the Database

1. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install Python dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Load OpenFlights relational data**:

   ```bash
   python -m etl.load_openflights
   ```

   This upserts airports and airlines, then idempotently merges routes using a generated `route_key` so rerunning the loader will not create duplicates. On a fresh database the command finishes in under a minute.

The loader also writes descriptive CSV dumps into `tmp/` to help inspect the generated text.

---

## Generate Embeddings

Embeddings are required for similarity search. They are stored as `VECTOR(768)` in MariaDB.

```bash
python -m etl.embeddings --only airports --limit 5000  # dry-run example
python -m etl.embeddings --only airlines
python -m etl.embeddings --only routes
```

Key details:

- Requires `GOOGLE_API_KEY` to be set in `.env`
- Uses batched calls to Gemini’s `text-embedding-004` (configurable via env)
- Respects `ON DUPLICATE KEY UPDATE` to keep embeddings in sync
- Supports filters: `--tz Asia/` to embed only a subset of airports, `--limit` to throttle row counts during testing

The script prints progress counters every few hundred rows and reuses the precomputed descriptive strings while batching. Rerun as needed; missing embeddings are detected via left joins.

---

## Run Everything with Docker Compose

We now ship Dockerfiles for the backend (FastAPI + Uvicorn) and frontend (Streamlit). Bring up the full stack with:

```bash
docker compose up --build
```

Ensure your `.env` (or shell environment) includes the Gemini settings—at minimum `GOOGLE_API_KEY`, `GEMINI_EMBED_MODEL`, and optionally `GEMINI_MULTI_EMBED_MODEL`—so the backend can embed user prompts.

After the containers are running:

- Bootstrap the data:

  ```bash
  docker compose run --rm backend python -m etl.load_openflights
  docker compose run --rm backend python -m etl.embeddings --only all
  ```

- FastAPI docs are served from `http://127.0.0.1:8000/docs`
- Streamlit UI lives at `http://127.0.0.1:8501`

When Streamlit lacks a local Gemini key it will automatically send `query_text` to the backend, which embeds the prompt as long as the backend has `GOOGLE_API_KEY` configured.

---

## Run the API

Launch the FastAPI service with Uvicorn:

```bash
uvicorn backend.app.main:app --reload
```

The API exposes:

- Interactive docs: `http://127.0.0.1:8000/docs`
- Health probe: `GET /health`
- Vector endpoints described in [API Reference](#api-reference)

Environment variables control connection details (see `.env`). The backend defaults to MariaDB Connector/Python and opens short-lived connections for each request. Each similarity endpoint now accepts either a raw `query_vec` (JSON array string) **or** a `query_text`, which the API embeds server-side with optional multilingual fallback.

---

## Run the Streamlit App

With the API running, start the UI:

```bash
streamlit run frontend/streamlit_app.py
```

Use the tabs to:

- Search **Airports** with optional timezone filters
- Search **Routes** with semantic prompts plus hard SQL filters (source, destination, max stops, avoid airline)
- Search **Airlines** with optional country restriction

Queries are embedded locally with the same Gemini model before being sent to the backend. If Streamlit does not have a local Gemini key it falls back to sending `query_text` so the backend can embed on its behalf.

---

## API Reference

All endpoints accept query parameters and return JSON arrays of ranked entities.

- `GET /health` → `{ "ok": true }` if the DB connection and `SELECT 1` succeed.
- `GET /similar-airports`
  - `query_vec` or `query_text` (one required)
  - `tz_prefix`: optional IANA prefix or exact match; converted to `LIKE 'prefix%'`
  - `k`: result count (1-200)
- `GET /similar-routes`
  - `query_vec` or `query_text` (one required)
  - `src`, `dst`: optional IATA codes (uppercased server-side)
  - `stops_max`: integer 0-3 filter
  - `avoid_airline`: optional airline code
  - `k`: result count
- `GET /similar-airlines`
  - `query_vec` or `query_text` (one required)
  - `country`: optional exact country string
  - `k`: result count

All similarity computations happen in MariaDB via `VEC_DISTANCE_COSINE(emb, VEC_FromText(%s))` so results remain consistent with the stored vectors.

---

## Cross-Platform Notes

- **macOS / Linux**: The suggested workflow uses `python3 -m venv`. Make sure `libmariadb` headers are available before installing requirements. Homebrew users can run `brew install mariadb-connector-c`.
- **Windows**: Use `py -3 -m venv .venv` and activate with `.venv\Scripts\activate`. Install the Microsoft Visual C++ Build Tools (required by the MariaDB connector). Docker Desktop for Windows provides the MariaDB service.
- **Containers / CI**: Because the ETL and backend rely solely on environment variables, you can run everything inside containers. Mount or bake the OpenFlights data files and `.env` file as needed.

---

## Verification on Apple M2 Mac

Tested on an Apple M2 (macOS Sonoma 14.x) with Python 3.11, Docker Desktop 4.31, and Homebrew MariaDB Connector/C. Steps performed:

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `docker compose up -d mariadb`
4. `python -m etl.load_openflights`
5. `python -m etl.embeddings --limit 100 --only routes` (sanity run) followed by `python -m etl.embeddings --only all`
6. `uvicorn backend.app.main:app --reload` and verified `GET /health`
7. `streamlit run frontend/streamlit_app.py` and executed sample searches for airports, routes, and airlines

All commands completed successfully; the Streamlit UI returned ranked results for the sample prompts.

---

## Troubleshooting

- **500 errors from similarity endpoints**: Ensure `VEC_DISTANCE_COSINE` exists. You must run MariaDB 11.4+ with the vector plugin; the provided Docker image already includes it.
- **Embedding script exits immediately**: Confirm `GOOGLE_API_KEY` is present and has access to the selected model (`models/text-embedding-004`).
- **503 errors when calling `query_text` endpoints**: The backend needs `GOOGLE_API_KEY`. Either supply one (preferred) or call the API with a precomputed `query_vec`.
- **`mariadb` Python package build failures (macOS)**: Install the Connector/C library via Homebrew before running `pip install -r requirements.txt`.
- **Duplicate route rows after rerunning the loader**: The loader now generates a `route_key` and upserts rows. If you have legacy duplicates, run `python -m etl.load_openflights` once after updating to clean them up.
- **Connection exhaustion during long ETL runs**: If you see `Too many connections`, lower the `--batch` size, close other DB clients, or raise MariaDB’s `max_connections`.

Happy exploring! 
