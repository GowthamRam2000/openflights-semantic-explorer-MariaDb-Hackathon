import json
import os
import time
from functools import lru_cache
from typing import Iterable, List, Sequence

from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
_RAW_DEFAULT_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004").strip()
_RAW_MULTI_MODEL = os.getenv("GEMINI_MULTI_EMBED_MODEL", _RAW_DEFAULT_MODEL).strip()
_EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))


def _normalize_model(name: str) -> str:
    name = (name or "").strip()
    aliases = {
        "gemini-embedding-001": "models/embedding-001",
        "embedding-001": "models/embedding-001",
        "text-embedding-004": "models/text-embedding-004",
    }
    if name.startswith("models/") or name.startswith("tunedModels/"):
        return name
    return aliases.get(name, f"models/{name}")


_DEFAULT_MODEL = _normalize_model(_RAW_DEFAULT_MODEL)
_MULTI_MODEL = _normalize_model(_RAW_MULTI_MODEL)


@lru_cache(maxsize=1)
def _genai():
    if not _API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not configured; provide query_vec instead of query_text")
    import google.generativeai as genai  

    genai.configure(api_key=_API_KEY)
    return genai


def _needs_multilingual(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def _pick_model(text: str, force_multilingual: bool) -> str:
    if force_multilingual:
        return _MULTI_MODEL
    if _needs_multilingual(text) and _MULTI_MODEL:
        return _MULTI_MODEL
    return _DEFAULT_MODEL


def embed_text(
    text: str,
    *,
    task_type: str = "RETRIEVAL_QUERY",
    force_multilingual: bool = False,
) -> List[float]:
    """Embed text via Gemini. Raises if no API key is configured."""

    text = (text or "").strip()
    if not text:
        raise ValueError("Cannot embed empty text")

    model = _pick_model(text, force_multilingual)
    genai = _genai()
    last_exc: Exception | None = None
    for attempt in range(4):
        try:
            resp = genai.embed_content(
                model=model,
                content=text,
                task_type=task_type,
                output_dimensionality=_EMBED_DIM,
            )
            embedding = resp.get("embedding") or resp.get("values")
            if not isinstance(embedding, Sequence):
                raise RuntimeError("Unexpected embedding response shape")
            vector = [float(x) for x in embedding]
            if len(vector) != _EMBED_DIM:
                raise RuntimeError(
                    f"Model {model} returned {len(vector)} dims; expected {_EMBED_DIM}"
                )
            return vector
        except Exception as exc:  
            last_exc = exc
            if attempt == 3:
                break
            time.sleep(0.4 * (2 ** attempt))
    raise RuntimeError(f"Embedding failed after retries: {last_exc}")


def vector_to_text(vec: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"


def sanitize_vector_string(vec_text: str) -> str:
    """Validate a JSON-like vector string and normalise formatting."""

    try:
        data = json.loads(vec_text)
    except json.JSONDecodeError as exc:  
        raise ValueError("Vector must be JSON array text, e.g. '[0.1, 0.2]'") from exc

    if not isinstance(data, list) or not data:
        raise ValueError("Vector JSON must decode to a non-empty list")

    return vector_to_text(float(x) for x in data)


def embedding_available() -> bool:
    return bool(_API_KEY)


__all__ = [
    "embed_text",
    "sanitize_vector_string",
    "vector_to_text",
    "embedding_available",
]
