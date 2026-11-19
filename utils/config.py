"""Shared config loading for ingest, chain and the UIs.

One loader (and one embedding-model default) instead of three private copies:
ingest-time and query-time MUST resolve the same embedding model, or the
index's vector dimension silently stops matching the query embeddings.
"""

from pathlib import Path

import yaml

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_config(path: str = "config.yaml") -> dict:
    """Parse the YAML config at `path` (relative to the working directory).

    Raises FileNotFoundError with a hint when run from the wrong directory.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"{path} not found — run from the project root.")
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def embedding_model_name(cfg: dict) -> str:
    """The embedding model both ingest and retrieval must agree on."""
    return cfg.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
