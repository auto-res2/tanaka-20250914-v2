"""src/preprocess.py
Data-related helper utilities.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset, load_from_disk  # noqa: F401 – load_from_disk may be used externally

__all__ = ["sha256", "load_prompts"]


def sha256(text: str) -> str:
    """Return the first 10 hex chars of the SHA-256 of the given text."""

    return hashlib.sha256(text.encode()).hexdigest()[:10]


def load_prompts(subset_cfg: Dict[str, str | int], *, cache_root: Path) -> List[str]:
    """Download (or load from local cache) a text field from a HF dataset."""

    ds_name: str = subset_cfg["dataset"]
    split: str = subset_cfg.get("split", "train")
    field: str = subset_cfg["field"]
    take: int | None = subset_cfg.get("take")

    cache_dir = cache_root / sha256(ds_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = load_dataset(ds_name, split=split, cache_dir=str(cache_dir))
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"[DATA] Could not download dataset {ds_name}: {e}") from e

    if field not in ds.column_names:
        raise RuntimeError(f"[DATA] Field '{field}' not in dataset columns {ds.column_names}")

    txt = ds[field]
    if take:
        txt = txt[:take]
    return [str(t) for t in txt]
