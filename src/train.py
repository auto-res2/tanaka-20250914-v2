"""train.py
Utility & (future) training helpers – shared across the project.
Because the published LiMiT experiments do not perform model *training* in the
released script, this file currently only contains small helper utilities that
are required by several other modules (fatal, json_dump, device_sync).  Should
training be added later, this is the place for optimiser / scheduler logic.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Dict

import torch

__all__ = [
    "fatal",
    "json_dump",
    "device_sync",
]


def fatal(msg: str) -> None:
    """Print message and abort the program with a non-zero exit code."""
    print(f"[FATAL] {msg}")
    sys.exit(1)


def json_dump(data: Dict[str, Any], path: pathlib.Path) -> None:
    """Light wrapper that makes sure the parent directory exists before dumping."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def device_sync() -> None:
    """Synchronise CUDA if available – useful for wall-clock timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
