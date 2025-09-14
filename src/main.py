"""
src/main.py – project entry-point
---------------------------------
Implements the command-line interface requested in the specification.
Two mutually exclusive flags trigger either a quick smoke-test or the
full experimental suite.  A failed smoke test aborts the full run.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from types import SimpleNamespace
from typing import Any, Dict

import yaml

from .evaluate import run_experiment
from .preprocess import ensure_coco_val

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dict_to_ns(d):
    """Recursively convert dictionaries → SimpleNamespace for dot access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(x) for x in d]
    return d

# -----------------------------------------------------------------------------
# CLI argument parsing
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="TA-FJLT-Cache Experiment Runner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Run a fast sanity check")
    g.add_argument("--full-experiment", action="store_true", help="Run the full suite")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main orchestration helpers
# -----------------------------------------------------------------------------

def _run_phase(cfg_path: pathlib.Path):
    cfg_raw = _load_yaml(cfg_path)

    # Make sure datasets are present so later steps can reuse the path ---------
    coco_dir = ensure_coco_val()

    # Inject local path for potential downstream reuse (not strictly required) -
    for exp in cfg_raw["experiments"]:
        datasets = exp.get("datasets", {})
        if "coco_val" in datasets:
            datasets["coco_val"]["local_real_dir"] = str(coco_dir)

    # Iterate through experiment blocks
    for exp in cfg_raw["experiments"]:
        run_experiment(exp, cfg_raw)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():  # pragma: no cover – CLI wrapper
    args = _parse_args()
    config_dir = pathlib.Path(__file__).resolve().parents[1] / "config"
    smoke_cfg = config_dir / "smoke_test.yaml"
    full_cfg = config_dir / "full_experiment.yaml"

    if args.smoke_test:
        _run_phase(smoke_cfg)
    elif args.full_experiment:
        # Phase-1 – Smoke test --------------------------------------------------
        try:
            print("\n[phase-1] Running smoke test …")
            _run_phase(smoke_cfg)
        except Exception as exc:  # pragma: no cover
            print(f"[error] Smoke test failed – aborting full experiment.\n{exc}")
            sys.exit(1)
        # Phase-2 – Full experiment -------------------------------------------
        print("\n[phase-2] Running full experiment …")
        _run_phase(full_cfg)


if __name__ == "__main__":
    main()
