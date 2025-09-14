"""
src/main.py – command-line entry point
-------------------------------------
Handles two-phase execution (smoke-test ⇒ full-experiment) according to the
specifications.
"""
from __future__ import annotations

import argparse, pathlib, sys
from types import SimpleNamespace
import yaml

from .evaluate import run_experiment
from .preprocess import ensure_coco_val

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dict_to_ns(d):
    """Recursively convert dictionaries to SimpleNamespace objects."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_ns(x) for x in d]
    return d

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="TA-FJLT-Cache Experiment Runner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Run a quick sanity check (≤2 min)")
    g.add_argument("--full-experiment", action="store_true", help="Run the full paper experiments")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def _run_phase(cfg_path: pathlib.Path):
    cfg_raw = _load_yaml(cfg_path)

    # Ensure datasets are present (currently only COCO)
    coco_dir = ensure_coco_val()

    # Inject local path so that later components can reuse it
    for exp in cfg_raw["experiments"]:
        datasets = exp.get("datasets", {})
        if "coco_val" in datasets:
            datasets["coco_val"]["local_real_dir"] = str(coco_dir)

    # Convert to namespaces for attribute access inside evaluate.py
    cfg_ns = _dict_to_ns(cfg_raw)

    # Execute all experiments defined in the YAML
    for exp in cfg_raw["experiments"]:
        run_experiment(exp, cfg_raw)


def main():
    args = _parse_args()
    config_dir = pathlib.Path(__file__).resolve().parents[1] / "config"
    smoke_cfg = config_dir / "smoke_test.yaml"
    full_cfg = config_dir / "full_experiment.yaml"

    if args.smoke_test:
        _run_phase(smoke_cfg)
    elif args.full_experiment:
        # Phase-1: smoke test
        try:
            print("\n[phase-1] Running smoke test …")
            _run_phase(smoke_cfg)
        except Exception as exc:
            print(f"[error] Smoke test failed – aborting full experiment.\n{exc}")
            sys.exit(1)
        # Phase-2: full experiment
        print("\n[phase-2] Running full experiment …")
        _run_phase(full_cfg)


if __name__ == "__main__":
    main()
