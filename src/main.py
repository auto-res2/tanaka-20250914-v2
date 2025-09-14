"""
src/main.py – command-line entry point
-------------------------------------
Implements the two-phase execution requested in the specification.
The script can be invoked as module (`python -m src.main …`).
"""
from __future__ import annotations

import argparse, pathlib, sys, textwrap
import yaml

from .evaluate import run_experiment
from .preprocess import ensure_coco_val

# -----------------------------------------------------------------------------
# Configuration loader
# -----------------------------------------------------------------------------

def _load_yaml(path: pathlib.Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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
    cfg = _load_yaml(cfg_path)
    # Prepare datasets (currently only COCO is required)
    coco_dir = ensure_coco_val()
    for exp in cfg["experiments"]:
        if "coco_val" in exp.get("datasets", {}):
            exp["datasets"]["coco_val"]["local_real_dir"] = str(coco_dir)

    # Execute all experiments defined in the YAML
    for exp in cfg["experiments"]:
        run_experiment(exp, cfg)


def main():
    args = _parse_args()
    config_dir = pathlib.Path(__file__).resolve().parents[1] / "config"
    smoke_cfg = config_dir / "smoke_test.yaml"
    full_cfg  = config_dir / "full_experiment.yaml"

    if args.smoke_test:
        _run_phase(smoke_cfg)
    elif args.full_experiment:
        # Phase-1: quick smoke test – abort on failure
        try:
            print("\n[phase-1] Running smoke test …")
            _run_phase(smoke_cfg)
        except Exception:
            print("[error] Smoke test failed – aborting full experiment.")
            sys.exit(1)
        # Phase-2: full experiment
        print("\n[phase-2] Running full experiment …")
        _run_phase(full_cfg)


if __name__ == "__main__":
    main()
