"""main.py – unified CLI entry-point
Supports the following invocations (as required in the prompt):

    uv run python -m src.main --smoke-test          # smoke only
    uv run python -m src.main --full-experiment     # smoke then full
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Tuple

import yaml

from .evaluate import run_experiment_1, run_experiment_2, run_experiment_3
from .train import fatal

# -----------------------------------------------------------------------------
# Constants & paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RESEARCH_DIR = PROJECT_ROOT / ".research" / "iteration10"
IMAGES_DIR = RESEARCH_DIR / "images"
RESULTS_DIR = RESEARCH_DIR  # JSON files stored directly here per prompt

# Ensure directories exist -----------------------------------------------------
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# CLI ARGUMENTS
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="LiMiT experiments runner")
parser.add_argument("--smoke-test", action="store_true", help="run smoke test only")
parser.add_argument(
    "--full-experiment",
    action="store_true",
    help="run full experiment (runs smoke test first)",
)

args = parser.parse_args()

# XOR – exactly one flag must be true
if not (args.smoke_test ^ args.full_experiment):
    fatal("You must specify exactly one of --smoke-test or --full-experiment.")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _load_cfg(name: str) -> Tuple[dict, Path]:
    cfg_path = CONFIG_DIR / name
    if not cfg_path.exists():
        fatal(f"Configuration file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f), cfg_path


# -----------------------------------------------------------------------------
# Phase 1: smoke test (always executed for validation)
# -----------------------------------------------------------------------------
print("==============  SMOKE  TEST  ==============")
smoke_cfg, _ = _load_cfg("smoke_test.yaml")
try:
    run_experiment_1(smoke_cfg, RESULTS_DIR, IMAGES_DIR)
except Exception as ex:  # pragma: no cover – fail-fast with traceback
    traceback.print_exc()
    fatal(f"Smoke test failed: {ex}")

# If only a smoke test was requested we are done here --------------------------
if args.smoke_test:
    print("Smoke test successful – exiting.")
    raise SystemExit(0)

# -----------------------------------------------------------------------------
# Phase 2: full experiment
# -----------------------------------------------------------------------------
print("\n==============  FULL  EXPERIMENT  ==============")
full_cfg, _ = _load_cfg("full_experiment.yaml")
run_experiment_1(full_cfg, RESULTS_DIR, IMAGES_DIR)
run_experiment_2(full_cfg, RESULTS_DIR, IMAGES_DIR)
run_experiment_3(full_cfg, RESULTS_DIR, IMAGES_DIR)
print("All experiments finished successfully.")
