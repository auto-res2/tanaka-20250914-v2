"""src/main.py
Entry-point script.  Supports smoke-test and full-experiment modes as per
specification.
Usage:
    uv run python -m src.main --smoke-test
    uv run python -m src.main --full-experiment
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import yaml

from .evaluate import ExperimentRunner

# ---------------------------------------------------------------------
# YAML default builders (identical to original script)
# ---------------------------------------------------------------------

def _default_smoke_yaml() -> Dict[str, Any]:
    return {
        "global": {
            "device": "cuda",
            "dtype": "bfloat16",
            "batch_size": 2,
            "num_workers": 4,
            "output_dir": "results",
            "figure_dir": "figures",
        },
        "experiments": [
            {
                "name": "exp1_smoke_memory_fidelity",
                "type": "image",
                "method_variants": [
                    {"name": "vanilla", "cache": None},
                    {
                        "name": "orchid",
                        "cache": {
                            "rank_c": 8,
                            "rank_f": 4,
                            "ost_momentum": 0.005,
                            "enable_scp": False,
                        },
                    },
                ],
                "model": "facebook/DiT-XL-2-256",
                "resolution": 256,
                "prompts": {
                    "dataset": "cococaptions",
                    "split": "validation",
                    "field": "caption",
                    "take": 10,
                },
                "sampler": {"steps": 4, "algorithm": "dpm_solver++"},
                "metrics": ["fid"],
            }
        ],
    }


def _default_full_yaml() -> Dict[str, Any]:
    return {
        "global": {
            "device": "cuda",
            "dtype": "bfloat16",
            "batch_size": 8,
            "num_workers": 8,
            "output_dir": "results",
            "figure_dir": "figures",
        },
        "experiments": [
            {
                "name": "experiment1_memory_fidelity_pareto",
                "type": "image",
                "model": "facebook/DiT-XL-2",
                "resolutions": [256, 512, 1024, 4096],
                "prompt_sets": [
                    {
                        "tag": "coco",
                        "dataset": "cococaptions",
                        "split": "validation",
                        "field": "caption",
                        "take": 5000,
                    },
                    {
                        "tag": "laion_art",
                        "dataset": "laion/laion-art",
                        "split": "train",
                        "field": "TEXT",
                        "take": 2000,
                    },
                    {
                        "tag": "nih_xray",
                        "dataset": "pmbaumgartner/nih-chest-xray-reports",
                        "split": "test",
                        "field": "report",
                        "take": 2000,
                    },
                ],
                "sampler": {"steps": 30, "algorithm": "dpm_solver++"},
                "method_variants": [
                    {"name": "vanilla", "cache": None},
                    {"name": "l2c", "cache": {"backend": "l2c"}},
                    {"name": "remora", "cache": {"backend": "remora"}},
                    {
                        "name": "orchid",
                        "cache": {
                            "rank_c": 16,
                            "rank_f": 8,
                            "ost_momentum": 0.005,
                            "enable_scp": False,
                        },
                    },
                ],
                "metrics": [
                    "fid",
                    "sfid",
                    "latency",
                    "peak_vram",
                    "macs",
                    "energy",
                ],
                "seeds": [0, 1, 2],
            }
        ],
    }

# ---------------------------------------------------------------------
# Config I/O helpers
# ---------------------------------------------------------------------

def _ensure_configs_exist(config_dir: Path):
    config_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = config_dir / "smoke_test.yaml"
    full_path = config_dir / "full_experiment.yaml"

    if not smoke_path.exists():
        with open(smoke_path, "w") as f:
            yaml.safe_dump(_default_smoke_yaml(), f)
    if not full_path.exists():
        with open(full_path, "w") as f:
            yaml.safe_dump(_default_full_yaml(), f)


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

def main():  # noqa: D401 – CLI entry-point
    parser = argparse.ArgumentParser(description="ORCHID memory-efficiency experiments")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="run quick CI smoke test")
    group.add_argument("--full-experiment", action="store_true", help="run full experiment")
    parser.add_argument("--config-dir", type=str, default="config", help="directory containing YAML config files")
    args = parser.parse_args()

    cfg_dir = Path(args.config_dir)
    _ensure_configs_exist(cfg_dir)

    cfg_path = cfg_dir / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    print(f"[CONFIG] Loading {cfg_path}")
    with open(cfg_path) as f:
        full_cfg = yaml.safe_load(f)

    global_cfg = full_cfg["global"]
    for exp_cfg in full_cfg["experiments"]:
        runner = ExperimentRunner(exp_cfg, global_cfg)
        runner.execute()


if __name__ == "__main__":
    main()
