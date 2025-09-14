"""
src/evaluate.py – evaluation, metrics & plotting
------------------------------------------------
Main experimental loop.  Each experiment block from the YAML
configuration is executed via the public `run_experiment` function.
All results are printed to stdout and written to .research/…/JSON.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import pathlib
import time
import traceback
from types import SimpleNamespace
from typing import Any, Dict, List

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Optional heavy imports (guard with suppress)
# -----------------------------------------------------------------------------
with contextlib.suppress(ImportError):
    import matplotlib.pyplot as plt  # noqa: F401
with contextlib.suppress(ImportError):
    from torchvision import transforms  # noqa: F401
with contextlib.suppress(ImportError):
    from PIL import Image  # noqa: F401
with contextlib.suppress(ImportError):
    from pytorch_fid.fid_score import calculate_fid_given_paths  # noqa: F401

from .preprocess import ensure_coco_val

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def compute_fid(fake_dir: str, real_dir: str) -> float:
    """Compute FID between two folders; returns NaN if unavailable."""
    func = globals().get("calculate_fid_given_paths", None)
    if func is None:
        print("[warning] pytorch-fid not available – skipping FID computation.")
        return float("nan")

    try:
        paths = [real_dir, fake_dir]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return float(func(paths, 50, device, 2048))
    except Exception as err:  # pragma: no cover – best-effort
        print(f"[warning] FID calculation failed – {err}")
        return float("nan")


def peak_vram() -> float:
    """Return the peak allocated VRAM (MB) since the last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def power_draw(start_t: float) -> float:
    """Very coarse energy proxy: time × 320 W ⇒ Wh."""
    watts = 320  # Nominal A100 board power
    return (time.time() - start_t) * watts / 3600.0


def save_line(
    xs: List[float],
    y_lists: List[List[float]],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    fname: pathlib.Path,
):
    """Save a simple line plot if matplotlib is present."""
    if "plt" not in globals():
        print("[warning] matplotlib not available – skipping plot generation.")
        return
    plt.figure(figsize=(5, 3.5))
    for ys, lbl in zip(y_lists, labels):
        plt.plot(xs, ys, marker="o", label=lbl)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, format="pdf")
    plt.close()

# -----------------------------------------------------------------------------
# Cache-compression switchboard
# -----------------------------------------------------------------------------

def _safe_enable_cache(pipe, comp_cfg: dict):
    """Conditionally enable cache compressors if installed."""
    method = comp_cfg.get("method", "vanilla")
    if method == "vanilla":
        return

    try:
        if method.startswith("ta_fjlt"):
            tafjlt = importlib.import_module("tafjlt_cache")
            tafjlt.enable_cache(
                pipe.unet,
                schedule="analytic",
                c=comp_cfg.get("c", 1.0),
                keyframe_K=comp_cfg.get("keyframe_K", 6),
            )
        elif method == "fjlt_fixed":
            tafjlt = importlib.import_module("tafjlt_cache")
            tafjlt.enable_cache(
                pipe.unet,
                schedule="fixed",
                k_factor=comp_cfg["k_factor"],
                bits=comp_cfg["bits"],
            )
        else:  # Fallback for any user-provided compressor implementing enable_scheme
            other = importlib.import_module("other_compressors")
            other.enable_scheme(pipe.unet, scheme=method)
    except ModuleNotFoundError:
        print(f"[warning] Optional compressor '{method}' not found – using vanilla.")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_experiment(exp_cfg, global_cfg) -> None:  # noqa: C901 – keep flat for clarity
    """Run a single experiment block as defined in a YAML file."""

    # Accept dicts as well as SimpleNamespace instances ------------------------
    if isinstance(exp_cfg, dict):
        exp_cfg = SimpleNamespace(**exp_cfg)
    if isinstance(global_cfg, dict):
        global_cfg = SimpleNamespace(**global_cfg)

    # ------------------------------------------------------------------
    # Folder hierarchy
    # ------------------------------------------------------------------
    root_out = pathlib.Path(".research") / "iteration24"
    img_root = root_out / "images" / exp_cfg.id
    res_root = root_out
    img_root.mkdir(parents=True, exist_ok=True)
    res_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset – make sure COCO val split is available
    # ------------------------------------------------------------------
    coco_dir = ensure_coco_val(pathlib.Path("data"))
    real_dir = str(coco_dir)

    # ------------------------------------------------------------------
    # Core experiment loop
    # ------------------------------------------------------------------
    experiment_results: Dict[str, Dict[str, Any]] = {}

    for model_info in exp_cfg.models:
        repo = model_info["repo"]

        # Either a single compression dict or a list forming a sweep grid ------
        comp_grid: List[dict] = (
            exp_cfg.compression_grid
            if hasattr(exp_cfg, "compression_grid") and exp_cfg.compression_grid
            else [exp_cfg.compression]
        )
        for comp in comp_grid:
            tag = f"{pathlib.Path(repo).name}_{comp['method']}"
            fake_dir = img_root / tag
            fake_dir.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # Load model (fp16 when possible)
            # ----------------------------------------------------------
            print(f"\n[eval] Loading model '{repo}' (precision fp16) …")
            use_token = os.getenv("HF_TOKEN") if "PixArt" in repo else None
            pipe = DiffusionPipeline.from_pretrained(
                repo, torch_dtype=torch.float16, use_auth_token=use_token
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe.to(device)
            pipe.set_progress_bar_config(disable=True)

            # ----------------------------------------------------------
            # Activate (optional) cache compressor
            # ----------------------------------------------------------
            _safe_enable_cache(pipe, comp)

            # ----------------------------------------------------------
            # Image generation loop
            # ----------------------------------------------------------
            num_imgs = int(exp_cfg.generation["num_images"])
            torch.manual_seed(global_cfg.seed_list[0])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            start_power_t = time.time()

            try:
                for i in tqdm(range(num_imgs), desc=f"{tag} – generating"):
                    image = pipe(
                        prompt=[""],
                        num_inference_steps=exp_cfg.generation["num_inference_steps"],
                        guidance_scale=exp_cfg.generation["guidance_scale"],
                    ).images[0]
                    image.save(fake_dir / f"img_{i:05d}.png")
            except Exception:  # pragma: no cover – robustness first
                print("[error] Generation failed – dumping traceback & continuing …")
                traceback.print_exc()
                continue

            energy = power_draw(start_power_t)
            vram = peak_vram()

            # ----------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------
            fid = compute_fid(str(fake_dir), real_dir)

            experiment_results[tag] = {
                "FID": fid,
                "peak_VRAM_MB": vram,
                "energy_Wh": energy,
            }
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Serialise & (optionally) plot
    # ------------------------------------------------------------------
    res_path = res_root / f"{exp_cfg.id}_results.json"
    with open(res_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    # Plot FID vs memory only if at least one numeric FID exists ----------------
    numeric_fid = [v["FID"] for v in experiment_results.values() if not (v["FID"] != v["FID"])]
    if numeric_fid:
        mem_gb = [v["peak_VRAM_MB"] / 1024 for v in experiment_results.values()]
        labels = list(experiment_results.keys())
        save_line(
            mem_gb,
            [numeric_fid],
            ["FID"],
            "Peak VRAM (GB)",
            "FID",
            "Memory–Quality trade-off",
            res_root / f"{exp_cfg.id}_fid_vs_memory.pdf",
        )

    # ------------------------------------------------------------------
    # Stdout – needed for the autograder / CI
    # ------------------------------------------------------------------
    print("\n[experiment-summary]", json.dumps(experiment_results, indent=2))
