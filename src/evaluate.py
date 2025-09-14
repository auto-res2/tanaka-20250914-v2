"""
src/evaluate.py – evaluation, metrics & plotting
------------------------------------------------
Refactored from the original monolithic script.  The public function
`run_experiment` is invoked by `src.main`.
"""
from __future__ import annotations

import json, os, pathlib, time, importlib, traceback, contextlib
from types import SimpleNamespace
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline

# -----------------------------------------------------------------------------
# Optional heavy imports (guarded)
# -----------------------------------------------------------------------------
with contextlib.suppress(ImportError):
    import matplotlib.pyplot as plt
with contextlib.suppress(ImportError):
    from torchvision import transforms
with contextlib.suppress(ImportError):
    from PIL import Image
with contextlib.suppress(ImportError):
    from pytorch_fid.fid_score import calculate_fid_given_paths

from .preprocess import ensure_coco_val

# -----------------------------------------------------------------------------
# Lightweight utility helpers (self-contained to avoid extra modules)
# -----------------------------------------------------------------------------

def compute_fid(fake_dir: str, real_dir: str) -> float:
    """Compute FID between two folders.  Falls back to NaN if libraries missing."""
    try:
        paths = [real_dir, fake_dir]
        # 50 batch size default / 2048 dims match original script
        return float(calculate_fid_given_paths(paths, 50, "cuda" if torch.cuda.is_available() else "cpu", 2048))
    except Exception as err:  # pragma: no cover – robustness over strictness
        print(f"[warning] FID calculation failed: {err}")
        return float("nan")


def peak_vram() -> float:
    """Peak allocated VRAM in MB since the last reset."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def power_draw(start_t: float) -> float:
    """Rough energy proxy = wall-clock seconds × nominal card power (320 W)."""
    watts = 320  # A100 PCIe typical board power
    return (time.time() - start_t) * watts / 3600.0  # Wh


def save_line(xs: List[float], y_lists: List[List[float]], labels: List[str],
              xlabel: str, ylabel: str, title: str, fname: pathlib.Path):
    """Save a simple line plot.  Works even when matplotlib is unavailable."""
    if 'plt' not in globals():
        print('[warning] matplotlib not available – skipping plot generation.')
        return
    plt.figure(figsize=(5, 3.5))
    for ys, lbl in zip(y_lists, labels):
        plt.plot(xs, ys, marker='o', label=lbl)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fname, format='pdf')
    plt.close()

# -----------------------------------------------------------------------------
# Cache-compression switchboard
# -----------------------------------------------------------------------------

def _safe_enable_cache(pipe, comp_cfg: dict):
    """Enable the requested cache compressor if the optional dependency exists."""
    method = comp_cfg.get("method", "vanilla")
    if method == "vanilla":
        return  # no compression requested

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
        else:
            other = importlib.import_module("other_compressors")
            other.enable_scheme(pipe.unet, scheme=method)
    except ModuleNotFoundError:
        print(f"[warning] Optional compressor '{method}' not found – using vanilla.")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def run_experiment(exp_cfg, global_cfg) -> None:  # noqa: C901 – keep flat for clarity
    """Run a single experiment block as defined in a YAML file."""
    # Accept both dicts and namespaces --------------------------------------------------
    if isinstance(exp_cfg, dict):
        exp_cfg = SimpleNamespace(**exp_cfg)
    if isinstance(global_cfg, dict):
        global_cfg = SimpleNamespace(**global_cfg)

    # ------------------------------------------------------------------
    # Folder layout
    # ------------------------------------------------------------------
    root_out = pathlib.Path(".research") / "iteration12"
    img_root = root_out / "images" / exp_cfg.id
    res_root = root_out
    img_root.mkdir(parents=True, exist_ok=True)
    res_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset (COCO val)
    # ------------------------------------------------------------------
    coco_dir = ensure_coco_val(pathlib.Path("data"))
    real_dir = str(coco_dir)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    experiment_results: Dict[str, Dict[str, Any]] = {}

    for model_info in exp_cfg.models:
        repo = model_info["repo"]
        # resolution currently unused but kept for future

        comp_grid: List[dict] = (
            exp_cfg.compression_grid if hasattr(exp_cfg, "compression_grid") and exp_cfg.compression_grid else [exp_cfg.compression]
        )
        for comp in comp_grid:
            tag = f"{pathlib.Path(repo).name}_{comp['method']}"
            fake_dir = img_root / tag
            fake_dir.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # Load model
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
            # Compression backend
            # ----------------------------------------------------------
            _safe_enable_cache(pipe, comp)

            # ----------------------------------------------------------
            # Generation
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
            except Exception:
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
    # Serialise & plot
    # ------------------------------------------------------------------
    res_path = res_root / f"{exp_cfg.id}_results.json"
    with open(res_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    # Memory vs FID plot – only if numeric FID values exist
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
    # Stdout for CI validation
    # ------------------------------------------------------------------
    print("\n[experiment-summary]", json.dumps(experiment_results, indent=2))
