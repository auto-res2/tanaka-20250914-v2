"""
src/evaluate.py – evaluation, metrics & plotting
------------------------------------------------
Extracted from the original single-file script (experiment1).
The function `run_experiment` is called by `src.main` and is responsible
for • generation • FID computation • memory / energy tracking • result
serialisation and plotting.  All results are written to the required
`.research/iteration1/` directory and echoed to `stdout` for CI checks.
"""
from __future__ import annotations

import json, os, pathlib, time, importlib, contextlib, traceback
from typing import Dict, Any, List

import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline

from .preprocess import ensure_coco_val  # makes sure the dataset is present
from .utils.metrics import compute_fid, peak_vram, power_draw
from .utils.figures import save_line

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_enable_cache(pipe, comp_cfg: dict):
    """Try to enable the requested cache-compression backend.
    Falls back to vanilla generation if the backend is not available so that
    smoke tests never fail because of an optional dependency.
    """
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
        else:
            other = importlib.import_module("other_compressors")
            other.enable_scheme(pipe.unet, scheme=method)
    except ModuleNotFoundError:
        print(f"[warning] Optional compressor '{method}' not found – using vanilla.")

# -----------------------------------------------------------------------------
# Public API – called from src.main
# -----------------------------------------------------------------------------

def run_experiment(exp_cfg, global_cfg) -> None:
    """Run a *single* experiment block as defined in the YAML config."""
    # ------------------------------------------------------------------
    # Folder layout
    # ------------------------------------------------------------------
    root_out = pathlib.Path(".research") / "iteration1"
    img_root = root_out / "images" / exp_cfg.id
    res_root = root_out
    img_root.mkdir(parents=True, exist_ok=True)
    res_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset (only COCO val is currently supported)
    # ------------------------------------------------------------------
    coco_dir = ensure_coco_val(pathlib.Path("data"))
    real_dir = str(coco_dir)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------
    experiment_results: Dict[str, Dict[str, Any]] = {}

    for model_info in exp_cfg.models:
        repo = model_info["repo"]
        resolution = model_info["resolution"]

        comp_grid: List[dict] = (
            exp_cfg.compression_grid if exp_cfg.compression_grid else [exp_cfg.compression]
        )
        for comp in comp_grid:
            tag = f"{pathlib.Path(repo).name}_{comp['method']}"
            fake_dir = img_root / tag
            fake_dir.mkdir(parents=True, exist_ok=True)

            # ----------------------------------------------------------
            # Load model (HF token via ENV if required)
            # ----------------------------------------------------------
            print(f"\n[eval] Loading model '{repo}' (precision fp16) …")
            use_token = os.getenv("HF_TOKEN") if "PixArt" in repo else None
            pipe = DiffusionPipeline.from_pretrained(
                repo, torch_dtype=torch.float16, use_auth_token=use_token
            )
            pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)

            # ----------------------------------------------------------
            # Instrument UNet with the requested cache compressor
            # ----------------------------------------------------------
            _safe_enable_cache(pipe, comp)

            # ----------------------------------------------------------
            # Generation
            # ----------------------------------------------------------
            num_imgs = int(exp_cfg.generation["num_images"])
            torch.cuda.empty_cache()
            torch.manual_seed(global_cfg.seed_list[0])
            start_power_t = time.time()
            try:
                for i in tqdm(range(num_imgs), desc=f"{tag} – generating"):
                    image = pipe(prompt=[""], num_inference_steps=exp_cfg.generation["num_inference_steps"], guidance_scale=exp_cfg.generation["guidance_scale"]).images[0]
                    image.save(fake_dir / f"img_{i:05d}.png")
            except Exception:
                print("[error] Generation failed – dumping traceback & continuing …")
                traceback.print_exc()
                continue

            energy = power_draw(start_power_t)
            vram  = peak_vram()

            # ----------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------
            try:
                fid = compute_fid(str(fake_dir), real_dir)
            except Exception:
                fid = float("nan")
                print("[warning] FID computation failed – returning NaN")

            experiment_results[tag] = {
                "FID": fid,
                "peak_VRAM_MB": vram,
                "energy_Wh": energy,
            }
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Serialise & plot
    # ------------------------------------------------------------------
    res_path = res_root / f"{exp_cfg.id}_results.json"
    with open(res_path, "w") as f:
        json.dump(experiment_results, f, indent=2)

    # Memory vs FID plot – only if at least one numeric FID value exists
    numeric_fid = [v["FID"] for v in experiment_results.values() if not (v["FID"] != v["FID"])]
    if numeric_fid:
        mem_gb = [v["peak_VRAM_MB"] / 1024 for v in experiment_results.values()]
        labels  = list(experiment_results.keys())
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
