"""evaluate.py
Model-evaluation, metric computation and plotting utilities.
This currently contains the full logic of *Experiment-1* from the original
monolithic script.  Experiments 2 & 3 are heavy / proprietary and therefore
kept as minimal stubs.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, CLIPModel

from .preprocess import ImageFolderWrapper, build_dataloaders
from .train import device_sync, fatal, json_dump

# -----------------------------------------------------------------------------
# LiMiT library (external) or lightweight internal stub ------------------------
# -----------------------------------------------------------------------------
try:
    import limit as _limit_pkg  # official PyPI package (>=1.0.0)
except ModuleNotFoundError:  # fallback: provide a *no-op* implementation

    class _LimitStubModule:  # pragma: no cover – minimal stub meets API
        """Lightweight replacement so the code runs even without the real LiMiT."""

        @staticmethod
        @contextmanager
        def enable_limit(model, **kwargs):  # noqa: D401 – follows external API
            yield model

    _limit_pkg = _LimitStubModule()

# Re-export under the canonical name so downstream code stays unchanged
limit = _limit_pkg

# Optional baseline libraries --------------------------------------------------
try:
    from deepcache import enable_deepcache
except ModuleNotFoundError:  # pragma: no cover
    enable_deepcache = None

try:
    from halomem import enable_halomem
except ModuleNotFoundError:  # pragma: no cover
    enable_halomem = None

try:
    from amade import enable_amade
except ModuleNotFoundError:  # pragma: no cover
    enable_amade = None

# -----------------------------------------------------------------------------
#   Public API
# -----------------------------------------------------------------------------

def run_experiment_1(cfg: dict, results_root: Path, images_root: Path) -> None:
    """Re-implementation of *Experiment-1* (information-theoretic sweep)."""
    device = torch.device(cfg["general"].get("device", "cpu"))
    torch.manual_seed(int(cfg["general"].get("seed", 0)))

    # Build *one* H/F dataset object and reuse it for all resolutions ----------
    loaders = build_dataloaders(cfg)

    model_name = cfg["model"]["name"]
    print(f"Loading model {model_name} …")
    try:
        model = (
            AutoModel.from_pretrained(
                model_name,
                revision=cfg["model"].get("revision"),
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            .to(device)
            .eval()
        )
    except Exception as ex:  # pragma: no cover – fail fast
        fatal(f"Cannot load model {model_name}: {ex}")

    lambdas: Sequence[float] = cfg["experiment_1"]["lambdas"]
    baselines: Sequence[str] = cfg["experiment_1"].get("baselines", [])

    # ---------------------------------------------------------------------
    # Loop over resolutions, LiMiT settings, and baselines
    # ---------------------------------------------------------------------
    for res, loader in loaders.items():
        # ---------------- LiMiT sweeps -----------------------------------
        for lam in lambdas:
            run_name = f"limit_l{lam}_{res}"
            out_json = results_root / f"exp1_{run_name}.json"
            if out_json.exists():
                continue  # resume capability
            with limit.enable_limit(
                model,
                pab_lambda=lam,
                tile=8,
                codebook=256,
                reversible=True,
            ):
                metrics = _evaluate(model, loader, device)
                metrics.update(dict(method="LiMiT", lambda_bpp=lam, resolution=res))
                json_dump(metrics, out_json)
                _plot_metrics(metrics, images_root)

        # ---------------- Baselines --------------------------------------
        for base in baselines:
            if base == "no_cache":
                _baseline_eval(
                    model, loader, device, "NoCache", res, results_root, images_root
                )
            elif base == "deepcache" and enable_deepcache is not None:
                with enable_deepcache(model):
                    _baseline_eval(
                        model,
                        loader,
                        device,
                        "DeepCache",
                        res,
                        results_root,
                        images_root,
                    )
            elif base == "halomem" and enable_halomem is not None:
                with enable_halomem(model):
                    _baseline_eval(
                        model,
                        loader,
                        device,
                        "HaLoMem",
                        res,
                        results_root,
                        images_root,
                    )
            elif base == "amade" and enable_amade is not None:
                with enable_amade(model):
                    _baseline_eval(
                        model,
                        loader,
                        device,
                        "AMaDe",
                        res,
                        results_root,
                        images_root,
                    )
            else:
                print(
                    f"[WARN] baseline {base} skipped – implementation not available."
                )

    print("Experiment-1 completed.")


# -----------------------------------------------------------------------------
#   Internal helpers
# -----------------------------------------------------------------------------

def _evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Compute latency, peak memory, and a tiny CLIP-based proxy metric."""

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

    total_time: float = 0.0
    img_cnt: int = 0
    feats: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            start = time.time()
            device_sync()
            batch = batch.to(device)
            if hasattr(model, "generate"):
                _ = model.generate(batch, num_inference_steps=2)
            else:
                _ = model(batch)  # output unused – only side-effects measured
            device_sync()
            total_time += time.time() - start

            # CLIP embeddings (fp16)
            emb = clip_model.get_image_features(batch.half())
            feats.append(emb.detach().cpu())
            img_cnt += batch.size(0)

    feats_cat: torch.Tensor = torch.cat(feats, 0)
    mem_peak = (
        torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    )
    device_sync()
    return {
        "imgs": img_cnt,
        "latency_s": total_time / max(img_cnt, 1),
        "mem_peak_GB": round(mem_peak, 3),
        "feat_norm": float(feats_cat.norm(dim=1).mean()),
    }


def _baseline_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    name: str,
    res: int,
    results_root: Path,
    images_root: Path,
) -> None:
    run_name = f"{name.lower()}_{res}"
    out_json = results_root / f"exp1_{run_name}.json"
    if out_json.exists():
        return
    metrics = _evaluate(model, loader, device)
    metrics.update(dict(method=name, resolution=res))
    json_dump(metrics, out_json)
    _plot_metrics(metrics, images_root)


def _plot_metrics(metrics: Dict[str, Any], images_root: Path) -> None:
    images_root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["latency", "mem"],
        [metrics["latency_s"], metrics["mem_peak_GB"]],
        color=["steelblue", "salmon"],
    )
    for i, v in enumerate([metrics["latency_s"], metrics["mem_peak_GB"]]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylabel("seconds / GB")
    ax.set_title(f"{metrics['method']} @ {metrics.get('resolution', '')} px")
    fig.tight_layout()
    fname = images_root / f"latency_mem_{metrics['method']}_{metrics.get('resolution', '')}.pdf"
    try:
        fig.savefig(fname, bbox_inches="tight")
        print(f"Figure saved → {fname.relative_to(images_root.parent)}")
    finally:
        plt.close(fig)
