"""src/evaluate.py
All evaluation logic, metrics and the high-level `ExperimentRunner` are
kept in this module.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Any, List

import numpy as np  # numpy is needed for seeding – keep the import
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  – deferred MPL import
import seaborn as sns  # noqa: E402
from torchvision import transforms as T

from datasets import load_dataset
from diffusers import DiTModel, DDPMScheduler
from fvcore.nn.flop_count import FlopCountAnalysis
from codecarbon import EmissionsTracker
from torchmetrics.image.fid import FrechetInceptionDistance

try:
    from torchmetrics.image.sifid import SpectralInceptionDistance
except ImportError:  # pragma: no cover – optional metric
    SpectralInceptionDistance = None

try:
    from orchid import OrchidCache, OrchidConfig
except ImportError as e:  # pragma: no cover – Orchid is a hard dependency
    raise RuntimeError(
        "[FATAL] ORCHID engine not found – please `pip install orchid==0.1.2`"
    ) from e

from .train import timed, sample_images  # relative import inside the package
from .preprocess import load_prompts, sha256

# ---------------------------------------------------------------------
# Metric wrappers
# ---------------------------------------------------------------------


class FIDMetric:
    """Thin wrapper around torchmetrics FID to simplify usage."""

    def __init__(self, device: torch.device):
        self.fid = FrechetInceptionDistance(reset_real_features=False).to(device)

    @torch.no_grad()
    def update_real(self, imgs: torch.Tensor):
        self.fid.update(imgs, real=True)

    @torch.no_grad()
    def update_fake(self, imgs: torch.Tensor):
        self.fid.update(imgs, real=False)

    def compute(self) -> float:
        return float(self.fid.compute().cpu())


class SFIDMetric:
    """Spectral FID wrapper (optional)."""

    def __init__(self, device: torch.device):
        if SpectralInceptionDistance is None:
            raise RuntimeError("torchmetrics>=0.11 required for sFID")
        self.metric = SpectralInceptionDistance(reset_real_features=False).to(device)

    @torch.no_grad()
    def update_real(self, imgs: torch.Tensor):
        self.metric.update(imgs, real=True)

    @torch.no_grad()
    def update_fake(self, imgs: torch.Tensor):
        self.metric.update(imgs, real=False)

    def compute(self) -> float:
        return float(self.metric.compute().cpu())


# ---------------------------------------------------------------------
# High-level Experiment runner
# ---------------------------------------------------------------------


class ExperimentRunner:
    """Runs one experiment block (one YAML entry)."""

    def __init__(self, cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.cfg = cfg
        self.global_cfg = global_cfg
        self.device = torch.device(global_cfg["device"])
        self.dtype = getattr(torch, global_cfg["dtype"])
        self.output_dir = Path(global_cfg["output_dir"]).expanduser()
        self.fig_dir = Path(global_cfg["figure_dir"]).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- metrics --------------------------- #
    def _prepare_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if "fid" in self.cfg["metrics"]:
            metrics["fid"] = FIDMetric(self.device)
        if "sfid" in self.cfg["metrics"] and SpectralInceptionDistance is not None:
            metrics["sfid"] = SFIDMetric(self.device)
        return metrics

    # ----------------------------- core run ------------------------- #

    @timed
    def run_once(
        self,
        prompts: List[str],
        variant: Dict[str, Any],
        resolution: int,
        seed: int,
    ) -> Dict[str, Any]:
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # ---------------- model & scheduler ------------------------- #
        model_id = self.cfg["model"]
        model: DiTModel = DiTModel.from_pretrained(model_id).to(self.device, dtype=self.dtype)
        model.sample_size = resolution  # override default size if needed
        model.eval()

        steps = self.cfg["sampler"]["steps"]
        scheduler = DDPMScheduler(
            num_inference_steps=steps,
            algorithm_type=self.cfg["sampler"]["algorithm"],
        )

        # ---------------- cache backend ----------------------------- #
        cache_cfg = variant.get("cache")
        if cache_cfg is None:
            context_mgr = torch.no_grad()
        else:
            orchid_conf = OrchidConfig(
                rank_c=cache_cfg.get("rank_c", 0),
                rank_f=cache_cfg.get("rank_f", 0),
                ost_momentum=cache_cfg.get("ost_momentum", 0.0),
                enable_scp=cache_cfg.get("enable_scp", False),
                backend=cache_cfg.get("backend", "orchid"),
            )
            context_mgr = OrchidCache(model, orchid_conf)

        tracker = EmissionsTracker(
            project_name="orchid",
            output_dir=str(self.output_dir),
            measure_power_secs=10,
        )

        with context_mgr:
            tracker.start()
            gen_imgs, latency = sample_images(
                model,
                scheduler,
                prompts,
                batch_size=self.global_cfg["batch_size"],
                device=self.device,
                dtype=self.dtype,
            )
            emissions: float = tracker.stop()

        peak_mem = torch.cuda.max_memory_reserved() / 1024 ** 3  # GiB
        metrics = self._prepare_metrics()
        for x in gen_imgs.split(self.global_cfg["batch_size"]):
            if "fid" in metrics:
                metrics["fid"].update_fake(x.float() / 255.0 * 2 - 1)
            if "sfid" in metrics:
                metrics["sfid"].update_fake(x.float() / 255.0 * 2 - 1)

        out: Dict[str, Any] = {
            "variant": variant["name"],
            "resolution": resolution,
            "seed": seed,
            "latency_sec": latency,
            "peak_vram_gb": peak_mem,
            "energy_j": emissions,
        }
        if "fid" in metrics:
            out["fid"] = metrics["fid"].compute()
        if "sfid" in metrics:
            out["sfid"] = metrics["sfid"].compute()

        # ---------------- FLOPs (proxy) ----------------------------- #
        inp = torch.randn(
            1, model.in_channels, resolution, resolution, device=self.device, dtype=self.dtype
        )
        flops = FlopCountAnalysis(model, (inp,))
        out["macs"] = float(sum(flops.by_module().values()))
        return out

    # ---------------------------- loop ------------------------------ #
    def execute(self) -> List[Dict[str, Any]]:
        print(f"\n========== Running Experiment: {self.cfg['name']} ==========")
        exp_results: List[Dict[str, Any]] = []
        resolutions = self.cfg.get("resolutions", [self.cfg.get("resolution")])
        seeds = self.cfg.get("seeds", [0])

        for res in resolutions:
            for prompt_set in self.cfg.get("prompt_sets", [self.cfg.get("prompts")]):
                prompts = load_prompts(prompt_set, cache_root=Path("data"))
                tag = prompt_set.get("tag", prompt_set["dataset"].split("/")[-1])

                # NOTE: real-image stats should be cached; omitted here.

                for variant in self.cfg["method_variants"]:
                    for seed in seeds:
                        res_dict, _wall = self.run_once(prompts, variant, res, seed)
                        res_dict.update({"prompt_set": tag})
                        exp_results.append(res_dict)
                        self._update_fig(
                            exp_results,
                            metric="fid",
                            topic="fid_vs_memory",
                            condition=f"{tag}_{res}p",
                        )
                        out_file = self.output_dir / f"{self.cfg['name']}_{tag}_{res}p_seed{seed}.json"
                        with open(out_file, "w") as f:
                            json.dump(res_dict, f, indent=2)
                        print(json.dumps(res_dict, indent=2))

        print("========== COMPLETED ==========")
        return exp_results

    # --------------------------- plots ----------------------------- #
    def _update_fig(self, exp_results: List[Dict[str, Any]], metric: str, topic: str, condition: str):
        sns.set_theme(style="whitegrid")
        df = [r for r in exp_results if metric in r]
        if not df:
            return
        plt.figure(figsize=(6, 4))
        sns.barplot(x=[r["variant"] for r in df], y=[r[metric] for r in df])
        for idx, r in enumerate(df):
            plt.text(idx, r[metric], f"{r[metric]:.2f}", ha="center", va="bottom")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} – {condition}")
        plt.tight_layout()
        fig_path = self.fig_dir / f"{topic}_{condition}.pdf"
        plt.savefig(fig_path, bbox_inches="tight", format="pdf")
        plt.close()
