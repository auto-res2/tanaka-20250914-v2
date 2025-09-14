"""src/train.py
Utility functions related to model inference / sampling as extracted from
 the original monolithic script.  Training logic itself is not required for
 the current project but we keep the filename for future extensibility.
"""
from __future__ import annotations

import time
from typing import Callable, List

import torch
from torchvision import transforms as T
from diffusers import DiTModel, DDPMScheduler

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def timed(fn: Callable) -> Callable:
    """Decorator to measure wall-clock time of a function call."""

    def wrapper(*args, **kwargs):  # noqa: ANN001 – we proxy any signature
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        return out, dt

    return wrapper


def _range_to_unit(img: torch.Tensor) -> torch.Tensor:
    """Map image from [-1,1] → [0,255] uint8."""

    img = (img.clamp(-1, 1) + 1) / 2.0  # → [0,1]
    return img.mul(255).type(torch.uint8)


IMAGE_POSTPROC: T.Compose = T.Compose([
    T.Lambda(_range_to_unit),
])


@timed
@torch.no_grad()
def sample_images(
    model: DiTModel,
    scheduler: DDPMScheduler,
    prompts: List[str],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Generate images for a list of text prompts.  Returns uint8 BCHW together
    with latency (sec) thanks to the `@timed` decorator.
    """

    images = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        text_inputs = model.tokenizer(
            batch, padding="max_length", max_length=128, return_tensors="pt"
        ).to(device)

        noise = torch.randn(
            len(batch),
            model.in_channels,
            model.sample_size,
            model.sample_size,
            device=device,
            dtype=dtype,
        )
        latents = noise
        for t in scheduler.timesteps:
            with torch.cuda.amp.autocast(dtype=dtype):
                latent_model_input = scheduler.scale_model_input(latents, t)
                noise_pred = model(
                    latent_model_input, t, encoder_hidden_states=text_inputs.input_ids
                ).sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        imgs = model.decode_first_stage(latents)  # BCHW in [-1,1] fp32
        imgs = IMAGE_POSTPROC(imgs.cpu())
        images.append(imgs)

    return torch.cat(images, dim=0)  # NCHW uint8
