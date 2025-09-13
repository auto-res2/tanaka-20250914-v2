"""preprocess.py
Dataset downloading & DataLoader construction utilities.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict

import torch
import torchvision
from datasets import load_dataset
from torch.utils.data import DataLoader

from .train import fatal

__all__ = [
    "ImageFolderWrapper",
    "build_dataloaders",
]


class ImageFolderWrapper(torch.utils.data.Dataset):
    """Simple wrapper that turns a HuggingFace *image* dataset into tensors."""

    def __init__(self, ds, resolution: int):
        self.ds = ds
        self.res = resolution
        self.tf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    resolution,
                    interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                ),
                torchvision.transforms.CenterCrop(resolution),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(0.5, 0.5),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        if "image" in item:
            img = item["image"]
        elif "img" in item:
            img = item["img"]
        elif "pixel_values" in item:
            img = item["pixel_values"]
        else:
            available_keys = list(item.keys())
            img_key = available_keys[0] if available_keys else None
            if img_key:
                img = item[img_key]
            else:
                raise KeyError(f"No image data found in dataset item. Available keys: {available_keys}")
        return self.tf(img)


# -----------------------------------------------------------------------------
# Public helper to create loaders for all requested resolutions
# -----------------------------------------------------------------------------

def build_dataloaders(cfg: dict) -> Dict[int, DataLoader]:
    """Download the dataset and build one DataLoader per requested resolution."""

    ds_name = cfg["dataset"]["hf_repo"]
    split = cfg["dataset"].get("split", "train")

    print(f"Downloading dataset {ds_name}:{split} …")
    try:
        ds = load_dataset(ds_name, split=split)
    except Exception as ex:
        fatal(f"Failed to load dataset {ds_name}: {ex}")

    max_imgs = cfg["dataset"].get("max_images") or len(ds)
    indices = list(range(max_imgs))

    loaders: Dict[int, DataLoader] = {}
    for res in cfg["dataset"]["resolution"]:
        sub_ds = ds.select(indices)
        dataset = ImageFolderWrapper(sub_ds, res)
        loader = DataLoader(
            dataset,
            batch_size=cfg["experiment_1"]["batch_size"],
            shuffle=False,
            num_workers=cfg["general"].get("num_workers", 0),
        )
        loaders[res] = loader
    return loaders
