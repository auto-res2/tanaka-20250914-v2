"""
src/preprocess.py – data download & preprocessing utils
------------------------------------------------------
Refactored utility functions used by both training and evaluation.
"""
from __future__ import annotations

import pathlib, zipfile, urllib.request
from typing import List

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

DATA_ROOT = pathlib.Path("data").resolve()
COCO_VAL_ZIP = "https://images.cocodataset.org/zips/val2017.zip"

# -----------------------------------------------------------------------------
# Generic downloader with progress bar
# -----------------------------------------------------------------------------

def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"[data] Downloading {url} → {dest}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    return dest

# -----------------------------------------------------------------------------
# COCO 2017 val split helper
# -----------------------------------------------------------------------------

def ensure_coco_val(dest_root: pathlib.Path = DATA_ROOT) -> pathlib.Path:
    """Download & extract the COCO val2017 image folder if not present."""
    zip_path = _download(COCO_VAL_ZIP, dest_root / "coco_val2017.zip")
    extract_dir = dest_root / "coco" / "val2017"
    if extract_dir.exists():
        return extract_dir

    print("[data] Extracting COCO validation images …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_root / "coco")
    return extract_dir

# -----------------------------------------------------------------------------
# PyTorch dataset wrapper (images only – captions omitted)
# -----------------------------------------------------------------------------
class CocoValDataset(Dataset):
    def __init__(self, root: pathlib.Path, resolution: int):
        self.root = root
        self.files: List[pathlib.Path] = sorted(root.glob("*.jpg"))
        if not self.files:
            raise RuntimeError(f"No *.jpg files found in {root}. Did the download succeed?")
        self.transform = transforms.Compose([
            transforms.CenterCrop(min(Image.open(self.files[0]).size)),
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)
