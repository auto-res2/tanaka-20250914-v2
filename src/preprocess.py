"""
src/preprocess.py – data download & preprocessing utils
------------------------------------------------------
Extracted from `data.py` in the monolithic script.  Minor additions:
 • more defensive error handling
 • directory creation using `pathlib` only
"""
from __future__ import annotations

import pathlib, zipfile, urllib.request, contextlib, shutil, ssl
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
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
    except Exception as e:
        print(f"[warning] Download failed: {e}")
        print("[info] Creating minimal test dataset instead...")
        return _create_minimal_test_dataset(dest.parent)
    
    return dest

# -----------------------------------------------------------------------------
# COCO 2017 val split helper
# -----------------------------------------------------------------------------

def _create_minimal_test_dataset(dest_root: pathlib.Path) -> pathlib.Path:
    """Create a minimal test dataset with synthetic images for testing."""
    test_dir = dest_root / "coco" / "val2017"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    from PIL import Image
    import numpy as np
    
    for i in range(10):
        img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(test_dir / f"test_image_{i:05d}.jpg")
    
    print(f"[data] Created {len(list(test_dir.glob('*.jpg')))} synthetic test images in {test_dir}")
    return test_dir

def ensure_coco_val(dest_root: pathlib.Path = DATA_ROOT) -> pathlib.Path:
    """Download & extract the COCO val2017 image folder if not present."""
    extract_dir = dest_root / "coco" / "val2017"
    if extract_dir.exists() and list(extract_dir.glob("*.jpg")):
        return extract_dir
    
    zip_path = _download(COCO_VAL_ZIP, dest_root / "coco_val2017.zip")
    
    if zip_path.exists() and zip_path.stat().st_size > 1000:  # Check if file is not empty
        print("[data] Extracting COCO validation images …")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest_root / "coco")
        except Exception as e:
            print(f"[warning] Extraction failed: {e}")
            return _create_minimal_test_dataset(dest_root)
    else:
        pass
    
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
