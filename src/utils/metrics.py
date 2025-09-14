"""
src/utils/metrics.py - Evaluation metrics and system monitoring
----------------------------------------------------------------
Implements FID computation, VRAM tracking, and power monitoring
for diffusion model experiments.
"""
import os
import time
import subprocess
import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
from scipy import linalg

class ImageDataset(Dataset):
    def __init__(self, path: Union[str, Path], transform=None):
        self.path = Path(path)
        self.transform = transform
        self.images = list(self.path.glob("*.png")) + list(self.path.glob("*.jpg"))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def get_inception_features(dataloader, device='cuda'):
    """Extract Inception-v3 features for FID computation."""
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Inception Distance between two distributions."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    
    covmean_result = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if isinstance(covmean_result, tuple):
        covmean = covmean_result[0]
    else:
        covmean = covmean_result
        
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean_result = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean_result, tuple):
            covmean = covmean_result[0]
        else:
            covmean = covmean_result
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.absolute(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def compute_fid(fake_dir: str, real_dir: str, batch_size: int = 32, device: str = 'cuda') -> float:
    """Compute FID between generated and real images."""
    try:
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        fake_dataset = ImageDataset(fake_dir, transform=transform)
        real_dataset = ImageDataset(real_dir, transform=transform)
        
        if len(fake_dataset) == 0 or len(real_dataset) == 0:
            return float('nan')
        
        fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        fake_features = get_inception_features(fake_loader, device)
        real_features = get_inception_features(real_loader, device)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        fid_score = calculate_fid(mu_fake, sigma_fake, mu_real, sigma_real)
        return float(fid_score)
        
    except Exception as e:
        print(f"[warning] FID computation failed: {e}")
        return float('nan')

def peak_vram() -> float:
    """Get peak VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def power_draw(start_time: float) -> float:
    """Estimate energy consumption in Wh based on runtime and typical GPU power."""
    try:
        runtime_hours = (time.time() - start_time) / 3600
        typical_gpu_power = 300  # Watts for A100
        return runtime_hours * typical_gpu_power
    except Exception:
        return 0.0
