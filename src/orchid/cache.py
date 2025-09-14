"""ORCHID cache implementation with all components."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

from .config import OrchidConfig


class HierarchicalSharedBasis:
    """Hierarchical Shared Basis (HSB) component."""
    
    def __init__(self, config: OrchidConfig, device: torch.device):
        self.config = config
        self.device = device
        self.coarse_basis: Optional[torch.Tensor] = None
        self.fine_bases: Dict[str, torch.Tensor] = {}
        self.initialized = False
    
    def initialize(self, activations: Dict[str, torch.Tensor]):
        """Initialize bases from sample activations."""
        if self.initialized:
            return
            
        all_acts = torch.cat([act.flatten(1) for act in activations.values()], dim=1)
        U, S, V = torch.svd(all_acts.float())
        self.coarse_basis = U[:, :self.config.rank_c].to(self.device)
        
        for layer_name, act in activations.items():
            act_flat = act.flatten(1).float()
            U_fine, _, _ = torch.svd(act_flat)
            self.fine_bases[layer_name] = U_fine[:, :self.config.rank_f].to(self.device)
        
        self.initialized = True
    
    def compress(self, activation: torch.Tensor, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress activation using hierarchical basis."""
        if not self.initialized:
            return activation, torch.zeros(1, device=self.device)
        
        act_flat = activation.flatten(1).float()
        
        coarse_coeff = torch.mm(act_flat, self.coarse_basis)
        
        if layer_name in self.fine_bases:
            fine_coeff = torch.mm(act_flat, self.fine_bases[layer_name])
            k = int(self.config.sparse_ratio * fine_coeff.numel())
            if k > 0:
                _, indices = torch.topk(fine_coeff.abs().flatten(), k)
                sparse_fine = torch.zeros_like(fine_coeff.flatten())
                sparse_fine[indices] = fine_coeff.flatten()[indices]
                fine_coeff = sparse_fine.reshape(fine_coeff.shape)
        else:
            fine_coeff = torch.zeros(act_flat.shape[0], self.config.rank_f, device=self.device)
        
        return coarse_coeff, fine_coeff
    
    def decompress(self, coarse_coeff: torch.Tensor, fine_coeff: torch.Tensor, 
                   layer_name: str, original_shape: torch.Size) -> torch.Tensor:
        """Decompress coefficients back to activation."""
        if not self.initialized:
            return torch.zeros(original_shape, device=self.device)
        
        if self.coarse_basis is None:
            return torch.zeros(original_shape, device=self.device)
        recon = torch.mm(coarse_coeff, self.coarse_basis.T)
        
        if layer_name in self.fine_bases and fine_coeff.numel() > 0:
            fine_recon = torch.mm(fine_coeff, self.fine_bases[layer_name].T)
            recon = recon + fine_recon
        
        return recon.reshape(original_shape)


class OnlineSubspaceTracker:
    """Online Subspace Tracker (OST) component."""
    
    def __init__(self, config: OrchidConfig):
        self.config = config
        self.momentum = config.ost_momentum
        self.novelty_threshold = config.novelty_threshold
    
    def update_basis(self, hsb: HierarchicalSharedBasis, new_activation: torch.Tensor) -> bool:
        """Update coarse basis using Oja-style update if novelty detected."""
        if not hsb.initialized or hsb.coarse_basis is None:
            return False
        
        act_flat = new_activation.flatten(1).float()
        
        proj = torch.mm(act_flat, hsb.coarse_basis)
        recon = torch.mm(proj, hsb.coarse_basis.T)
        residual = act_flat - recon
        novelty_score = torch.norm(residual, dim=1).mean().item()
        
        if novelty_score > self.novelty_threshold:
            for _ in range(10):  # 10 iterations as specified
                proj = torch.mm(act_flat, hsb.coarse_basis)
                update = torch.mm(act_flat.T, proj) - torch.mm(hsb.coarse_basis, torch.mm(proj.T, proj))
                hsb.coarse_basis = hsb.coarse_basis + self.momentum * update
                
                U, _, V = torch.svd(hsb.coarse_basis)
                hsb.coarse_basis = U[:, :hsb.coarse_basis.shape[1]]
            
            return True
        
        return False


class SymplecticCoefficientPredictor:
    """Symplectic Coefficient Predictor (SCP) component."""
    
    def __init__(self, config: OrchidConfig, rank: int, device: torch.device):
        self.config = config
        self.device = device
        self.rank = rank
        
        self.A = torch.randn(rank, rank, device=device) * 0.01
        self.A = self.A - self.A.T  # Ensure skew-symmetry
        
        self.coefficient_history: List[torch.Tensor] = []
    
    def predict_next(self, current_coeff: torch.Tensor) -> torch.Tensor:
        """Predict next coefficient using Hamiltonian dynamics."""
        if not self.config.enable_scp:
            return current_coeff
        
        dt = 0.1  # Fixed timestep
        next_coeff = current_coeff + dt * torch.mm(current_coeff, self.A)
        
        return next_coeff
    
    def update_dynamics(self, coeff_pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Update dynamics matrix A from coefficient pairs."""
        if len(coeff_pairs) < 2:
            return
        
        lr = 1e-4
        for c_t, c_t1 in coeff_pairs[-10:]:  # Use last 10 pairs
            predicted = self.predict_next(c_t)
            error = c_t1 - predicted
            
            grad_A = torch.mm(c_t.T, error)
            grad_A = grad_A - grad_A.T  # Ensure skew-symmetric gradient
            self.A = self.A + lr * grad_A


class OrchidCache:
    """Main ORCHID cache context manager."""
    
    def __init__(self, model: nn.Module, config: OrchidConfig):
        self.model = model
        self.config = config
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        
        self.hsb = HierarchicalSharedBasis(config, self.device)
        self.ost = OnlineSubspaceTracker(config)
        self.scp = SymplecticCoefficientPredictor(config, config.rank_c, self.device)
        
        self.activation_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.coefficient_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.step_count = 0
        self.reuse_window = getattr(config, 'reuse_window', 10)
        
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_names: List[str] = []
        
    def __enter__(self):
        """Enter context manager and install hooks."""
        self._install_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and remove hooks."""
        self._remove_hooks()
    
    def _install_hooks(self):
        """Install forward hooks to capture activations."""
        def make_hook(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._process_activation(name, output)
                return output
            return hook
        
        for name, module in self.model.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower() or 'attention' in name.lower():
                if len(list(module.children())) == 0:  # Leaf module
                    handle = module.register_forward_hook(make_hook(name))
                    self.hooks.append(handle)
                    self.layer_names.append(name)
    
    def _remove_hooks(self):
        """Remove all installed hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _process_activation(self, layer_name: str, activation: torch.Tensor):
        """Process activation through ORCHID pipeline."""
        self.step_count += 1
        
        if not self.hsb.initialized:
            sample_acts = {layer_name: activation.detach()}
            self.hsb.initialize(sample_acts)
        
        if (layer_name in self.coefficient_cache and 
            self.step_count % self.reuse_window != 0):
            
            coarse_coeff, fine_coeff = self.coefficient_cache[layer_name]
            if self.config.enable_scp:
                coarse_coeff = self.scp.predict_next(coarse_coeff)
            
            cached_activation = self.hsb.decompress(
                coarse_coeff, fine_coeff, layer_name, activation.shape
            )
            activation.data = cached_activation.data
            return
        
        coarse_coeff, fine_coeff = self.hsb.compress(activation.detach(), layer_name)
        
        if self.step_count % 5 == 0:  # Update every 5 steps
            self.ost.update_basis(self.hsb, activation.detach())
        
        self.coefficient_cache[layer_name] = (coarse_coeff, fine_coeff)
        
        if layer_name in self.activation_cache:
            prev_coeff, _ = self.activation_cache.get(layer_name, (coarse_coeff, fine_coeff))
            self.scp.coefficient_history.append((prev_coeff, coarse_coeff))
            
            if len(self.scp.coefficient_history) >= 10:
                self.scp.update_dynamics(self.scp.coefficient_history[-10:])
        
        self.activation_cache[layer_name] = (coarse_coeff, fine_coeff)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        total_cache_size = 0
        for coarse, fine in self.coefficient_cache.values():
            total_cache_size += coarse.numel() * 4  # Assume float32
            total_cache_size += fine.numel() * 4
        
        return {
            "cache_size_mb": total_cache_size / (1024 * 1024),
            "num_cached_layers": len(self.coefficient_cache),
            "compression_ratio": 0.1,  # Placeholder
        }
