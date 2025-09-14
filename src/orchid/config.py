"""Configuration classes for ORCHID method."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class OrchidConfig:
    """Configuration for ORCHID memory optimization method.
    
    Args:
        rank_c: Rank of coarse shared basis (default: 16)
        rank_f: Rank of fine block-specific basis (default: 8)
        ost_momentum: Momentum for Online Subspace Tracker updates (default: 0.005)
        enable_scp: Whether to enable Symplectic Coefficient Predictor (default: False)
        backend: Backend method to use ("orchid", "l2c", "remora") (default: "orchid")
        sparse_ratio: Ratio of coefficients to keep in Sparse Delta Coding (default: 0.15)
        enable_rr: Whether to enable Reversible Refresh (default: True)
        novelty_threshold: Threshold for OST novelty detection (default: 0.1)
    """
    
    rank_c: int = 16
    rank_f: int = 8
    ost_momentum: float = 0.005
    enable_scp: bool = False
    backend: str = "orchid"
    sparse_ratio: float = 0.15
    enable_rr: bool = True
    novelty_threshold: float = 0.1
    reuse_window: int = 10
