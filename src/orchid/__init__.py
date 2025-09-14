"""ORCHID: Online-adaptive Recursively-Compressed HIerarchical Dynamics

A plug-in memory engine for diffusion transformers that implements:
- Hierarchical Shared Basis (HSB)
- Online Subspace Tracker (OST)
- Symplectic Coefficient Predictor (SCP)
- Sparse Delta Coding (SDC)
- Reversible Refresh (RR)
"""

from .config import OrchidConfig
from .cache import OrchidCache

__all__ = ["OrchidConfig", "OrchidCache"]
