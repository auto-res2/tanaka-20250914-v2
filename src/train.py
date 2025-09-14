"""
src/train.py – training related utilities (currently not used)
----------------------------------------------------------------
The original monolithic script contains no model-training logic; all
experiments work with publicly available diffusion checkpoints.  A stub
module is therefore provided so that future iterations can introduce
fine-tuning or additional learning routines without changing the public
API of `src.main`.
"""
from __future__ import annotations

# Placeholder function ---------------------------------------------------------

def noop_train(*args, **kwargs):
    """A no-op training stub so that callers do not have to gate imports."""
    print("[train] No training routine implemented – skipping …")
