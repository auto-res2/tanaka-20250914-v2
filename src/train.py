"""
src/train.py – training related utilities (currently not used)
----------------------------------------------------------------
A minimal stub kept for future fine-tuning work.  Nothing in the
current experimental workflow calls a training routine, but we keep
this module so that external scripts can safely import
`src.train.noop_train` without having to add try/except guards.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Public no-op placeholder
# -----------------------------------------------------------------------------

def noop_train(*args, **kwargs):  # pragma: no cover
    """A no-op training stub so that callers do not have to gate imports."""
    print("[train] No training routine implemented – skipping …")
