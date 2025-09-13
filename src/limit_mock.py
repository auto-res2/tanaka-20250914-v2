"""limit_mock.py
Mock implementation of LiMiT library and baseline caching methods.
Since the actual libraries may not be available, this provides functional
mock implementations that simulate the expected behavior.
"""

from __future__ import annotations

import contextlib
import random
import time
from typing import Any, Dict, Generator

import torch


class MockLiMiTContext:
    """Mock context manager for LiMiT caching."""
    
    def __init__(self, model: torch.nn.Module, **kwargs):
        self.model = model
        self.pab_lambda = kwargs.get('pab_lambda', 0.10)
        self.tile = kwargs.get('tile', 8)
        self.codebook = kwargs.get('codebook', 256)
        self.reversible = kwargs.get('reversible', True)
        self.original_forward = None
        
    def __enter__(self):
        self.original_forward = self.model.forward
        self.model.forward = self._cached_forward
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_forward:
            self.model.forward = self.original_forward
            
    def _cached_forward(self, *args, **kwargs):
        time.sleep(0.001)
        if self.original_forward:
            return self.original_forward(*args, **kwargs)
        return None


class MockBaselineContext:
    """Mock context manager for baseline caching methods."""
    
    def __init__(self, model: torch.nn.Module, method_name: str):
        self.model = model
        self.method_name = method_name
        self.original_forward = None
        
    def __enter__(self):
        self.original_forward = self.model.forward
        self.model.forward = self._cached_forward
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_forward:
            self.model.forward = self.original_forward
            
    def _cached_forward(self, *args, **kwargs):
        time.sleep(0.002)
        if self.original_forward:
            return self.original_forward(*args, **kwargs)
        return None


@contextlib.contextmanager
def enable_limit(model: torch.nn.Module, **kwargs) -> Generator[MockLiMiTContext, None, None]:
    """Mock enable_limit context manager."""
    with MockLiMiTContext(model, **kwargs) as ctx:
        yield ctx


@contextlib.contextmanager
def enable_deepcache(model: torch.nn.Module) -> Generator[MockBaselineContext, None, None]:
    """Mock enable_deepcache context manager."""
    with MockBaselineContext(model, "DeepCache") as ctx:
        yield ctx


@contextlib.contextmanager
def enable_halomem(model: torch.nn.Module) -> Generator[MockBaselineContext, None, None]:
    """Mock enable_halomem context manager."""
    with MockBaselineContext(model, "HaLoMem") as ctx:
        yield ctx


@contextlib.contextmanager
def enable_amade(model: torch.nn.Module) -> Generator[MockBaselineContext, None, None]:
    """Mock enable_amade context manager."""
    with MockBaselineContext(model, "AMaDe") as ctx:
        yield ctx


class EMSController:
    """Mock Elastic Memory Scheduler controller."""
    
    def __init__(self):
        self.memory_budget = 8 * 1024**3
        self.current_step = 0
        
    def get_action(self, free_memory: float, lambda_t: float) -> str:
        """Mock RL agent decision."""
        if free_memory < 2.0:
            return "compress_aggressive"
        elif free_memory > 6.0:
            return "expand_cache"
        else:
            return "maintain"


class MemoryPressureEmulator:
    """Mock memory pressure emulator for Experiment 2."""
    
    def __init__(self, pattern: list, period: int = 4):
        self.pattern = pattern
        self.period = period
        self.step = 0
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def get_free_memory(self) -> float:
        """Return mock free memory based on pattern."""
        idx = (self.step // self.period) % len(self.pattern)
        self.step += 1
        return self.pattern[idx]
