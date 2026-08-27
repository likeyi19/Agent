"""Process-local lifecycle management for loaded EpiZoo models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import torch

from . import epizoo as epizoo_backend


@dataclass(frozen=True)
class _EpiZooModelCacheKey:
    checkpoint_path: str
    device: str
    dtype: torch.dtype


_MODEL_CACHE: dict[_EpiZooModelCacheKey, Any] = {}
_MODEL_CACHE_LOCK = Lock()


def _normalize_device(device: str | torch.device) -> torch.device:
    resolved = epizoo_backend._resolve_device(device)
    if resolved.type == "cuda" and resolved.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def _cache_key(
    checkpoint_path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[_EpiZooModelCacheKey, Path, torch.device]:
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    resolved_device = _normalize_device(device)
    key = _EpiZooModelCacheKey(
        checkpoint_path=str(resolved_checkpoint),
        device=str(resolved_device),
        dtype=dtype,
    )
    return key, resolved_checkpoint, resolved_device


def get_cached_epizoo_model(
    checkpoint_path: str | Path = epizoo_backend.DEFAULT_CHECKPOINT_PATH,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Return one loaded model per normalized checkpoint/device/dtype key.

    Initialization occurs while holding a process-local lock so simultaneous
    cache misses cannot materialize duplicate multi-billion-parameter models.
    A model is registered only after ``load_model`` returns successfully.
    """

    key, resolved_checkpoint, resolved_device = _cache_key(
        checkpoint_path, device, dtype
    )
    with _MODEL_CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        model = epizoo_backend.load_model(
            checkpoint_path=resolved_checkpoint,
            device=resolved_device,
            dtype=dtype,
        )
        _MODEL_CACHE[key] = model
        return model


def clear_epizoo_backend_cache() -> None:
    """Remove cached model references without global CUDA allocator side effects.

    Active callers may retain their own references and continue using a model.
    This function does not call ``torch.cuda.empty_cache()``.
    """

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


__all__ = ["clear_epizoo_backend_cache", "get_cached_epizoo_model"]
