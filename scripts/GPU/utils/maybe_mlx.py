from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class MLXEnv:
    available: bool
    mlx: Optional[Any]
    mx: Optional[Any]


def try_import_mlx() -> MLXEnv:
    """Best-effort import of MLX.

    This code is designed to run on macOS Apple silicon.
    If MLX isn't available, the rest of the system should still run (CPU fallback).
    """
    try:
        import mlx  # type: ignore
        import mlx.core as mx  # type: ignore
        return MLXEnv(available=True, mlx=mlx, mx=mx)
    except Exception:
        return MLXEnv(available=False, mlx=None, mx=None)
