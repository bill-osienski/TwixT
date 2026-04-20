"""Value model for TwixT position evaluation.

Ported from assets/js/ai/valueModel.js
Expected model JSON shape (from train_value.py output):
{
  "type": "logistic_regression",
  "generatedAt": "...",
  "feature_keys": [ ... order of features ... ],
  "weights": [bias, w1, w2, ...],
  "preproc": {
    "standardize": true|false,
    "mean": [ ... same length as feature_keys ... ] | null,
    "std":  [ ... same length as feature_keys ... ] | null
  },
  "params": {...},
  "metrics": {...}
}
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.maybe_mlx import try_import_mlx


# Default scale for value model adjustment (from search.json)
DEFAULT_VALUE_MODEL_SCALE = 600.0


@dataclass
class ValueModel:
    """Logistic regression value model for position evaluation."""
    feature_keys: List[str]
    weights: List[float]  # [bias, w1, w2, ...]
    standardize: bool = False
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    scale: float = DEFAULT_VALUE_MODEL_SCALE

    def _sigmoid(self, z: float) -> float:
        """Sigmoid function with numerical stability."""
        if z < -35:
            return 1e-15
        if z > 35:
            return 1 - 1e-15
        return 1.0 / (1.0 + math.exp(-z))

    def _apply_preproc(self, x: List[float]) -> List[float]:
        """Apply standardization if enabled."""
        if not self.standardize or self.mean is None or self.std is None:
            return x
        if len(self.mean) != len(x) or len(self.std) != len(x):
            return x
        out = []
        for i in range(len(x)):
            denom = self.std[i] if self.std[i] != 0 else 1.0
            out.append((x[i] - self.mean[i]) / denom)
        return out

    def _build_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Build feature vector in the order the model expects."""
        x = []
        for key in self.feature_keys:
            val = features.get(key, 0.0)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = 0.0
            x.append(float(val))
        return x

    def evaluate(self, features: Dict[str, float]) -> Dict[str, Optional[float]]:
        """Evaluate the model on given features.

        Returns:
            Dict with:
            - probability: P(win | features)
            - logit: bias + w·x (before sigmoid)
            - adjustment: (probability - 0.5) * scale
        """
        if not self.weights or len(self.weights) < 2:
            return {"probability": None, "logit": None, "adjustment": None}

        # Build and preprocess feature vector
        x = self._build_feature_vector(features)
        x = self._apply_preproc(x)

        # Check shape
        if len(x) + 1 != len(self.weights):
            return {"probability": None, "logit": None, "adjustment": None}

        # Compute logit: bias + w·x
        z = self.weights[0]  # bias
        for i in range(len(x)):
            z += self.weights[i + 1] * x[i]

        # Compute probability
        p = self._sigmoid(z)

        return {
            "probability": p,
            "logit": z,
            "adjustment": (p - 0.5) * self.scale
        }


# Global cached model
_cached_model: Optional[ValueModel] = None
_cached_path: Optional[str] = None


def load_value_model(path: Path) -> Optional[ValueModel]:
    """Load value model from JSON file.

    Args:
        path: Path to value-model.json

    Returns:
        ValueModel if loaded successfully, None otherwise
    """
    global _cached_model, _cached_path

    # Return cached if same path
    str_path = str(path)
    if _cached_model is not None and _cached_path == str_path:
        return _cached_model

    if not path.exists():
        return None

    try:
        d = json.loads(path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        return None

    # Validate required fields
    feature_keys = d.get("feature_keys", [])
    weights = d.get("weights", [])

    if not feature_keys or not weights:
        return None
    if len(weights) != len(feature_keys) + 1:
        return None

    # Optional preprocessing
    preproc = d.get("preproc", {})
    standardize = preproc.get("standardize", False)
    mean = preproc.get("mean") if standardize else None
    std = preproc.get("std") if standardize else None

    # Validate preproc if present
    if standardize:
        if not mean or not std:
            standardize = False
        elif len(mean) != len(feature_keys) or len(std) != len(feature_keys):
            standardize = False

    model = ValueModel(
        feature_keys=feature_keys,
        weights=weights,
        standardize=standardize,
        mean=mean,
        std=std,
    )

    _cached_model = model
    _cached_path = str_path
    return model


def get_cached_model() -> Optional[ValueModel]:
    """Get the currently cached model."""
    return _cached_model


def try_load_value_model() -> Optional[ValueModel]:
    """Try to load value model from common locations."""
    candidates = [
        Path("value-model.json"),
        Path("assets/value-model.json"),
    ]
    for path in candidates:
        model = load_value_model(path)
        if model is not None:
            return model
    return None
