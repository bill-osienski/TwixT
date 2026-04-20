from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

from ..utils.maybe_mlx import try_import_mlx


@dataclass
class RidgeModel:
    feature_names: List[str]
    coef: Any  # mx.array or numpy.ndarray
    intercept: float
    r2: float

    def predict_one(self, x: Dict[str, float]) -> float:
        v = [float(x.get(name, 0.0)) for name in self.feature_names]
        # Support both MLX and numpy arrays
        try:
            # MLX: coef is 1D
            return float((self.coef @ self._as_array(v)) + self.intercept)
        except Exception:
            # numpy
            import numpy as np

            return float(np.dot(self.coef, np.array(v, dtype=np.float32)) + self.intercept)

    def _as_array(self, v: Sequence[float]):
        env = try_import_mlx()
        if env.available and env.mx is not None:
            return env.mx.array(v)
        import numpy as np

        return np.array(v, dtype=np.float32)


def _r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_bar = sum(y_true) / max(1, len(y_true))
    ss_tot = sum((y - y_bar) ** 2 for y in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def fit_ridge(
    samples: List[Dict[str, Any]],
    *,
    feature_names: List[str],
    target_key: str = "bias",
    weight_key: str = "weight",
    l2: float = 1.0,
    min_samples: int = 32,
) -> Optional[RidgeModel]:
    """Fit a tiny ridge regression used for predicted-bias gating.

    We keep this small and robust:
    - weighted samples
    - CPU fallback if MLX isn't available
    """
    rows = []
    ys: List[float] = []
    ws: List[float] = []

    for s in samples:
        if target_key not in s:
            continue
        y = float(s[target_key])
        w = float(s.get(weight_key, 1.0))
        # drop near-zero info samples (very draw heavy or near zero bias)
        if abs(y) < 1e-6:
            continue
        rows.append([float(s.get(name, 0.0)) for name in feature_names])
        ys.append(y)
        ws.append(max(1e-6, w))

    if len(rows) < min_samples:
        return None

    env = try_import_mlx()
    if env.available and env.mx is not None:
        mx = env.mx
        X = mx.array(rows)
        yv = mx.array(ys)
        wv = mx.array(ws)

        # Weighted ridge: solve (X^T W X + l2 I) b = X^T W y
        W = mx.diag(wv)
        Xt = mx.transpose(X)
        XtW = Xt @ W
        A = XtW @ X + l2 * mx.eye(X.shape[1])
        b = XtW @ yv

        coef = mx.linalg.solve(A, b)
        intercept = 0.0
        # r2 on CPU for simplicity
        y_pred = [float(coef @ mx.array(r)) for r in rows]
        r2 = _r2_score(ys, y_pred)
        return RidgeModel(feature_names=feature_names, coef=coef, intercept=intercept, r2=r2)

    # Numpy fallback
    import numpy as np

    X = np.array(rows, dtype=np.float64)
    yv = np.array(ys, dtype=np.float64)
    wv = np.array(ws, dtype=np.float64)

    W = np.diag(wv)
    XtW = X.T @ W
    A = XtW @ X + l2 * np.eye(X.shape[1])
    b = XtW @ yv
    coef = np.linalg.solve(A, b)
    y_pred = (X @ coef)
    r2 = _r2_score(ys, y_pred.tolist())

    return RidgeModel(feature_names=feature_names, coef=coef, intercept=0.0, r2=r2)


def predicted_bias_gate(
    model: Optional[RidgeModel],
    knobs: Dict[str, float],
    *,
    max_abs_bias: float = 0.08,
    min_r2: float = 0.20,
) -> Tuple[bool, Optional[float]]:
    """Return (allowed, predicted_bias).

    If the model isn't trustworthy, we allow and return None.
    """
    if model is None or model.r2 < min_r2:
        return True, None

    pred = model.predict_one(knobs)
    return (abs(pred) <= max_abs_bias), float(pred)
