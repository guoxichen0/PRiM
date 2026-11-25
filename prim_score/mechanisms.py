"""Conditional mechanisms and PRiM scores."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd


def validate_data(normal: pd.DataFrame, anomalous: pd.DataFrame) -> None:
    """Validate the two samples used by all public methods."""
    for name, frame in (("normal", normal), ("anomalous", anomalous)):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError(f"{name} must be a non-empty pandas DataFrame")
        if not frame.columns.is_unique:
            raise ValueError(f"{name} must have unique column names")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must contain only finite numeric values")
    if list(normal.columns) != list(anomalous.columns):
        raise ValueError("normal and anomalous columns must match in the same order")
    if len(normal) < 2:
        raise ValueError("normal must contain at least two observations")


def prim_score(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
    node: str,
    parents: Iterable[str],
    *,
    mechanism: Literal["linear", "gp"] = "gp",
    gp_max_samples: int | None = None,
    gp_random_state: int | None = 0,
    gp_optimizer: str | None = None,
) -> float:
    """Compute the PRiM score for one variable.

    The conditional-mean mechanism is fitted only on normal data and then
    reused unchanged for anomalous data. ``mechanism="gp"`` uses an exact
    Gaussian process and is intended for nonlinear additive-noise SEMs.
    """
    parent_list = list(parents)
    y_normal = normal[node].to_numpy(dtype=float)
    y_anomalous = anomalous[node].to_numpy(dtype=float)

    if mechanism not in {"linear", "gp"}:
        raise ValueError("mechanism must be either 'linear' or 'gp'")

    if parent_list and mechanism == "linear":
        x_normal = normal[parent_list].to_numpy(dtype=float)
        x_anomalous = anomalous[parent_list].to_numpy(dtype=float)
        design_normal = np.column_stack((np.ones(len(normal)), x_normal))
        design_anomalous = np.column_stack((np.ones(len(anomalous)), x_anomalous))
        coefficients, *_ = np.linalg.lstsq(design_normal, y_normal, rcond=None)
        residual_normal = y_normal - design_normal @ coefficients
        residual_anomalous = y_anomalous - design_anomalous @ coefficients
    elif parent_list:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
            from sklearn.preprocessing import StandardScaler
        except ImportError as error:
            raise ImportError(
                "Gaussian-process mechanisms require scikit-learn."
            ) from error

        x_normal = normal[parent_list].to_numpy(dtype=float)
        x_anomalous = anomalous[parent_list].to_numpy(dtype=float)
        scaler = StandardScaler().fit(x_normal)
        scaled_normal = scaler.transform(x_normal)
        scaled_anomalous = scaler.transform(x_anomalous)

        if gp_max_samples is not None and len(scaled_normal) > gp_max_samples:
            if gp_max_samples < 2:
                raise ValueError("gp_max_samples must be at least 2 or None")
            rng = np.random.default_rng(gp_random_state)
            positions = np.sort(
                rng.choice(len(scaled_normal), size=gp_max_samples, replace=False)
            )
        else:
            positions = np.arange(len(scaled_normal))

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(np.ones(len(parent_list)), (1e-2, 1e2))
            + WhiteKernel(0.1, (1e-5, 1e1))
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-8,
            normalize_y=True,
            optimizer=gp_optimizer,
            n_restarts_optimizer=0,
            random_state=gp_random_state,
        )
        model.fit(scaled_normal[positions], y_normal[positions])
        residual_normal = y_normal - model.predict(scaled_normal)
        residual_anomalous = y_anomalous - model.predict(scaled_anomalous)
    else:
        normal_mean = float(np.mean(y_normal))
        residual_normal = y_normal - normal_mean
        residual_anomalous = y_anomalous - normal_mean

    residual_sd = float(np.std(residual_normal, ddof=1))
    scale_floor = np.finfo(float).eps * max(
        1.0, float(np.sqrt(np.mean(np.square(residual_normal))))
    )
    denominator = max(residual_sd, scale_floor)
    mean_shift = float(np.mean(residual_anomalous) - np.mean(residual_normal))
    return abs(mean_shift) / denominator


def squared_z_scores(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
) -> dict[str, float]:
    """Return mean per-observation squared Z scores for every variable."""
    validate_data(normal, anomalous)
    normal_values = normal.to_numpy(dtype=float)
    anomalous_values = anomalous.to_numpy(dtype=float)
    means = np.mean(normal_values, axis=0)
    scales = np.std(normal_values, axis=0, ddof=1)
    if np.any(scales <= np.finfo(float).eps):
        bad = [str(normal.columns[i]) for i in np.flatnonzero(scales <= 0)]
        raise ValueError(f"normal-data standard deviation is zero for {bad}")
    values = np.mean(np.square((anomalous_values - means) / scales), axis=0)
    return {str(node): float(values[i]) for i, node in enumerate(normal.columns)}

