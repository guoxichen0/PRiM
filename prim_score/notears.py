"""NOTEARS structure learning, including an isolated nonlinear worker."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd


def _is_acyclic(weights: np.ndarray) -> bool:
    adjacency = np.asarray(weights) != 0
    np.fill_diagonal(adjacency, False)
    indegree = adjacency.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(indegree == 0))
    visited = 0
    while ready:
        parent = int(ready.pop())
        visited += 1
        for child in np.flatnonzero(adjacency[parent]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    return visited == len(adjacency)


def learn_notears_dag(
    normal: pd.DataFrame,
    *,
    model_type: Literal["linear", "nonlinear"] = "nonlinear",
    mlp_hidden_dims: Sequence[int] = (10,),
    mlp_bias: bool = True,
    loss_type: Literal["l2", "logistic", "poisson"] = "l2",
    lambda1: float | None = None,
    lambda2: float = 0.01,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Learn a linear or nonlinear DAG with the official NOTEARS package."""
    values = normal.to_numpy(dtype=np.float64)
    if model_type == "linear":
        try:
            from notears.linear import notears_linear
        except ImportError as error:
            raise ImportError(
                "Install the official NOTEARS package before learning a DAG."
            ) from error
        weights = notears_linear(
            values,
            lambda1=0.1 if lambda1 is None else float(lambda1),
            loss_type=loss_type,
            max_iter=max_iter,
            h_tol=h_tol,
            rho_max=rho_max,
            w_threshold=w_threshold,
        )
    elif model_type == "nonlinear":
        try:
            import torch
            from notears.nonlinear import NotearsMLP, notears_nonlinear
        except ImportError as error:
            raise ImportError(
                "Nonlinear NOTEARS requires PyTorch and the official NOTEARS "
                "package. Install the `notears` optional dependencies."
            ) from error
        hidden_dims = tuple(int(width) for width in mlp_hidden_dims)
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("mlp_hidden_dims must contain positive integers")
        previous_dtype = torch.get_default_dtype()
        previous_rng_state = torch.random.get_rng_state()
        torch.set_default_dtype(torch.double)
        try:
            if random_state is not None:
                torch.manual_seed(int(random_state))
            model = NotearsMLP(
                dims=[values.shape[1], *hidden_dims, 1],
                bias=bool(mlp_bias),
            )
            weights = notears_nonlinear(
                model,
                values,
                lambda1=0.01 if lambda1 is None else float(lambda1),
                lambda2=float(lambda2),
                max_iter=int(max_iter),
                h_tol=float(h_tol),
                rho_max=float(rho_max),
                w_threshold=float(w_threshold),
            )
        finally:
            torch.random.set_rng_state(previous_rng_state)
            torch.set_default_dtype(previous_dtype)
    else:
        raise ValueError("model_type must be 'linear' or 'nonlinear'")

    adjacency = (np.asarray(weights) != 0).astype(int)
    if not _is_acyclic(adjacency):
        raise ValueError("NOTEARS returned a cyclic adjacency matrix")
    return pd.DataFrame(adjacency, index=normal.columns, columns=normal.columns)


def _supports_nonlinear_notears(python_executable: Path) -> bool:
    if not python_executable.is_file():
        return False
    completed = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import torch; from notears.nonlinear import NotearsMLP, notears_nonlinear",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _resolve_notears_python(python_executable: str | Path | None) -> Path:
    candidates: list[Path] = []
    if python_executable is not None:
        candidates.append(Path(python_executable).expanduser())
    configured = os.environ.get("PRIM_NOTEARS_PYTHON")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(sys.executable))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        conda_root = prefix.parent.parent if prefix.parent.name == "envs" else prefix
        if os.name == "nt":
            candidates.append(conda_root / "envs" / "rca-notears" / "python.exe")
        else:
            candidates.append(conda_root / "envs" / "rca-notears" / "bin" / "python")

    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if str(resolved) in checked:
            continue
        checked.append(str(resolved))
        if _supports_nonlinear_notears(resolved):
            return resolved
    raise ImportError(
        "No Python environment with PyTorch and nonlinear NOTEARS was found. "
        "Install `.[notears]` or set PRIM_NOTEARS_PYTHON to its Python executable. "
        f"Checked: {checked}"
    )


def learn_notears_dag_isolated(
    normal: pd.DataFrame,
    *,
    python_executable: str | Path | None = None,
    **kwargs: object,
) -> pd.DataFrame:
    """Learn NOTEARS in a subprocess so PyTorch stays isolated from PC+RCIT."""
    resolved_python = _resolve_notears_python(python_executable)
    request = {"normal": normal, "kwargs": kwargs}
    with tempfile.TemporaryDirectory(prefix="prim_notears_") as temporary:
        temporary_path = Path(temporary)
        request_path = temporary_path / "request.pkl"
        result_path = temporary_path / "result.pkl"
        with request_path.open("wb") as handle:
            pickle.dump(request, handle, protocol=pickle.HIGHEST_PROTOCOL)
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        completed = subprocess.run(
            [
                str(resolved_python),
                str(Path(__file__).resolve()),
                "--worker-input",
                str(request_path),
                "--worker-output",
                str(result_path),
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"Nonlinear NOTEARS worker failed: {details}")
        with result_path.open("rb") as handle:
            result = pickle.load(handle)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("NOTEARS worker returned an invalid DAG")
    return result


def _worker(input_path: Path, output_path: Path) -> None:
    # Executing this file directly would otherwise let ``prim_score/notears.py``
    # shadow the external ``notears`` package on sys.path.
    module_directory = Path(__file__).resolve().parent
    sys.path = [
        entry
        for entry in sys.path
        if Path(entry or ".").resolve() != module_directory
    ]
    with input_path.open("rb") as handle:
        request = pickle.load(handle)
    result = learn_notears_dag(request["normal"], **request["kwargs"])
    with output_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.worker_input is None or arguments.worker_output is None:
        raise SystemExit("Both --worker-input and --worker-output are required")
    _worker(arguments.worker_input, arguments.worker_output)
