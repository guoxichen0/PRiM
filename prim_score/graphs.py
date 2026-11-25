"""Graph utilities for DAGs and project-format CPDAGs."""

from __future__ import annotations

import time
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
import pandas as pd


def validate_graph(graph: pd.DataFrame, columns: pd.Index, *, dag: bool) -> np.ndarray:
    """Validate a parent-row/child-column 0/1 adjacency matrix."""
    if not isinstance(graph, pd.DataFrame):
        raise TypeError("graph must be a pandas DataFrame")
    if list(graph.index) != list(columns) or list(graph.columns) != list(columns):
        raise ValueError("graph row/column labels must match the data columns")
    values = graph.to_numpy(dtype=float)
    if not np.isin(values, (0.0, 1.0)).all() or np.any(np.diag(values) != 0):
        raise ValueError("graph must use 0/1 entries and have a zero diagonal")
    if dag and not is_acyclic(values != 0):
        raise ValueError("the supplied DAG contains a directed cycle")
    return values.astype(int)


def is_acyclic(adjacency: np.ndarray) -> bool:
    adjacency = np.asarray(adjacency, dtype=bool)
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


def parents_from_dag(dag: pd.DataFrame) -> dict[str, list[str]]:
    values = validate_graph(dag, dag.columns, dag=True)
    nodes = [str(node) for node in dag.columns]
    return {
        nodes[child]: [nodes[parent] for parent in np.flatnonzero(values[:, child])]
        for child in range(len(nodes))
    }


def _v_structures(adjacency: np.ndarray) -> set[tuple[int, int, int]]:
    skeleton = (adjacency != 0) | (adjacency.T != 0)
    structures: set[tuple[int, int, int]] = set()
    for child in range(len(adjacency)):
        parents = np.flatnonzero(adjacency[:, child] & ~adjacency[child, :])
        for left, right in combinations((int(x) for x in parents), 2):
            if not skeleton[left, right]:
                structures.add((min(left, right), child, max(left, right)))
    return structures


def enumerate_consistent_dags(
    cpdag: pd.DataFrame,
    *,
    max_dags: int = 100_000,
) -> list[pd.DataFrame]:
    """Enumerate DAGs in a small CPDAG's Markov equivalence class.

    This transparent enumerator is intended for Algorithm 1 and small or
    moderately ambiguous CPDAGs. Its worst-case cost is exponential in the
    number of undirected edges; Algorithm 2 avoids this global enumeration.
    """
    values = validate_graph(cpdag, cpdag.columns, dag=False)
    directed = ((values == 1) & (values.T == 0)).astype(int)
    undirected = [
        (i, j)
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if values[i, j] == values[j, i] == 1
    ]
    if 2 ** len(undirected) > max_dags:
        raise ValueError(
            "CPDAG has too many undirected-edge orientations for exhaustive "
            f"enumeration ({2 ** len(undirected):,} > max_dags={max_dags:,}). "
            "Use local_prim or raise max_dags deliberately."
        )
    target_v_structures = _v_structures(directed)
    dags: list[pd.DataFrame] = []
    for choices in product((0, 1), repeat=len(undirected)):
        candidate = directed.copy()
        for (left, right), choice in zip(undirected, choices):
            candidate[left, right] = int(choice == 0)
            candidate[right, left] = int(choice == 1)
        if is_acyclic(candidate) and _v_structures(candidate) == target_v_structures:
            dags.append(pd.DataFrame(candidate, index=cpdag.index, columns=cpdag.columns))
    if not dags:
        raise ValueError("the supplied CPDAG has no consistent DAG extension")
    return dags


def possible_parent_sets(cpdag: pd.DataFrame, node: str) -> list[list[str]]:
    """Return locally valid parent sets from local IDA (Algorithm 2)."""
    values = validate_graph(cpdag, cpdag.columns, dag=False)
    nodes = [str(item) for item in cpdag.columns]
    if node not in nodes:
        raise ValueError(f"unknown node: {node}")
    target = nodes.index(node)
    directed_parents = [
        i for i in range(len(nodes)) if values[i, target] == 1 and values[target, i] == 0
    ]
    siblings = [
        i for i in range(len(nodes)) if values[i, target] == values[target, i] == 1
    ]
    result: list[list[str]] = []
    for mask in product((0, 1), repeat=len(siblings)):
        new_parents = [s for s, take in zip(siblings, mask) if take]
        incoming = directed_parents + new_parents
        locally_valid = all(
            left == right or values[left, right] == 1 or values[right, left] == 1
            for left in new_parents
            for right in incoming
        )
        if locally_valid:
            result.append([nodes[i] for i in incoming])
    return result


def topological_layers(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=bool)
    indegree = adjacency.sum(axis=0).astype(int)
    layers = np.zeros(len(adjacency), dtype=int)
    ready = list(np.flatnonzero(indegree == 0))
    visited = 0
    while ready:
        parent = int(ready.pop(0))
        visited += 1
        for child in np.flatnonzero(adjacency[parent]):
            child = int(child)
            layers[child] = max(layers[child], layers[parent] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(adjacency):
        raise ValueError("graph contains a directed cycle")
    return layers


def directed_reachability(cpdag: pd.DataFrame) -> np.ndarray:
    values = validate_graph(cpdag, cpdag.columns, dag=False)
    reach = (values == 1) & (values.T == 0)
    if not is_acyclic(reach):
        raise ValueError("the directed part of the CPDAG contains a cycle")
    for middle in range(len(reach)):
        reach |= reach[:, [middle]] & reach[[middle], :]
    np.fill_diagonal(reach, False)
    return reach


def independence_alpha(sample_size: int) -> float:
    """Use the sample-size-dependent CI level from the experiments."""
    if sample_size <= 1_000:
        return 0.05
    if sample_size <= 5_000:
        return 0.01
    return 0.001


def _causal_learn_to_cpdag(graph: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Convert causal-learn endpoints to parent-row/child-column encoding."""
    graph = np.asarray(graph)
    cpdag = np.zeros(graph.shape, dtype=int)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            endpoints = (int(graph[left, right]), int(graph[right, left]))
            if endpoints == (0, 0):
                continue
            if endpoints == (-1, 1):
                cpdag[left, right] = 1
            elif endpoints == (1, -1):
                cpdag[right, left] = 1
            elif endpoints == (-1, -1):
                cpdag[left, right] = cpdag[right, left] = 1
            else:
                raise ValueError(
                    "PC returned an unsupported endpoint pattern between "
                    f"{names[left]} and {names[right]}: {endpoints}"
                )
    return pd.DataFrame(cpdag, index=names, columns=names)


def _learn_xges(
    data: pd.DataFrame,
    *,
    bic_alpha: float,
    extended_search: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from xges import XGES
    except ImportError as error:
        raise ImportError(
            "Linear CPDAG discovery requires XGES. Install it with "
            "`python -m pip install xges`."
        ) from error
    learner = XGES(alpha=float(bic_alpha))
    pdag = learner.fit(
        data.to_numpy(dtype=float),
        extended_search=bool(extended_search),
        verbose=0,
    )
    matrix = pdag.to_adjacency_matrix().astype(int)
    cpdag = pd.DataFrame(matrix, index=data.columns, columns=data.columns)
    return cpdag, {
        "learner": "XGES",
        "bic_alpha": float(bic_alpha),
        "extended_search": bool(extended_search),
    }


def _learn_pc_rcit(
    data: pd.DataFrame,
    *,
    alpha: float,
    rcit_approx: str,
    rcit_num_f: int,
    rcit_num_f2: int,
    random_state: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as error:
        raise ImportError(
            "Nonlinear CPDAG discovery requires causal-learn. Install it "
            "with `python -m pip install causal-learn`."
        ) from error

    previous_state = np.random.get_state()
    if random_state is not None:
        np.random.seed(int(random_state))
    try:
        causal_graph = pc(
            data.to_numpy(dtype=float),
            alpha=float(alpha),
            indep_test="rcit",
            stable=True,
            uc_rule=0,
            uc_priority=2,
            verbose=False,
            show_progress=False,
            node_names=[str(column) for column in data.columns],
            approx=rcit_approx,
            num_f=int(rcit_num_f),
            num_f2=int(rcit_num_f2),
            rcit=True,
        )
    finally:
        np.random.set_state(previous_state)

    names = [node.get_name() for node in causal_graph.G.nodes]
    cpdag = _causal_learn_to_cpdag(causal_graph.G.graph, names)
    return cpdag, {
        "learner": "PC+RCIT",
        "alpha": float(alpha),
        "ci_test": "rcit",
        "rcit_approx": rcit_approx,
        "rcit_num_f": int(rcit_num_f),
        "rcit_num_f2": int(rcit_num_f2),
        "random_state": random_state,
    }


def learn_cpdag(
    data: pd.DataFrame,
    *,
    model_type: Literal["linear", "nonlinear"] = "nonlinear",
    alpha: float | None = None,
    xges_bic_alpha: float = 2.0,
    xges_extended_search: bool = False,
    rcit_approx: str = "hbe",
    rcit_num_f: int = 10,
    rcit_num_f2: int = 5,
    random_state: int | None = 0,
) -> dict[str, Any]:
    """Learn a CPDAG; PC+RCIT is the default nonlinear estimator.

    Pass ``model_type='linear'`` explicitly to use XGES instead.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a non-empty pandas DataFrame")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("data must contain only finite numeric values")
    resolved_model_type = str(model_type).lower()
    if resolved_model_type not in {"linear", "nonlinear"}:
        raise ValueError("model_type must be 'linear' or 'nonlinear'")
    selected_alpha = independence_alpha(len(data)) if alpha is None else float(alpha)

    start = time.perf_counter()
    if resolved_model_type == "linear":
        cpdag, metadata = _learn_xges(
            data,
            bic_alpha=xges_bic_alpha,
            extended_search=xges_extended_search,
        )
    else:
        cpdag, metadata = _learn_pc_rcit(
            data,
            alpha=selected_alpha,
            rcit_approx=rcit_approx,
            rcit_num_f=rcit_num_f,
            rcit_num_f2=rcit_num_f2,
            random_state=random_state,
        )
    return {
        "cpdag": cpdag,
        "model_type": resolved_model_type,
        "runtime_sec": time.perf_counter() - start,
        **metadata,
    }

