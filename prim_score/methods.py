"""The five public methods included in the PRiM-score repository."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from .graphs import (
    directed_reachability,
    enumerate_consistent_dags,
    parents_from_dag,
    possible_parent_sets,
    topological_layers,
    validate_graph,
)
from .mechanisms import prim_score, squared_z_scores, validate_data


def _rank(scores: Mapping[str, float], top_k: int, method: str) -> dict[str, Any]:
    if not 1 <= top_k <= len(scores):
        raise ValueError("top_k must be between 1 and the number of variables")
    ranking = sorted(scores, key=lambda node: (-float(scores[node]), str(node)))
    numeric = {str(node): float(scores[node]) for node in scores}
    return {
        "method": method,
        "top_k": int(top_k),
        "predicted_roots": ranking[:top_k],
        "ranking": ranking,
        "scores": {node: numeric[node] for node in ranking},
    }


def _dag_scores(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
    dag: pd.DataFrame,
    *,
    mechanism: Literal["linear", "gp"],
    gp_max_samples: int | None,
    gp_random_state: int | None,
    gp_optimizer: str | None,
) -> dict[str, float]:
    parents = parents_from_dag(dag)
    return {
        str(node): prim_score(
            normal,
            anomalous,
            str(node),
            parents[str(node)],
            mechanism=mechanism,
            gp_max_samples=gp_max_samples,
            gp_random_state=gp_random_state,
            gp_optimizer=gp_optimizer,
        )
        for node in normal.columns
    }


def standard_prim(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
    cpdag: pd.DataFrame,
    top_k: int,
    *,
    max_dags: int = 100_000,
    mechanism: Literal["linear", "gp"] = "gp",
    gp_max_samples: int | None = None,
    gp_random_state: int | None = 0,
    gp_optimizer: str | None = None,
) -> dict[str, Any]:
    """Algorithm 1: PRiM-score with global MEC search."""
    validate_data(normal, anomalous)
    if top_k >= normal.shape[1]:
        raise ValueError("Algorithm 1 needs top_k < number of variables")
    candidates = enumerate_consistent_dags(cpdag, max_dags=max_dags)
    candidate_scores = [
        _dag_scores(
            normal,
            anomalous,
            dag,
            mechanism=mechanism,
            gp_max_samples=gp_max_samples,
            gp_random_state=gp_random_state,
            gp_optimizer=gp_optimizer,
        )
        for dag in candidates
    ]
    leakages = [sorted(scores.values(), reverse=True)[top_k] for scores in candidate_scores]
    selected = int(np.argmin(leakages))
    result = _rank(candidate_scores[selected], top_k, "PRiM-Score-Standard")
    result.update(
        {
            "num_candidate_dags": len(candidates),
            "selected_dag_index": selected,
            "selection_leakage": float(leakages[selected]),
            "selected_dag": candidates[selected].astype(int).to_dict(),
        }
    )
    return result


def local_prim(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
    cpdag: pd.DataFrame,
    top_k: int,
    *,
    mechanism: Literal["linear", "gp"] = "gp",
    gp_max_samples: int | None = None,
    gp_random_state: int | None = 0,
    gp_optimizer: str | None = None,
) -> dict[str, Any]:
    """Algorithm 2: local PRiM-score using local IDA."""
    validate_data(normal, anomalous)
    all_parent_sets: dict[str, list[list[str]]] = {}
    scores: dict[str, float] = {}
    selected_parent_sets: dict[str, list[str]] = {}
    for node in (str(column) for column in normal.columns):
        parent_sets = possible_parent_sets(cpdag, node)
        candidate_scores = [
            prim_score(
                normal,
                anomalous,
                node,
                parents,
                mechanism=mechanism,
                gp_max_samples=gp_max_samples,
                gp_random_state=gp_random_state,
                gp_optimizer=gp_optimizer,
            )
            for parents in parent_sets
        ]
        selected = int(np.argmin(candidate_scores))
        scores[node] = float(candidate_scores[selected])
        all_parent_sets[node] = parent_sets
        selected_parent_sets[node] = parent_sets[selected]
    result = _rank(scores, top_k, "PRiM-Score-Local")
    result.update(
        {
            "selected_parent_sets": selected_parent_sets,
            "possible_parent_sets": all_parent_sets,
        }
    )
    return result


def prim_notears(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
    notears_dag: pd.DataFrame,
    top_k: int,
    *,
    mechanism: Literal["linear", "gp"] = "gp",
    gp_max_samples: int | None = None,
    gp_random_state: int | None = 0,
    gp_optimizer: str | None = None,
) -> dict[str, Any]:
    """PRiM score evaluated on one NOTEARS DAG."""
    validate_data(normal, anomalous)
    scores = _dag_scores(
        normal,
        anomalous,
        notears_dag,
        mechanism=mechanism,
        gp_max_samples=gp_max_samples,
        gp_random_state=gp_random_state,
        gp_optimizer=gp_optimizer,
    )
    return _rank(scores, top_k, "PRiM-Score-NOTEARS")


def notears_2r(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
    notears_dag: pd.DataFrame,
    num_roots: int,
) -> dict[str, Any]:
    """NOTEARS-2R: top-2R squared-Z set ordered by NOTEARS layers."""
    validate_data(normal, anomalous)
    if not isinstance(num_roots, (int, np.integer)) or isinstance(num_roots, (bool, np.bool_)):
        raise TypeError("num_roots must be an integer")
    if not 1 <= int(num_roots) <= normal.shape[1]:
        raise ValueError("num_roots must be between 1 and the number of variables")
    adjacency = validate_graph(notears_dag, normal.columns, dag=True) != 0
    layers = topological_layers(adjacency)
    layer_by_node = {str(node): int(layers[i]) for i, node in enumerate(normal.columns)}
    scores = squared_z_scores(normal, anomalous)
    by_z = sorted(scores, key=lambda node: (-scores[node], node))
    count = min(2 * int(num_roots), len(by_z))
    selected = by_z[:count]
    selected_set = set(selected)
    ordered = sorted(selected, key=lambda node: (layer_by_node[node], -scores[node], node))
    ranking = ordered + [node for node in by_z if node not in selected_set]
    return {
        "method": "NOTEARS-2R",
        "num_roots": int(num_roots),
        "predicted_roots": ranking[:num_roots],
        "ranking": ranking,
        "scores": {node: scores[node] for node in ranking},
        "selected_top_2r": selected,
        "topological_layers": layer_by_node,
    }


def cpdag_2r(
    normal: pd.DataFrame,
    anomalous: pd.DataFrame,
    cpdag: pd.DataFrame,
    num_roots: int,
) -> dict[str, Any]:
    """CPDAG-2R: top-2R squared-Z set ordered by compelled ancestry."""
    validate_data(normal, anomalous)
    if not isinstance(num_roots, (int, np.integer)) or isinstance(num_roots, (bool, np.bool_)):
        raise TypeError("num_roots must be an integer")
    if not 1 <= int(num_roots) <= normal.shape[1]:
        raise ValueError("num_roots must be between 1 and the number of variables")
    reach = directed_reachability(cpdag)
    nodes = [str(node) for node in normal.columns]
    position = {node: i for i, node in enumerate(nodes)}
    scores = squared_z_scores(normal, anomalous)
    by_z = sorted(scores, key=lambda node: (-scores[node], node))
    count = min(2 * int(num_roots), len(by_z))
    selected = by_z[:count]
    selected_positions = [position[node] for node in selected]
    precedence = reach[np.ix_(selected_positions, selected_positions)].copy()
    indegree = precedence.sum(axis=0).astype(int)
    active = np.ones(count, dtype=bool)
    ordered: list[str] = []
    while len(ordered) < count:
        eligible = np.flatnonzero(active & (indegree == 0))
        if eligible.size == 0:
            raise ValueError("the compelled CPDAG precedence relation contains a cycle")
        chosen = min(
            (int(i) for i in eligible),
            key=lambda i: (-scores[selected[i]], selected[i]),
        )
        ordered.append(selected[chosen])
        active[chosen] = False
        indegree -= precedence[chosen].astype(int)
        indegree[chosen] = -1
    selected_set = set(selected)
    ranking = ordered + [node for node in by_z if node not in selected_set]
    return {
        "method": "CPDAG-2R",
        "num_roots": int(num_roots),
        "predicted_roots": ranking[:num_roots],
        "ranking": ranking,
        "scores": {node: scores[node] for node in ranking},
        "selected_top_2r": selected,
    }

