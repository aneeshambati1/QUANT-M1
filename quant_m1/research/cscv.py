"""
Combinatorial symmetric cross-validation splits with purge/embargo.

No greedy path builder.
"""
from __future__ import annotations

import itertools
from typing import Iterator, Tuple

import numpy as np


def combinatorial_splits(n_groups: int, n_test: int) -> Iterator[Tuple[int, ...]]:
    if not (1 <= n_test < n_groups):
        raise ValueError("need 1 <= n_test < n_groups")
    return itertools.combinations(range(n_groups), n_test)


def purge_embargo(
    train_idx,
    test_idx,
    label_horizon: int,
    embargo: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove training labels overlapping the test block, plus embargo after test.
    label_horizon: label at i uses [i, i + label_horizon].
    """
    train = set(int(i) for i in train_idx)
    test = np.asarray(sorted(int(i) for i in test_idx), dtype=int)
    if len(test) == 0:
        return np.asarray(sorted(train), dtype=int), test
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon/embargo >= 0")

    tmin, tmax = int(test[0]), int(test[-1])
    kept = []
    for i in sorted(train):
        if tmin <= i <= tmax:
            continue
        if i < tmin and i + label_horizon >= tmin:
            continue
        if tmax < i <= tmax + embargo:
            continue
        kept.append(i)
    return np.asarray(kept, dtype=int), test


def cscv_splits(
    n_obs: int,
    n_groups: int,
    n_test: int,
    label_horizon: int = 1,
    embargo: int = 0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if n_obs < n_groups:
        raise ValueError("n_obs >= n_groups")
    boundaries = np.linspace(0, n_obs, n_groups + 1, dtype=int)
    groups = [np.arange(boundaries[i], boundaries[i + 1]) for i in range(n_groups)]

    for test_groups in combinatorial_splits(n_groups, n_test):
        test_idx = np.concatenate([groups[g] for g in test_groups])
        train_idx = np.concatenate([groups[g] for g in range(n_groups) if g not in test_groups])
        train_idx, test_idx = purge_embargo(train_idx, test_idx, label_horizon, embargo)
        yield train_idx, test_idx
