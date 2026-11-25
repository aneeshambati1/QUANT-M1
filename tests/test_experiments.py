"""The declared search space must stay frozen and exclude the already-run cell."""
from __future__ import annotations

import itertools

from experiments import momentum_grid as mg
from quant_m1.research.baseline import UNIVERSE


def test_grid_excludes_already_run_cell():
    cells = set(itertools.product(mg.GRID_LOOKBACKS, mg.GRID_HORIZONS))
    assert mg.ALREADY_RUN not in cells


def test_grid_size_is_declared():
    cells = list(itertools.product(mg.GRID_LOOKBACKS, mg.GRID_HORIZONS))
    assert len(cells) == 12
    assert len(set(cells)) == 12


def test_grid_uses_baseline_universe():
    assert len(UNIVERSE) == 10
    assert mg.FAMILY == "momentum"