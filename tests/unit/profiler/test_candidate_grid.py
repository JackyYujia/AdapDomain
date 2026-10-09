import pytest

from adapdomain.profiler.candidate_grid import CandidateGrid


def test_seed_grid_dimensions():
    g = CandidateGrid(seed_dims=(4, 4, 3), boundary_dims=(8, 8))
    seeds = list(g.seed_points())
    assert len(seeds) == 4 * 4 * 3
    assert all(0.0 <= s.v <= 1.0 and 0.0 <= s.m <= 1.0 and 0.0 <= s.h <= 1.0 for s in seeds)


def test_boundary_dims_validation():
    with pytest.raises(ValueError):
        CandidateGrid(seed_dims=(0, 0, 0), boundary_dims=(8, 8))


def test_boundary_refinement_grid():
    g = CandidateGrid(seed_dims=(4, 4, 3), boundary_dims=(8, 8))
    boundary = list(g.boundary_points(hbm=0.5))
    assert len(boundary) == 8 * 8
