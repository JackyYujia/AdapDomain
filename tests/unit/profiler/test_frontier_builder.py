import pytest

from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation


def test_builder_builds_snapshot_from_observations():
    obs = [
        MsprofObservation(model="bert-base", point_id="a",
                          raw_cube=0.40, raw_vector=0.20, raw_hbm=0.40,
                          batch_size=1, slo_ms=15.0,
                          p50_ms=4.0, p95_ms=6.0, p99_ms=7.0, goodput_qps=100.0,
                          feasible=True),
        MsprofObservation(model="bert-base", point_id="b",
                          raw_cube=0.60, raw_vector=0.25, raw_hbm=0.55,
                          batch_size=1, slo_ms=15.0,
                          p50_ms=5.0, p95_ms=9.0, p99_ms=10.0, goodput_qps=130.0,
                          feasible=True),
    ]
    snap = FrontierBuilder(
        cube_range=(0.0, 1.0), vector_range=(0.0, 1.0),
    ).build(model_name="bert-base", version=1, observations=obs, source_run_id="r0")
    assert snap.dominant_domain in {"matrix", "vector", "balanced"}
    assert snap.safe_point().point_id == "a"
    assert all(0.0 <= p.norm_cube <= 1.0 for p in snap.points)


def test_normalization_clamps_above_one():
    obs = [MsprofObservation(model="m", point_id="x",
                             raw_cube=1.2, raw_vector=0.5, raw_hbm=0.5,
                             batch_size=1, slo_ms=10.0,
                             p50_ms=5, p95_ms=8, p99_ms=9, goodput_qps=80,
                             feasible=True)]
    snap = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0)).build(
        model_name="m", version=1, observations=obs, source_run_id="r",
    )
    assert snap.points[0].norm_cube == pytest.approx(1.0)
