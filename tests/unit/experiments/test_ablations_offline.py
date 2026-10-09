import random

from adapdomain.experiments.ablations import (
    bo_baseline, binary_baseline, boundary_error, harmonicfit_variant_score, random_baseline,
)
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.profiler.frontier_builder import MsprofObservation
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights


def _evaluate(point: GridPoint) -> MsprofObservation:
    feasible = point.v + point.m <= 1.2
    return MsprofObservation(
        model="m", point_id=f"v{point.v:.2f}-m{point.m:.2f}",
        raw_cube=point.m, raw_vector=point.v, raw_hbm=point.h,
        batch_size=1, slo_ms=20.0,
        p50_ms=4, p95_ms=6, p99_ms=7, goodput_qps=100 if feasible else 5, feasible=feasible,
    )


def test_baselines_produce_observations():
    rng = random.Random(0)
    assert len(random_baseline(budget=8, evaluate=_evaluate, rng=rng)) == 8
    assert len(binary_baseline(budget=9, evaluate=_evaluate)) == 9
    assert len(bo_baseline(budget=8, evaluate=_evaluate, rng=rng)) == 8


def test_variant_score_removes_specific_terms():
    w = ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05)
    r = RiskTerms(x=0.2, h=0.2, q=0.2, f=0.2, u=0.2)
    full = harmonicfit_variant_score(variant="full", rho=[0.5, 0.5], risks=r, weights=w)
    no_x = harmonicfit_variant_score(variant="no_interference", rho=[0.5, 0.5], risks=r, weights=w)
    assert no_x > full  # removing the X penalty raises the score
