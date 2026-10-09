import math

from adapdomain.profiler.acquisition import AcquisitionWeights, acquisition
from adapdomain.profiler.candidate_grid import GridPoint


def test_acquisition_paper_form():
    weights = AcquisitionWeights(alpha=0.40, beta=0.40, gamma=0.20, tau=0.15)
    s = acquisition(
        candidate=GridPoint(v=0.5, m=0.5, h=0.5),
        uncertainty=0.6,
        distance_to_frontier=0.10,
        marginal_goodput_gain=0.30,
        weights=weights,
    )
    expected = 0.40 * 0.6 + 0.40 * math.exp(-0.10 / 0.15) + 0.20 * 0.30
    assert s == math.isclose(s, expected, rel_tol=1e-9) or abs(s - expected) < 1e-9
