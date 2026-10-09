import pytest

from adapdomain.core.types import EPSILON_DOMAIN_FIT
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights, compute_score


def test_skewed_fit_scores_lower_than_balanced():
    weights = ScoreWeights(0.40, 0.0, 0.0, 0.0, 0.0, 0.0)
    risks = RiskTerms(x=0.0, h=0.0, q=0.0, f=0.0, u=0.0)
    balanced = compute_score(rho=[0.5, 0.5], weights=weights, risks=risks).final
    skewed = compute_score(rho=[0.9, 0.1], weights=weights, risks=risks).final
    assert balanced > skewed


def test_full_score_includes_subtractive_risks():
    weights = ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05)
    risks = RiskTerms(x=0.2, h=0.1, q=0.1, f=0.1, u=0.1)
    out = compute_score(rho=[0.5, 0.5], weights=weights, risks=risks)
    assert out.ceff > 0
    assert out.final < out.ceff
