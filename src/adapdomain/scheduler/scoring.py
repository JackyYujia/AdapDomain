"""HarmonicFit edge scoring (paper Eq. harmonicfit_score)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from adapdomain.core.placement import ScoreBreakdown
from adapdomain.core.shape import harmonic_complementarity
from adapdomain.core.types import EPSILON_DOMAIN_FIT


@dataclass(frozen=True)
class ScoreWeights:
    lambda_c: float
    lambda_x: float
    lambda_h: float
    lambda_q: float
    lambda_f: float
    lambda_u: float


@dataclass(frozen=True)
class RiskTerms:
    x: float
    h: float
    q: float
    f: float
    u: float


def compute_score(*, rho: Sequence[float], weights: ScoreWeights, risks: RiskTerms,
                  eps: float = EPSILON_DOMAIN_FIT) -> ScoreBreakdown:
    ceff = harmonic_complementarity(list(rho), eps=eps)
    final = (
        weights.lambda_c * ceff
        - weights.lambda_x * risks.x
        - weights.lambda_h * risks.h
        - weights.lambda_q * risks.q
        - weights.lambda_f * risks.f
        - weights.lambda_u * risks.u
    )
    return ScoreBreakdown(ceff=ceff, x=risks.x, h=risks.h, q=risks.q, f=risks.f, u=risks.u, final=final)
