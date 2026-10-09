"""HarmonicFit edge construction helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from adapdomain.core.shape import DeviceResidual, TenantShape, domain_fit_ratio
from adapdomain.core.types import EPSILON_DOMAIN_FIT


@dataclass(frozen=True)
class EdgeRho:
    rho_per_domain: list[float]
    fit_ratios_by_domain: dict[str, float]


def edge_rho(replica: TenantShape, device: DeviceResidual, *, domains: Sequence[str], eps: float = EPSILON_DOMAIN_FIT) -> EdgeRho:
    rhos: list[float] = []
    by_domain: dict[str, float] = {}
    for name in domains:
        r = replica.domains.get(name, 0.0)
        R = device.residual.get(name, 0.0)
        val = domain_fit_ratio(r=r, R=R, eps=eps)
        rhos.append(val)
        by_domain[name] = val
    return EdgeRho(rho_per_domain=rhos, fit_ratios_by_domain=by_domain)
