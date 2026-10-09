"""Tenant shape, device residual, normalization and complementarity primitives.

All formulas follow the paper:
    r_i^j = (x_i^j - x_min^j) / (x_sat^j - x_min^j)
    R_k^j = max(0, 1 - Σ r_i^j)
    rho   = min(r/(R+eps), R/(r+eps))
    COMP  = min(1, K / Σ(1/(rho_j + eps)))
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import DEFAULT_K, EPSILON_DOMAIN_FIT, DeviceId, FrontierVersion, ReplicaId


class FeasibleInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float
    x_sat: float

    @field_validator("x_sat")
    @classmethod
    def _check_order(cls, v: float, info) -> float:
        if v <= info.data.get("x_min", float("-inf")):
            raise ValueError("x_sat must be strictly greater than x_min")
        return v


class ShapeNormalizer:
    """Normalize raw counter values into [0, 1] using the feasible interval."""

    def normalize(self, value: float, interval: FeasibleInterval) -> float:
        norm = (value - interval.x_min) / (interval.x_sat - interval.x_min)
        if norm < 0.0:
            return 0.0
        if norm > 1.0:
            return 1.0
        return float(norm)


class TenantShape(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    replica_id: ReplicaId
    domains: dict[str, float]
    slo_ms: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    frontier_version: FrontierVersion
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("domains")
    @classmethod
    def _values_in_unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for name, x in v.items():
            if not (0.0 <= x <= 1.5):
                # We allow >1.0 here so that infeasible candidate evaluation
                # can still construct shapes; update_residual floors residuals.
                raise ValueError(f"domain {name}: {x} outside [0, 1.5]")
        return v


class DeviceResidual(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: DeviceId
    residual: dict[str, float]
    placed_replicas: list[ReplicaId]
    hbm_pressure: float = 0.0
    queue_pressure: float = 0.0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def empty(cls, device_id: int, domains: Iterable[str]) -> "DeviceResidual":
        return cls(
            device_id=DeviceId(device_id),
            residual={d: 1.0 for d in domains},
            placed_replicas=[],
        )


def update_residual(state: DeviceResidual, placed: Sequence[TenantShape]) -> DeviceResidual:
    """Apply `placed` tenants to the residual state and return a new DeviceResidual."""
    residual = dict(state.residual)
    replicas = list(state.placed_replicas)
    for tenant in placed:
        for name in residual:
            usage = tenant.domains.get(name, 0.0)
            residual[name] = max(0.0, residual[name] - usage)
        replicas.append(tenant.replica_id)
    return DeviceResidual(
        device_id=state.device_id,
        residual=residual,
        placed_replicas=replicas,
        hbm_pressure=state.hbm_pressure,
        queue_pressure=state.queue_pressure,
        updated_at=datetime.now(timezone.utc),
    )


def domain_fit_ratio(r: float, R: float, eps: float = EPSILON_DOMAIN_FIT) -> float:
    return min(r / (R + eps), R / (r + eps))


def harmonic_complementarity(
    rho_per_domain: Sequence[float],
    *,
    K: int | None = None,
    eps: float = EPSILON_DOMAIN_FIT,
) -> float:
    k = K if K is not None else len(rho_per_domain)
    if k <= 0:
        raise ValueError("K must be positive")
    inv_sum = sum(1.0 / (rho + eps) for rho in rho_per_domain)
    if inv_sum == 0:
        return 1.0
    return min(1.0, k / inv_sum)


def complementarity_2d(rho_v: float, rho_c: float, *, eps: float = EPSILON_DOMAIN_FIT) -> float:
    """Two-domain CEFF (paper Eq. ceff_2d):
        CEFF = 2 * rho_v * rho_c / (rho_v + rho_c + eps)
    """
    return (2.0 * rho_v * rho_c) / (rho_v + rho_c + eps)


__all__ = [
    "DEFAULT_K",
    "DeviceResidual",
    "FeasibleInterval",
    "ShapeNormalizer",
    "TenantShape",
    "complementarity_2d",
    "domain_fit_ratio",
    "harmonic_complementarity",
    "update_residual",
]
