"""Typed config models loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return yaml.safe_load(p.read_text()) or {}


_BaseT = TypeVar("_BaseT", bound="_Base")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls: type[_BaseT], path: str | Path) -> _BaseT:
        return cls.model_validate(load_yaml(path))


class DomainConfig(_Base):
    capacity: float
    unit: str
    telemetry_key: str
    schedulable: bool


class SetDeviceResLimitConfig(_Base):
    cube_limit_type: int | None = None
    vector_limit_type: int | None = None
    hbm_limit_type: int | None = None


class AclConfig(_Base):
    init_config_path: str | None = None
    set_device_res_limit: SetDeviceResLimitConfig = Field(default_factory=SetDeviceResLimitConfig)


class MsprofConfig(_Base):
    binary: str = "msprof"
    default_metrics: list[str] = Field(default_factory=list)
    output_root: str = "data/raw/msprof"

class HardwareConfig(_Base):
    platform: str
    device_ids: list[int]
    domains: dict[str, DomainConfig]
    acl: AclConfig = Field(default_factory=AclConfig)
    msprof: MsprofConfig = Field(default_factory=MsprofConfig)

    @field_validator("device_ids")
    @classmethod
    def _unique_ids(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("device_ids must be unique")
        return v


class HarmonicFitWeights(_Base):
    lambda_c: float
    lambda_x: float
    lambda_h: float
    lambda_q: float
    lambda_f: float
    lambda_u: float


class HarmonicFitInterference(_Base):
    eta: float


class HarmonicFitConfig(_Base):
    weights: HarmonicFitWeights
    interference: HarmonicFitInterference
    epsilon_domain_fit: float
    k_domains: list[str]
    control_budget_ms: float


class ActiveVCConfig(_Base):
    budget: int
    alpha: float
    beta: float
    gamma: float
    tau: float
    seed_grid: list[int]
    boundary_grid: list[int]
    hbm_strategy: str

class RuntimeEWMA(_Base):
    alpha: float


class RuntimeSmoothGuardThresholds(_Base):
    p95_drift: float
    residual_drift: float
    hbm_high: float
    queue_high: float


class RuntimeSmoothGuard(_Base):
    consecutive_trigger_count: int
    cooldown_seconds: float
    thresholds: RuntimeSmoothGuardThresholds


class RuntimePreScale(_Base):
    forecast_horizon_s: float


class RuntimeNightShift(_Base):
    migration_budget: int
    low_load_windows: int


class RuntimeConfig(_Base):
    ewma: RuntimeEWMA
    smooth_guard: RuntimeSmoothGuard
    prescale: RuntimePreScale
    nightshift: RuntimeNightShift


class ExperimentReplayConfig(_Base):
    seeds: list[int]
    windows: int
    arrival_rates: list[float]
    slo_thresholds_ms: list[float] | None = None
    output_root: str


class ExperimentAblationConfig(_Base):
    seeds: list[int]
    windows: int
    harmonicfit_variants: list[str]
    activevc_baselines: list[str]
    budgets: list[int]
    output_root: str
