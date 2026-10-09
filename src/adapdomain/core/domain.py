"""DomainSpec for K-domain accelerator resources."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    capacity: float = Field(gt=0)
    unit: str = Field(min_length=1)
    schedulable: bool
    telemetry_key: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        return v.strip().lower()
