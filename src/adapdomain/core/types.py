"""Common type aliases and numeric constants."""
from __future__ import annotations

from typing import NewType

EPSILON_DOMAIN_FIT: float = 1e-6
DEFAULT_K: int = 2

ModelId = NewType("ModelId", str)
DeviceId = NewType("DeviceId", int)
ReplicaId = NewType("ReplicaId", str)
RunId = NewType("RunId", str)
Seed = NewType("Seed", int)
FrontierVersion = NewType("FrontierVersion", int)
