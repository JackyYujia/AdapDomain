"""Fragmentation risk: penalize residuals dominated by a single domain after placement."""
from __future__ import annotations

from adapdomain.core.shape import DeviceResidual, TenantShape


def fragmentation_risk(device: DeviceResidual, replica: TenantShape) -> float:
    new_residual: dict[str, float] = {}
    for name, value in device.residual.items():
        usage = replica.domains.get(name, 0.0)
        new_residual[name] = max(0.0, value - usage)
    if not new_residual:
        return 0.0
    values = list(new_residual.values())
    total = sum(values)
    if total <= 1e-9:
        return 1.0
    biggest = max(values)
    # 0 when balanced, approaches 1 when residual is dominated by a single domain
    return max(0.0, min(1.0, (biggest / total) - 0.5) * 2.0)
