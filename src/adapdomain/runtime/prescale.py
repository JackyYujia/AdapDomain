"""PreScale: select device to prewarm based on scarce-domain residual."""
from __future__ import annotations

from typing import Sequence

from adapdomain.core.shape import DeviceResidual


class PreScale:
    def pick_for(self, *, scarce_domain: str, residuals: Sequence[DeviceResidual]) -> int | None:
        best: tuple[int, float] | None = None
        for r in residuals:
            value = r.residual.get(scarce_domain, 0.0)
            if best is None or value > best[1]:
                best = (r.device_id, value)
        return best[0] if best else None
