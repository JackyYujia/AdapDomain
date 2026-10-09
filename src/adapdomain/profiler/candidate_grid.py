"""Sparse seed and dense V:M boundary grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class GridPoint:
    v: float
    m: float
    h: float


class CandidateGrid:
    def __init__(self, *, seed_dims: tuple[int, int, int], boundary_dims: tuple[int, int]) -> None:
        if min(seed_dims) <= 0 or min(boundary_dims) <= 0:
            raise ValueError("grid dimensions must be positive")
        self._seed_dims = seed_dims
        self._boundary_dims = boundary_dims

    def seed_points(self) -> Iterator[GridPoint]:
        nv, nm, nh = self._seed_dims
        for iv in range(nv):
            for im in range(nm):
                for ih in range(nh):
                    yield GridPoint(
                        v=iv / max(nv - 1, 1),
                        m=im / max(nm - 1, 1),
                        h=ih / max(nh - 1, 1),
                    )

    def boundary_points(self, hbm: float) -> Iterator[GridPoint]:
        nv, nm = self._boundary_dims
        for iv in range(nv):
            for im in range(nm):
                yield GridPoint(
                    v=iv / max(nv - 1, 1),
                    m=im / max(nm - 1, 1),
                    h=hbm,
                )
