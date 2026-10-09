"""Deterministic seed iteration for replay/ablation."""
from __future__ import annotations

import random
from typing import Iterator


def seeded_iter(seeds: list[int]) -> Iterator[tuple[int, random.Random]]:
    for seed in seeds:
        rng = random.Random(seed)
        yield seed, rng
