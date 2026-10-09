"""ActiveVC orchestration over a model registry. Uses an injected `evaluate` callable
so this module remains independent of the ACL backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from adapdomain.profiler.activevc import ActiveVC, ActiveVCConfigRuntime
from adapdomain.profiler.acquisition import AcquisitionWeights
from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.storage.frontier_store import FrontierStore


@dataclass(frozen=True)
class ModelSpec:
    name: str
    om_path: str
    slo_ms: float
    batch_size: int


def profile_all(
    *,
    models: list[ModelSpec],
    budget: int,
    weights: AcquisitionWeights,
    evaluate_factory: Callable[[ModelSpec], Callable[[GridPoint], MsprofObservation]],
    store: FrontierStore,
) -> Mapping[str, str]:
    cfg = ActiveVCConfigRuntime(budget=budget, weights=weights)
    builder = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0))
    versions: dict[str, str] = {}
    for spec in models:
        result = ActiveVC(config=cfg, builder=builder).run(
            model_name=spec.name, version=1, source_run_id=f"{spec.name}-v1",
            evaluate=evaluate_factory(spec),
        )
        path = store.save(result.snapshot)
        versions[spec.name] = str(path)
    return versions
