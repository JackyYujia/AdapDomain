from datetime import datetime, timezone

import pytest

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot
from adapdomain.storage.frontier_store import FrontierStore


def _snap(version: int) -> FrontierSnapshot:
    pt = FrontierPoint(
        model_name="bert-base", point_id=f"p{version}",
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=0.3, norm_vector=0.3,
        batch_size=1, slo_ms=15.0,
        p50_ms=2, p95_ms=4, p99_ms=5, goodput_qps=80.0,
        feasible=True, confidence=0.9,
    )
    return FrontierSnapshot(
        model_name="bert-base", version=version, k_domains=["cube", "vector"],
        points=[pt], safe_point_id=pt.point_id, dominant_domain="balanced",
        confidence_summary=0.9, source_run_id="r",
        created_at=datetime.now(timezone.utc),
    )


def test_store_write_atomic_and_read_latest(tmp_path):
    store = FrontierStore(tmp_path)
    store.save(_snap(1))
    store.save(_snap(2))
    latest = store.load_latest("bert-base")
    assert latest.version == 2
    assert store.load_version("bert-base", 1).version == 1
