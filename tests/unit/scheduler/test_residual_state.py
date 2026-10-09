import pytest

from adapdomain.core.placement import PlacementPlan, ScoreBreakdown
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.residual_state import ClusterState


def _shape(rid: str, cube: float, vec: float) -> TenantShape:
    return TenantShape(model_name="m", replica_id=rid,
                       domains={"cube": cube, "vector": vec},
                       slo_ms=15.0, batch_size=1,
                       frontier_version=1, confidence=0.9)


def _residual(did: int) -> DeviceResidual:
    return DeviceResidual.empty(device_id=did, domains=["cube", "vector"])


def test_apply_placement_consumes_residual():
    state = ClusterState.from_residuals([_residual(0), _residual(1)])
    state.apply_placement(replica=_shape("r0", cube=0.3, vec=0.2), device_id=0)
    assert state.residual_of(0).residual["cube"] == pytest.approx(0.7)
    assert state.residual_of(1).residual["cube"] == 1.0


def test_revert_restores_residual():
    state = ClusterState.from_residuals([_residual(0)])
    state.apply_placement(replica=_shape("r0", cube=0.4, vec=0.2), device_id=0)
    state.revert_placement(replica_id="r0", device_id=0)
    assert state.residual_of(0).residual["cube"] == 1.0
    assert "r0" not in state.residual_of(0).placed_replicas
