import math
import pytest

from adapdomain.core.domain import DomainSpec
from adapdomain.core.shape import (
    DeviceResidual,
    FeasibleInterval,
    ShapeNormalizer,
    TenantShape,
    complementarity_2d,
    domain_fit_ratio,
    harmonic_complementarity,
    update_residual,
)


def test_domain_spec_validation():
    DomainSpec(name="cube", capacity=20, unit="logical_units", schedulable=True, telemetry_key="aic_mac_ratio")
    with pytest.raises(ValueError):
        DomainSpec(name="cube", capacity=0, unit="x", schedulable=True, telemetry_key="k")


def test_shape_normalizer_paper_formula():
    interval = FeasibleInterval(x_min=2.0, x_sat=10.0)
    norm = ShapeNormalizer().normalize(value=6.0, interval=interval)
    assert norm == pytest.approx(0.5)


def test_shape_normalizer_clamps_and_rejects_invalid_interval():
    norm = ShapeNormalizer()
    interval = FeasibleInterval(x_min=1.0, x_sat=2.0)
    assert norm.normalize(0.5, interval) == pytest.approx(0.0)
    assert norm.normalize(2.5, interval) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        FeasibleInterval(x_min=5.0, x_sat=5.0)


def test_tenant_shape_and_device_residual_roundtrip():
    tenant = TenantShape(
        model_name="bert-base",
        replica_id="bert-base#0",
        domains={"cube": 0.40, "vector": 0.20},
        slo_ms=15.0,
        batch_size=1,
        frontier_version=1,
        confidence=0.9,
    )
    residual = DeviceResidual.empty(device_id=0, domains=["cube", "vector"])
    new_residual = update_residual(residual, [tenant])
    assert new_residual.residual["cube"] == pytest.approx(0.60)
    assert new_residual.residual["vector"] == pytest.approx(0.80)
    assert new_residual.placed_replicas == ["bert-base#0"]


def test_residual_floors_at_zero():
    residual = DeviceResidual.empty(device_id=0, domains=["cube", "vector"])
    heavy = TenantShape(
        model_name="x", replica_id="x#0",
        domains={"cube": 1.2, "vector": 0.0},
        slo_ms=1.0, batch_size=1, frontier_version=1, confidence=1.0,
    )
    assert update_residual(residual, [heavy]).residual["cube"] == pytest.approx(0.0)


def test_domain_fit_ratio_symmetric_and_bounded():
    rho = domain_fit_ratio(r=0.4, R=0.4, eps=1e-6)
    assert rho == pytest.approx(1.0)
    rho_low = domain_fit_ratio(r=0.9, R=0.1, eps=1e-6)
    assert 0 < rho_low < 1


def test_harmonic_complementarity_penalises_one_sided_fit():
    balanced = harmonic_complementarity([0.5, 0.5])
    skewed = harmonic_complementarity([0.9, 0.1])
    assert balanced > skewed
    assert balanced <= 1.0
    assert skewed > 0.0


def test_complementarity_2d_matches_harmonic_form():
    rho_v, rho_c = 0.6, 0.3
    expected = (2 * rho_v * rho_c) / (rho_v + rho_c + 1e-6)
    assert math.isclose(complementarity_2d(rho_v, rho_c, eps=1e-6), expected, rel_tol=1e-9)
