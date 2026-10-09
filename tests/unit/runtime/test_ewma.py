import pytest

from adapdomain.runtime.ewma import Ewma


def test_ewma_converges_to_steady():
    e = Ewma(alpha=0.5)
    for _ in range(20):
        e.observe(10.0)
    assert e.value == pytest.approx(10.0, abs=1e-3)


def test_ewma_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        Ewma(alpha=0)
    with pytest.raises(ValueError):
        Ewma(alpha=1.5)
