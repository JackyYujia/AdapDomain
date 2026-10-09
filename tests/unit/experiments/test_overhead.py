from adapdomain.experiments.overhead import OverheadResult, measure


def test_measure_returns_ordered_metrics():
    def noop() -> None:
        sum(range(1000))
    res: OverheadResult = measure(noop, size=10, repeats=10)
    assert res.size == 10
    assert res.mean_ms >= 0
    assert res.p95_ms >= res.mean_ms
    assert res.max_ms >= res.p95_ms
