import pytest

from adapdomain.core.workload import (
    Replica,
    Request,
    WorkloadTrace,
)


def test_request_validates_arrival_and_deadline():
    r = Request(request_id="r0", model="bert-base", arrival_ms=0.0, deadline_ms=20.0)
    assert r.deadline_ms > r.arrival_ms
    with pytest.raises(Exception):
        Request(request_id="x", model="m", arrival_ms=5.0, deadline_ms=4.0)


def test_workload_trace_jsonl_roundtrip(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = WorkloadTrace(
        replicas=[Replica(replica_id="bert-base#0", model="bert-base", slo_ms=15.0, batch_size=1)],
        requests=[
            Request(request_id="r0", model="bert-base", arrival_ms=0.0, deadline_ms=15.0),
            Request(request_id="r1", model="bert-base", arrival_ms=1.0, deadline_ms=16.0),
        ],
    )
    trace.to_jsonl(path)
    loaded = WorkloadTrace.from_jsonl(path)
    assert len(loaded.requests) == 2
    assert loaded.replicas[0].model == "bert-base"


def test_slice_returns_requests_in_window():
    trace = WorkloadTrace(
        replicas=[],
        requests=[
            Request(request_id="r0", model="m", arrival_ms=0.0, deadline_ms=10.0),
            Request(request_id="r1", model="m", arrival_ms=5.0, deadline_ms=15.0),
            Request(request_id="r2", model="m", arrival_ms=12.0, deadline_ms=22.0),
        ],
    )
    sub = trace.slice(start_ms=1.0, end_ms=10.0)
    assert [r.request_id for r in sub.requests] == ["r1"]
