"""Request-level LLM serving metrics from timestamped runtime logs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RequestRecord:
    request_id: str
    policy: str
    trace: str
    seed: int
    arrival_ms: float
    first_token_ms: float | None
    completion_ms: float | None
    output_tokens: int
    status: str


@dataclass(frozen=True)
class RequestSummary:
    policy: str
    trace: str
    seed: int
    offered: int
    passed: int
    goodput_qps: float
    violation_pct: float
    ttft_p95_ms: float | None
    itl_p95_ms: float | None


def load_request_csv(path: str | Path) -> list[RequestRecord]:
    """Load one row per request; timestamps share a monotonic millisecond origin."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"request_id", "policy", "trace", "seed", "arrival_ms", "first_token_ms",
                "completion_ms", "output_tokens", "status"}
    if not rows:
        raise ValueError("request log has no rows")
    if missing := required - set(rows[0]):
        raise ValueError(f"request log is missing columns: {', '.join(sorted(missing))}")

    records: list[RequestRecord] = []
    seen: set[tuple[str, str, int, str]] = set()
    for row in rows:
        record = RequestRecord(
            request_id=row["request_id"].strip(),
            policy=row["policy"].strip(),
            trace=row["trace"].strip(),
            seed=int(row["seed"]),
            arrival_ms=float(row["arrival_ms"]),
            first_token_ms=_optional_float(row["first_token_ms"]),
            completion_ms=_optional_float(row["completion_ms"]),
            output_tokens=int(row["output_tokens"]),
            status=row["status"].strip().lower(),
        )
        key = (record.policy, record.trace, record.seed, record.request_id)
        if not record.request_id or key in seen:
            raise ValueError(f"missing or duplicate request id: {key}")
        if record.output_tokens < 0:
            raise ValueError(f"negative output_tokens: {key}")
        if record.status == "ok" and (record.first_token_ms is None or record.completion_ms is None
                                      or record.output_tokens < 1):
            raise ValueError(f"completed request has incomplete timing: {key}")
        if record.first_token_ms is not None and record.first_token_ms < record.arrival_ms:
            raise ValueError(f"first token precedes arrival: {key}")
        if (record.completion_ms is not None and record.first_token_ms is not None
                and record.completion_ms < record.first_token_ms):
            raise ValueError(f"completion precedes first token: {key}")
        seen.add(key)
        records.append(record)
    return records


def summarize_requests(
    records: list[RequestRecord], *, duration_s: float,
    ttft_slo_ms: float = 500.0, itl_slo_ms: float = 40.0,
) -> list[RequestSummary]:
    """Goodput counts requests meeting both TTFT and mean ITL SLOs."""
    if duration_s <= 0 or ttft_slo_ms <= 0 or itl_slo_ms <= 0:
        raise ValueError("duration and SLO thresholds must be positive")
    groups: dict[tuple[str, str, int], list[RequestRecord]] = {}
    for record in records:
        groups.setdefault((record.policy, record.trace, record.seed), []).append(record)

    summaries = []
    for (policy, trace, seed), group in sorted(groups.items()):
        ttfts: list[float] = []
        itls: list[float] = []
        passed = 0
        for record in group:
            if record.status != "ok":
                continue
            assert record.first_token_ms is not None and record.completion_ms is not None
            ttft = record.first_token_ms - record.arrival_ms
            # Mean ITL is the interval from first to last token over the remaining tokens.
            itl = (record.completion_ms - record.first_token_ms) / max(record.output_tokens - 1, 1)
            ttfts.append(ttft)
            itls.append(itl)
            if ttft <= ttft_slo_ms and itl <= itl_slo_ms:
                passed += 1
        summaries.append(RequestSummary(
            policy=policy, trace=trace, seed=seed, offered=len(group), passed=passed,
            goodput_qps=passed / duration_s,
            violation_pct=100.0 * (1.0 - passed / len(group)),
            ttft_p95_ms=_percentile(ttfts, 95), itl_p95_ms=_percentile(itls, 95),
        ))
    return summaries


def write_request_summaries(summaries: list[RequestSummary], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("policy", "trace", "seed", "offered", "passed", "goodput_qps",
                         "violation_pct", "ttft_p95_ms", "itl_p95_ms"))
        for item in summaries:
            writer.writerow((item.policy, item.trace, item.seed, item.offered, item.passed,
                             item.goodput_qps, item.violation_pct, item.ttft_p95_ms, item.itl_p95_ms))


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def _percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction
