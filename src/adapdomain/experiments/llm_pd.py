"""Offline LLM prefill/decode deployment experiment.

This module is a runnable harness for the paper's LLM extension. It can run
today with a deterministic built-in LLaMA-7B-like phase profile, and it can also
consume measured CSVs once Ascend runs are available.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


POLICIES = (
    "no_partition",
    "static_pd_colocation",
    "pd_separate",
    "pd_dynamic_partition",
)


@dataclass(frozen=True)
class PhaseProfile:
    phase: str
    cube: float
    vector: float
    hbm: float
    service_ms: float
    slo_ms: float
    kv_cache_gb: float
    batch_size: int
    sampling_policy: str = "greedy"

    @property
    def v_to_c(self) -> float:
        return self.vector / max(self.cube, 1e-9)


@dataclass(frozen=True)
class PhaseRequest:
    phase: str
    arrival_ms: float
    count: int = 1


@dataclass(frozen=True)
class LlmPdInputs:
    profiles: dict[str, PhaseProfile]
    trace: list[PhaseRequest]
    num_cards: int = 4
    replicas_per_card: int = 2
    cube_capacity: float = 1.0
    vector_capacity: float = 1.0
    hbm_capacity: float = 1.0


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    goodput_qps: float
    violation_pct: float
    avg_latency_ms: float
    p95_latency_ms: float
    accepted: int
    total: int
    placement: dict[str, list[str]]


def default_profiles() -> dict[str, PhaseProfile]:
    """LLaMA-7B-like default profiles for local smoke runs.

    These values are deliberately illustrative. Replace them with measured CSV
    rows before reporting paper numbers.
    """
    return {
        "prefill": PhaseProfile(
            phase="prefill",
            cube=0.62,
            vector=0.22,
            hbm=0.42,
            service_ms=32.0,
            slo_ms=85.0,
            kv_cache_gb=3.0,
            batch_size=8,
            sampling_policy="none",
        ),
        "decode": PhaseProfile(
            phase="decode",
            cube=0.24,
            vector=0.46,
            hbm=0.55,
            service_ms=18.0,
            slo_ms=55.0,
            kv_cache_gb=6.0,
            batch_size=1,
            sampling_policy="top_p",
        ),
    }


def bursty_trace(
    *,
    windows: int = 24,
    window_ms: float = 1000.0,
    prefill_base: int = 10,
    decode_base: int = 18,
    burst_every: int = 4,
    burst_multiplier: float = 2.2,
) -> list[PhaseRequest]:
    trace: list[PhaseRequest] = []
    for w in range(windows):
        mult = burst_multiplier if w % burst_every == 0 else 1.0
        # Alternate burst dominance to mimic prompt spikes followed by decode pressure.
        prefill_mult = mult if (w // burst_every) % 2 == 0 else 1.0
        decode_mult = mult if (w // burst_every) % 2 == 1 else 1.0
        trace.append(PhaseRequest("prefill", arrival_ms=w * window_ms, count=max(1, round(prefill_base * prefill_mult))))
        trace.append(PhaseRequest("decode", arrival_ms=w * window_ms + window_ms * 0.35, count=max(1, round(decode_base * decode_mult))))
    return trace


def load_profiles_csv(path: str | Path, *, precision: str | None = None) -> dict[str, PhaseProfile]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} has no rows")
    precisions = {row["precision"].strip().upper() for row in rows if row.get("precision")}
    if precision is not None:
        if not precisions:
            raise ValueError("profile CSV has no precision column")
        rows = [row for row in rows if row.get("precision", "").strip().upper() == precision.upper()]
        if not rows:
            raise ValueError(f"no profiles for precision {precision}")
    elif len(precisions) > 1:
        raise ValueError("profile CSV mixes precisions; select one with --precision")
    profiles: dict[str, list[PhaseProfile]] = {"prefill": [], "decode": []}
    operating_points: dict[str, set[tuple[str, str, str]]] = {"prefill": set(), "decode": set()}
    for row in rows:
        phase = row["phase"].strip().lower()
        if phase not in profiles:
            continue
        operating_points[phase].add((row.get("batch_size", ""), row.get("prompt_tokens", ""),
                                     row.get("kv_cache_tokens", "")))
        profiles[phase].append(
            PhaseProfile(
                phase=phase,
                cube=_float(row, "cube_util"),
                vector=_float(row, "vector_util"),
                hbm=_float(row, "hbm_bw_util"),
                service_ms=_float(row, "p95_ms"),
                slo_ms=_float(row, "slo_ms"),
                kv_cache_gb=_float(row, "kv_cache_gb", default=0.0),
                batch_size=int(_float(row, "batch_size", default=1.0)),
                sampling_policy=row.get("sampling_policy", "unknown"),
            )
        )
    out: dict[str, PhaseProfile] = {}
    for phase, values in profiles.items():
        if not values:
            continue
        if len(operating_points[phase]) > 1:
            raise ValueError(f"{path} mixes {phase} operating points; filter the CSV to one setting per phase")
        out[phase] = PhaseProfile(
            phase=phase,
            cube=mean(v.cube for v in values),
            vector=mean(v.vector for v in values),
            hbm=mean(v.hbm for v in values),
            service_ms=mean(v.service_ms for v in values),
            slo_ms=mean(v.slo_ms for v in values),
            kv_cache_gb=mean(v.kv_cache_gb for v in values),
            batch_size=round(mean(v.batch_size for v in values)),
            sampling_policy=values[0].sampling_policy,
        )
    if {"prefill", "decode"} - set(out):
        raise ValueError(f"{path} must include both prefill and decode rows")
    return out


def load_trace_csv(path: str | Path) -> list[PhaseRequest]:
    rows = list(csv.DictReader(Path(path).open(newline="")))
    trace: list[PhaseRequest] = []
    for row in rows:
        phase = row["phase"].strip().lower()
        if phase not in {"prefill", "decode"}:
            continue
        trace.append(
            PhaseRequest(
                phase=phase,
                arrival_ms=_float(row, "arrival_ms"),
                count=max(1, int(_float(row, "count", default=1.0))),
            )
        )
    if not trace:
        raise ValueError(f"{path} has no usable prefill/decode requests")
    return sorted(trace, key=lambda r: r.arrival_ms)


def run_llm_pd_experiment(inputs: LlmPdInputs) -> list[PolicyResult]:
    return [_simulate_policy(policy, inputs) for policy in POLICIES]


def write_results(results: Iterable[PolicyResult], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in results]
    with (out / "llm_pd_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "policy",
                "goodput_qps",
                "violation_pct",
                "avg_latency_ms",
                "p95_latency_ms",
                "accepted",
                "total",
                "placement_json",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "policy": r.policy,
                "goodput_qps": f"{r.goodput_qps:.6f}",
                "violation_pct": f"{r.violation_pct:.6f}",
                "avg_latency_ms": f"{r.avg_latency_ms:.6f}",
                "p95_latency_ms": f"{r.p95_latency_ms:.6f}",
                "accepted": r.accepted,
                "total": r.total,
                "placement_json": json.dumps(r.placement, sort_keys=True),
            })
    with (out / "llm_pd_summary.md").open("w", encoding="utf-8") as f:
        f.write("| Policy | Goodput (qps) | Violation (%) | p95 (ms) | Accepted/Total |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r.policy} | {r.goodput_qps:.2f} | {r.violation_pct:.2f} | {r.p95_latency_ms:.2f} | {r.accepted}/{r.total} |\n")


def _simulate_policy(policy: str, inputs: LlmPdInputs) -> PolicyResult:
    slots = _build_slots(policy, inputs)
    latencies: list[float] = []
    accepted = 0
    total = 0
    finish_times = {idx: 0.0 for idx in range(len(slots))}

    for req in inputs.trace:
        profile = inputs.profiles[req.phase]
        candidates = [idx for idx, slot in enumerate(slots) if req.phase in slot["phases"]]
        if not candidates:
            total += req.count
            latencies.extend([math.inf] * req.count)
            continue
        for _ in range(req.count):
            total += 1
            idx = min(candidates, key=lambda c: finish_times[c])
            slot = slots[idx]
            service = _service_time(profile, slot)
            start = max(req.arrival_ms, finish_times[idx])
            finish = start + service
            finish_times[idx] = finish
            latency = finish - req.arrival_ms
            latencies.append(latency)
            if latency <= profile.slo_ms:
                accepted += 1

    finite = [x for x in latencies if math.isfinite(x)]
    duration_ms = max([r.arrival_ms for r in inputs.trace], default=0.0) + max((p.slo_ms for p in inputs.profiles.values()), default=1.0)
    goodput = accepted / max(duration_ms / 1000.0, 1e-9)
    violation = 100.0 * (1.0 - accepted / max(total, 1))
    return PolicyResult(
        policy=policy,
        goodput_qps=goodput,
        violation_pct=violation,
        avg_latency_ms=mean(finite) if finite else math.inf,
        p95_latency_ms=_percentile(finite, 95.0) if finite else math.inf,
        accepted=accepted,
        total=total,
        placement=_placement_summary(slots),
    )


def _build_slots(policy: str, inputs: LlmPdInputs) -> list[dict[str, object]]:
    profiles = inputs.profiles
    slots: list[dict[str, object]] = []
    if policy == "no_partition":
        # Whole-card LLM replicas, phase-oblivious. No core split and no PD specialization.
        for card in range(inputs.num_cards):
            slots.append({
                "card": card,
                "phases": {"prefill", "decode"},
                "cube_share": 1.0,
                "vector_share": 1.0,
                "hbm_share": 1.0,
                "mode": "whole_card",
            })
    elif policy == "static_pd_colocation":
        # Static PD mixed on every card. Prefill and decode are colocated, but
        # each phase receives a fixed half-card partition.
        for card in range(inputs.num_cards):
            slots.append({
                "card": card,
                "phases": {"prefill"},
                "cube_share": 0.50,
                "vector_share": 0.50,
                "hbm_share": 0.50,
                "mode": "static_prefill_mixed",
            })
            slots.append({
                "card": card,
                "phases": {"decode"},
                "cube_share": 0.50,
                "vector_share": 0.50,
                "hbm_share": 0.50,
                "mode": "static_decode_mixed",
            })
    elif policy == "pd_separate":
        # Half the cards are prefill-only and half decode-only.
        split = max(1, inputs.num_cards // 2)
        for card in range(inputs.num_cards):
            phase = "prefill" if card < split else "decode"
            slots.append({
                "card": card,
                "phases": {phase},
                "cube_share": 1.0,
                "vector_share": 1.0,
                "hbm_share": 1.0,
                "mode": f"{phase}_separate",
            })
    elif policy == "pd_dynamic_partition":
        # Phase-aware partitioning on each card. Shares come from normalized complementary demand.
        prefill = profiles["prefill"]
        decode = profiles["decode"]
        total_cube = prefill.cube + decode.cube
        total_vector = prefill.vector + decode.vector
        prefill_cube = _clamp(prefill.cube / max(total_cube, 1e-9), 0.35, 0.75)
        prefill_vector = _clamp(prefill.vector / max(total_vector, 1e-9), 0.20, 0.55)
        for card in range(inputs.num_cards):
            slots.append({
                "card": card,
                "phases": {"prefill"},
                "cube_share": prefill_cube,
                "vector_share": prefill_vector,
                "hbm_share": 0.45,
                "mode": "dynamic_prefill_partition",
            })
            slots.append({
                "card": card,
                "phases": {"decode"},
                "cube_share": 1.0 - prefill_cube,
                "vector_share": 1.0 - prefill_vector,
                "hbm_share": 0.55,
                "mode": "dynamic_decode_partition",
            })
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return slots


def _service_time(profile: PhaseProfile, slot: dict[str, object]) -> float:
    cube_share = float(slot["cube_share"])
    vector_share = float(slot["vector_share"])
    hbm_share = float(slot["hbm_share"])
    cube_slowdown = profile.cube / max(cube_share, 1e-9)
    vector_slowdown = profile.vector / max(vector_share, 1e-9)
    hbm_slowdown = profile.hbm / max(hbm_share, 1e-9)
    bottleneck = max(1.0, cube_slowdown, vector_slowdown, hbm_slowdown)
    phase_bonus = 0.92 if (profile.phase in slot["phases"] and str(slot["mode"]).startswith(profile.phase)) else 1.0
    if str(slot["mode"]).startswith("dynamic"):
        phase_bonus = 0.88
    return profile.service_ms * bottleneck * phase_bonus


def _placement_summary(slots: list[dict[str, object]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for slot in slots:
        card = str(slot["card"])
        phases = "+".join(sorted(str(p) for p in slot["phases"]))
        out.setdefault(card, []).append(f"{slot['mode']}:{phases}")
    return out


def _float(row: dict[str, str], key: str, *, default: float | None = None) -> float:
    value = (row.get(key) or "").strip()
    if value == "":
        if default is None:
            raise ValueError(f"missing required numeric field {key!r}")
        return default
    return float(value)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
