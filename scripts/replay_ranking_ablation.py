"""Compare card ranking rules on the same phase-arrival replay.

This is a placement sensitivity study, not a reconstruction of live-serving
request goodput. Archived phase arrivals and current profiles are different
experimental cohorts; using them together requires an explicit CLI flag.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

DOMAINS = ("cube", "vector", "hbm")
ARCHIVE_TRACE = "llm_phase/cloud_burst_main/phase_trace.csv"


@dataclass(frozen=True)
class Profile:
    phase: str
    batch: int
    demand: tuple[float, float, float]
    p95_ms: float
    slo_ms: float
    hold_steps: int


@dataclass(frozen=True)
class Arrival:
    phase: str
    arrival_ms: float
    count: int


@dataclass(frozen=True)
class Occupant:
    phase: str
    end_ms: float
    demand: tuple[float, float, float]


@dataclass(frozen=True)
class RiskTerms:
    x: float
    h: float
    q: float
    f: float
    u: float


def load_profiles(path: Path, *, output_tokens: int) -> dict[str, Profile]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = {
        "prefill": (1, 1024, ""),
        "decode": (16, "", 2048),
    }
    profiles = {}
    for phase, (batch, prompt, kv) in selected.items():
        matches = [
            row for row in rows
            if row["phase"] == phase
            and row["precision"] == "FP16"
            and int(row["batch"]) == batch
            and (not prompt or int(row["prompt_tokens"]) == prompt)
            and (not kv or int(row["kv_tokens"]) == kv)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one FP16 {phase} profile, found {len(matches)}")
        row = matches[0]
        profiles[phase] = Profile(
            phase=phase,
            batch=batch,
            demand=tuple(float(row[f"{domain}_fraction"] if domain != "hbm" else row["hbm_bandwidth_fraction"]) for domain in DOMAINS),
            p95_ms=float(row["latency_p95_ms"]),
            slo_ms=500.0 if phase == "prefill" else 40.0,
            hold_steps=1 if phase == "prefill" else output_tokens,
        )
    return profiles


def load_interference(path: Path, *, configuration: str) -> dict[tuple[str, str], tuple[float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if row["configuration"] != configuration:
            continue
        pair = (row["measured_phase"], row["co_resident_phase"])
        mean = float(row["mean_slowdown"])
        tail = float(row["p95_slowdown"]) if row["p95_slowdown"] else mean
        if mean < 1.0 or tail < 1.0:
            raise ValueError(f"invalid slowdown for {pair}")
        result[pair] = (mean, tail)
    if set(result) != {("prefill", "decode"), ("decode", "prefill")}:
        raise ValueError("selected interference configuration must cover both phase directions")
    return result


def load_trace(path: Path | None, archive: Path | None) -> list[Arrival]:
    if bool(path) == bool(archive):
        raise ValueError("pass exactly one of --trace or --supplement-zip")
    if archive:
        with zipfile.ZipFile(archive) as bundle:
            raw = bundle.read(ARCHIVE_TRACE).decode("utf-8-sig")
    else:
        raw = path.read_text(encoding="utf-8-sig")
    arrivals = [
        Arrival(row["phase"], float(row["arrival_ms"]), int(row["count"]))
        for row in csv.DictReader(io.StringIO(raw))
    ]
    if not arrivals or any(a.count <= 0 or a.arrival_ms < 0 for a in arrivals):
        raise ValueError("trace must contain positive counts and nonnegative arrival times")
    if any(a.phase not in {"prefill", "decode"} for a in arrivals):
        raise ValueError("trace contains an unknown phase")
    return sorted(arrivals, key=lambda a: a.arrival_ms)


def feasible(
    profile: Profile,
    occupants: list[Occupant],
    interference: dict[tuple[str, str], tuple[float, float]],
    profiles: dict[str, Profile],
) -> tuple[bool, float]:
    residual = tuple(1.0 - sum(o.demand[j] for o in occupants) for j in range(3))
    if any(profile.demand[j] > residual[j] + 1e-9 for j in range(3)):
        return False, 0.0
    mean_slowdown = tail_slowdown = 1.0
    for other in occupants:
        pair = (profile.phase, other.phase)
        reverse_pair = (other.phase, profile.phase)
        if pair not in interference or reverse_pair not in interference:
            raise ValueError(f"missing measured interference entry {pair}")
        mean_slowdown = max(mean_slowdown, interference[pair][0])
        tail_slowdown = max(tail_slowdown, interference[pair][1])
        other_profile = profiles[other.phase]
        if other_profile.p95_ms * interference[reverse_pair][1] > other_profile.slo_ms:
            return False, 0.0
    if profile.p95_ms * tail_slowdown > profile.slo_ms:
        return False, 0.0
    return True, mean_slowdown


def ranking_score(policy: str, demand: tuple[float, ...], residual: tuple[float, ...]) -> float:
    if policy == "tetris":
        return sum(d * r for d, r in zip(demand, residual))
    if policy == "harmonic":
        ratios = [min(d / (r + 1e-9), r / (d + 1e-9)) for d, r in zip(demand, residual) if d + r > 0]
        return min(1.0, len(ratios) / sum(1.0 / (value + 1e-9) for value in ratios))
    raise ValueError(f"unknown policy: {policy}")


def shape_value(mode: str, demand: tuple[float, ...], residual: tuple[float, ...]) -> float:
    """Pure shape term; the same risk ledger can be added to every variant."""
    if len(demand) != len(residual) or not demand:
        raise ValueError("demand and residual must have the same nonzero dimension")
    if mode in {"harmonic", "tetris"}:
        return ranking_score(mode, demand, residual)
    if mode == "cosine":
        dot = sum(d * r for d, r in zip(demand, residual))
        norms = math.sqrt(sum(d * d for d in demand) * sum(r * r for r in residual))
        return dot / norms if norms else 0.0
    if mode == "no_shape":
        return 0.0
    fits = [min(d / (r + 1e-9), r / (d + 1e-9)) for d, r in zip(demand, residual) if d + r > 0]
    if mode == "arithmetic":
        return sum(fits) / len(fits)
    if mode == "minimum":
        return min(fits)
    if mode == "geometric":
        return math.prod(fits) ** (1.0 / len(fits))
    raise ValueError(f"unknown shape mode: {mode}")


def score_with_shared_risks(
    mode: str,
    demand: tuple[float, ...],
    residual: tuple[float, ...],
    risks: RiskTerms,
) -> float:
    """Paper weights with an identical X/H/Q/F/U vector for each shape mode."""
    return (
        0.40 * shape_value(mode, demand, residual)
        - 0.25 * risks.x
        - 0.15 * risks.h
        - 0.10 * risks.q
        - 0.07 * risks.f
        - 0.05 * risks.u
    )


def replay(
    policy: str,
    profiles: dict[str, Profile],
    interference: dict[tuple[str, str], tuple[float, float]],
    arrivals: list[Arrival],
    *,
    cards: int = 4,
    seed: int = 0,
) -> dict[str, int | float | str]:
    if cards <= 0:
        raise ValueError("cards must be positive")
    rng = random.Random(seed)
    state: list[list[Occupant]] = [[] for _ in range(cards)]
    offered_jobs = admitted_jobs = offered_units = admitted_units = colocations = 0
    mixed_occupancy_choices = informative_rank_choices = voluntary_colocations = 0
    for arrival in arrivals:
        profile = profiles[arrival.phase]
        for occupants in state:
            occupants[:] = [o for o in occupants if o.end_ms > arrival.arrival_ms]
        remaining = arrival.count
        while remaining:
            units = min(profile.batch, remaining)
            remaining -= units
            offered_jobs += 1
            offered_units += units
            candidates = []
            for card, occupants in enumerate(state):
                allowed, slowdown = feasible(profile, occupants, interference, profiles)
                if allowed:
                    residual = tuple(1.0 - sum(o.demand[j] for o in occupants) for j in range(3))
                    candidates.append((card, residual, slowdown, bool(occupants)))
            if not candidates:
                continue
            had_idle_alternative = len({item[3] for item in candidates}) > 1
            mixed_occupancy_choices += had_idle_alternative
            if policy != "random":
                scores = [ranking_score(policy, profile.demand, item[1]) for item in candidates]
                informative_rank_choices += max(scores) - min(scores) > 1e-8
            if policy == "random":
                card, _, slowdown, _ = rng.choice(candidates)
            else:
                card, _, slowdown, _ = max(
                    candidates,
                    key=lambda item: (ranking_score(policy, profile.demand, item[1]), -item[0]),
                )
            colocations += bool(state[card])
            voluntary_colocations += had_idle_alternative and bool(state[card])
            admitted_jobs += 1
            admitted_units += units
            state[card].append(Occupant(
                phase=profile.phase,
                end_ms=arrival.arrival_ms + profile.p95_ms * profile.hold_steps * slowdown,
                demand=profile.demand,
            ))
    return {
        "policy": policy,
        "seed": seed,
        "offered_phase_jobs": offered_jobs,
        "admitted_phase_jobs": admitted_jobs,
        "offered_trace_units": offered_units,
        "admitted_trace_units": admitted_units,
        "admission_pct": round(100.0 * admitted_units / offered_units, 2),
        "colocated_phase_jobs": colocations,
        "colocations_with_idle_alternative": voluntary_colocations,
        "mixed_occupancy_choices": mixed_occupancy_choices,
        "informative_rank_choices": informative_rank_choices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--interference", type=Path, required=True)
    parser.add_argument("--configuration", default="phase_aware_plus_HBM")
    trace = parser.add_mutually_exclusive_group(required=True)
    trace.add_argument("--trace", type=Path, help="matching phase,arrival_ms,count CSV")
    trace.add_argument("--supplement-zip", type=Path, help="archived older phase trace")
    parser.add_argument("--acknowledge-cross-cohort", action="store_true")
    parser.add_argument("--cards", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    args = parser.parse_args()
    if args.supplement_zip and not args.acknowledge_cross_cohort:
        parser.error("archived arrivals differ from current profiles; pass --acknowledge-cross-cohort for an exploratory replay")
    if args.output_tokens <= 0 or args.random_seeds <= 0:
        parser.error("--output-tokens and --random-seeds must be positive")
    profiles = load_profiles(args.profiles, output_tokens=args.output_tokens)
    interference = load_interference(args.interference, configuration=args.configuration)
    arrivals = load_trace(args.trace, args.supplement_zip)
    results = [replay(policy, profiles, interference, arrivals, cards=args.cards) for policy in ("harmonic", "tetris")]
    results.extend(replay("random", profiles, interference, arrivals, cards=args.cards, seed=seed) for seed in range(args.random_seeds))
    report = {
        "status": "exploratory_cross_cohort" if args.supplement_zip else "same_cohort_claim_not_verified",
        "metric": "phase-slot admission under a fixed p95 holding-time model; not live request goodput",
        "ranking": "harmonic CEFF vs pre-placement demand-residual dot product vs uniform random feasible card",
        "guards": "same Cube/Vector/HBM budgets and measured phase-conditioned slowdown for all policies",
        "trace_units": "archived phase counts converted to ceil(count / current profile batch) jobs" if args.supplement_zip else "trace counts converted to phase jobs",
        "results": results,
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
