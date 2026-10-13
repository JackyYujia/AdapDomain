"""Replay LLaMA phase policies with recorded and equal service coefficients.

Uses only the standard library and the project's existing llm_pd.py model.
The input profile and trace CSVs must refer to the same workload.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path


def load_model():
    module_path = Path(__file__).resolve().parents[1] / "src/adapdomain/experiments/llm_pd.py"
    spec = importlib.util.spec_from_file_location("llm_pd_sensitivity", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def equal_coefficient_service_time(profile, slot):
    bottleneck = max(
        1.0,
        profile.cube / float(slot["cube_share"]),
        profile.vector / float(slot["vector_share"]),
        profile.hbm / float(slot["hbm_share"]),
    )
    return profile.service_ms * bottleneck


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, help="Measured phase-profile CSV")
    parser.add_argument("trace", type=Path, help="Matched phase-request trace CSV")
    parser.add_argument("--expected-summary", type=Path, help="Archived policy-summary CSV to cross-check")
    args = parser.parse_args()

    module = load_model()
    inputs = module.LlmPdInputs(
        profiles=module.load_profiles_csv(args.profiles),
        trace=module.load_trace_csv(args.trace),
    )
    recorded = module.run_llm_pd_experiment(inputs)
    if args.expected_summary:
        with args.expected_summary.open(newline="", encoding="utf-8") as stream:
            expected = {row["policy"]: row for row in csv.DictReader(stream)}
        for row in recorded:
            reference = expected[row.policy]
            for field in ("goodput_qps", "violation_pct", "p95_latency_ms"):
                if abs(getattr(row, field) - float(reference[field])) > 1e-5:
                    raise ValueError(f"Recorded replay differs from archive: {row.policy} {field}")
        print("Recorded-policy replay matches the archived summary.")

    module._service_time = equal_coefficient_service_time
    equal = module.run_llm_pd_experiment(inputs)
    print("scenario,policy,accepted,total,goodput_qps,violation_pct,p95_latency_ms")
    for scenario, results in (("recorded", recorded), ("equal_coeff_1.00", equal)):
        for row in results:
            print(
                f"{scenario},{row.policy},{row.accepted},{row.total},"
                f"{row.goodput_qps:.6f},{row.violation_pct:.6f},{row.p95_latency_ms:.6f}"
            )


if __name__ == "__main__":
    main()
