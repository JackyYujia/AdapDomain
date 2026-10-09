from pathlib import Path

from adapdomain.experiments.llm_pd import (
    LlmPdInputs,
    bursty_trace,
    default_profiles,
    run_llm_pd_experiment,
    write_results,
)


def test_llm_pd_default_experiment_runs(tmp_path: Path):
    results = run_llm_pd_experiment(LlmPdInputs(profiles=default_profiles(), trace=bursty_trace(windows=4)))
    by_policy = {r.policy: r for r in results}

    assert {"no_partition", "static_pd_colocation", "pd_separate", "pd_dynamic_partition"} <= set(by_policy)
    assert by_policy["pd_dynamic_partition"].goodput_qps >= 0
    assert by_policy["pd_dynamic_partition"].total > 0

    write_results(results, tmp_path)
    assert (tmp_path / "llm_pd_summary.csv").exists()
    assert (tmp_path / "llm_pd_summary.md").exists()


def test_dynamic_partition_beats_static_in_default_smoke():
    results = run_llm_pd_experiment(LlmPdInputs(profiles=default_profiles(), trace=bursty_trace(windows=8)))
    by_policy = {r.policy: r for r in results}

    assert by_policy["pd_dynamic_partition"].goodput_qps >= by_policy["static_pd_colocation"].goodput_qps

