from pathlib import Path

import pytest

from adapdomain.experiments.llm_pd import (
    LlmPdInputs,
    bursty_trace,
    default_profiles,
    run_llm_pd_experiment,
    write_results,
    load_profiles_csv,
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


def test_measured_profile_keeps_bandwidth_separate_from_memory(tmp_path: Path):
    path = tmp_path / "profiles.csv"
    path.write_text(
        "phase,cube_util,vector_util,hbm_bw_util,hbm_gb,kv_cache_gb,p95_ms,slo_ms,batch_size\n"
        "prefill,0.71,0.19,0.22,20,2,151,500,1\n"
        "decode,0.14,0.34,0.78,40,8,26.1,40,16\n", encoding="utf-8",
    )
    profiles = load_profiles_csv(path)
    assert profiles["prefill"].hbm == pytest.approx(0.22)
    assert profiles["decode"].hbm == pytest.approx(0.78)
    assert profiles["decode"].kv_cache_gb == pytest.approx(8)


def test_measured_profile_requires_hbm_bandwidth(tmp_path: Path):
    path = tmp_path / "profiles.csv"
    path.write_text(
        "phase,cube_util,vector_util,hbm_gb,p95_ms,slo_ms,batch_size\n"
        "prefill,0.71,0.19,20,151,500,1\n"
        "decode,0.14,0.34,40,26.1,40,16\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hbm_bw_util"):
        load_profiles_csv(path)


def test_mixed_precision_profiles_need_selection(tmp_path: Path):
    path = tmp_path / "profiles.csv"
    path.write_text(
        "phase,precision,cube_util,vector_util,hbm_bw_util,p95_ms,slo_ms,batch_size\n"
        "prefill,FP16,0.71,0.19,0.22,151,500,1\n"
        "decode,FP16,0.14,0.34,0.78,26.1,40,16\n"
        "prefill,W8A8,0.58,0.27,0.17,101,500,1\n"
        "decode,W8A8,0.12,0.36,0.66,20.3,40,16\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mixes precisions"):
        load_profiles_csv(path)
    selected = load_profiles_csv(path, precision="W8A8")
    assert selected["decode"].hbm == pytest.approx(0.66)
