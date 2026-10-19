import csv

import pytest

from adapdomain.experiments.llm_requests import (
    load_request_csv,
    summarize_requests,
    write_request_summaries,
)


def test_request_level_goodput_uses_both_slos(tmp_path):
    source = tmp_path / "requests.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("request_id", "policy", "trace", "seed", "arrival_ms", "first_token_ms",
                         "completion_ms", "output_tokens", "status"))
        writer.writerows([
            ("r1", "phase_hbm", "burst", 1, 0, 100, 370, 10, "ok"),
            ("r2", "phase_hbm", "burst", 1, 1000, 1600, 1700, 10, "ok"),
            ("r3", "phase_hbm", "burst", 1, 2000, 2100, 2600, 10, "ok"),
            ("r4", "phase_hbm", "burst", 1, 3000, "", "", 0, "error"),
        ])
    result = summarize_requests(load_request_csv(source), duration_s=4.0)
    assert len(result) == 1
    assert result[0].passed == 1
    assert result[0].goodput_qps == pytest.approx(0.25)
    assert result[0].violation_pct == pytest.approx(75.0)
    assert result[0].ttft_p95_ms is not None
    write_request_summaries(result, tmp_path / "summary.csv")
    assert (tmp_path / "summary.csv").exists()


def test_request_log_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "requests.csv"
    source.write_text(
        "request_id,policy,trace,seed,arrival_ms,first_token_ms,completion_ms,output_tokens,status\n"
        "r1,full,burst,1,0,100,200,2,ok\n"
        "r1,full,burst,1,1,101,201,2,ok\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_request_csv(source)
