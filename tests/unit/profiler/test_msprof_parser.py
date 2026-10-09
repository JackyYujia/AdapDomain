from pathlib import Path

import pytest

from adapdomain.profiler.msprof_parser import MsprofParser, MsprofSchema


FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sample_msprof"


def test_default_schema_parses_minimal_csv():
    schema = MsprofSchema(
        cann_version="8.3.RC1",
        op_name_keys=["op_name"], duration_keys=["task_duration"],
        cube_keys=["aic_mac_ratio"], vector_keys=["aiv_vec_ratio"], hbm_keys=["hbm_bw_util"],
    )
    parser = MsprofParser(schema=schema)
    rows = parser.parse_csv(FIXTURES / "cann83_minimal.csv")
    assert {r["op_name"] for r in rows} == {"GEMM", "ELEMWISE", "LAYERNORM"}
    gemm = next(r for r in rows if r["op_name"] == "GEMM")
    assert gemm["cube"] == pytest.approx(0.80)
    assert gemm["vector"] == pytest.approx(0.10)


def test_variant_schema_remaps_columns():
    schema = MsprofSchema(
        cann_version="8.3.RC1",
        op_name_keys=["OpName"], duration_keys=["Duration"],
        cube_keys=["AicMacRatio"], vector_keys=["AivVecRatio"], hbm_keys=["HBMBwUtilization"],
    )
    rows = MsprofParser(schema=schema).parse_csv(FIXTURES / "cann83_variant.csv")
    assert rows[0]["cube"] == pytest.approx(0.75)


def test_missing_column_logs_warning_not_crash(caplog):
    schema = MsprofSchema(
        cann_version="8.3.RC1",
        op_name_keys=["op_name"], duration_keys=["task_duration"],
        cube_keys=["aic_mac_ratio"], vector_keys=["aiv_vec_ratio"], hbm_keys=["missing_field"],
    )
    rows = MsprofParser(schema=schema).parse_csv(FIXTURES / "cann83_minimal.csv")
    assert all(r["hbm"] is None for r in rows)
    assert any("missing_field" in r.message for r in caplog.records)
