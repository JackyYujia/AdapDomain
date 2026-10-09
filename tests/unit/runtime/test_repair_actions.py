import pytest

from adapdomain.backends.base import ResourceQuota
from adapdomain.runtime.repair_actions import RepairLogger, perform_quota_repair


def test_quota_repair_logs_event(tmp_path):
    logger = RepairLogger(tmp_path / "events.jsonl")
    perform_quota_repair(logger=logger, device_id=0, replica_id="r0", new_quota=ResourceQuota(device_id=0, cube_units=5, vector_units=10), reason="hbm_high")
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert lines
    assert "quota_repair" in lines[0]
