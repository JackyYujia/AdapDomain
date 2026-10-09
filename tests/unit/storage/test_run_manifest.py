import json

from adapdomain.storage.run_manifest import RunManifest


def test_manifest_writes_required_fields(tmp_path):
    out = tmp_path / "run"
    rm = RunManifest.start(run_id="r0", command="adapdomain replay e2e", out_dir=out)
    rm.add_config_hash("abc123")
    rm.set_hardware("ascend_910b3", device_count=4, cann_version="8.3.RC1")
    rm.set_status("done")
    rm.finalize()

    data = json.loads((out / "manifest.json").read_text())
    assert data["run_id"] == "r0"
    assert data["status"] == "done"
    assert data["device_info"]["count"] == 4
