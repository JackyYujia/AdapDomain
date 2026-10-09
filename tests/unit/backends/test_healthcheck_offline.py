"""Healthcheck must produce a report even without ACL/msprof."""
from adapdomain.backends.ascend.healthcheck import run_healthcheck


def test_offline_returns_report():
    rep = run_healthcheck()
    assert rep.python_version
    assert isinstance(rep.acl_importable, bool)
    assert rep.device_count is None or isinstance(rep.device_count, int)
