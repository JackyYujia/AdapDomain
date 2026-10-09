import importlib
import sys

import pytest


def test_acl_runtime_raises_when_acl_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "acl", None)
    if "adapdomain.backends.ascend.acl_runtime" in sys.modules:
        del sys.modules["adapdomain.backends.ascend.acl_runtime"]
    mod = importlib.import_module("adapdomain.backends.ascend.acl_runtime")
    rt = mod.AclRuntime()
    with pytest.raises(mod.AclInitError):
        rt.init()
