"""Each ascend device/context/stream module imports lazily — must not pull `acl` at import time."""
import importlib


def test_device_module_importable_without_acl():
    importlib.import_module("adapdomain.backends.ascend.device")


def test_context_module_importable_without_acl():
    importlib.import_module("adapdomain.backends.ascend.context")


def test_stream_module_importable_without_acl():
    importlib.import_module("adapdomain.backends.ascend.stream")
