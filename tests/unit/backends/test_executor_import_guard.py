"""tests/unit/backends/test_executor_import_guard.py"""
import importlib


def test_executor_import_does_not_pull_acl():
    importlib.import_module("adapdomain.backends.ascend.executor")
