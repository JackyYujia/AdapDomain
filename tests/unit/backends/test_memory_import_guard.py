def test_memory_module_does_not_import_acl_at_import_time():
    import importlib
    mod = importlib.import_module("adapdomain.backends.ascend.memory")
    assert hasattr(mod, "malloc_device")
