# Ascend 910B3 ACL backend notes

This document records the ACL Python contract used by AdapDomain.

## Lifecycle order

```
acl.init
  acl.rt.set_device(device_id)
    acl.rt.set_device_res_limit(device_id, type, value)   # optional
    acl.rt.create_context(device_id) → ctx
    acl.rt.create_stream() → stream
    acl.mdl.load_from_file(om_path) → model_id
    # …operator execution…
    acl.mdl.unload(model_id)
    acl.rt.destroy_stream(stream)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(device_id)
acl.finalize
```

## `acl.rt.set_device_res_limit`

References:

- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/runtimeapi/aclcppdevg_03_1879.html
- https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/runtimeapi/aclcppdevg_03_1881.html

- Must be called after `set_device` and before any operator execution.
- Last call wins per `(device_id, type)`.
- Huawei CANN 910 beta documents `aclrtDevResLimitType` as
  `ACL_RT_DEV_RES_CUBE_CORE = 0` and `ACL_RT_DEV_RES_VECTOR_CORE = 1`.
  These values are prefilled in `configs/hardware/ascend_910b3_4card.yaml`
  under `acl.set_device_res_limit`.
- No HBM resource-limit type is configured. HBM is tracked as a risk and
  telemetry domain, not as a schedulable ACL quota in this prototype.

## Cleanup expectations

Every allocation must use the `__enter__`/`__exit__` paths in
`backends/ascend/`. Forced unload on error is required to avoid HBM leaks
during long runs.
