# msprof data contract

Parser column maps live in `MsprofSchema`. The CANN 8.3.RC1 baseline schema is
the default; when the remote machine ships a different CANN version, add a new
schema entry under `configs/profiling/activevc_default.yaml` rather than
modifying the parser.

Expected fields after parsing:

| Key | Meaning |
| --- | --- |
| op_name | Operator name string. |
| duration | Operator wall time in ms. |
| cube | AI Core matrix-engine utilization in [0, 1]. |
| vector | AI Core vector-engine utilization in [0, 1]. |
| hbm | HBM bandwidth utilization in [0, 1] or None. |

Missing fields produce a row entry with `None` and a single warning log line.
The parser MUST NOT raise on missing fields; this lets profiling runs continue
when a new CANN version drops or renames a column.
