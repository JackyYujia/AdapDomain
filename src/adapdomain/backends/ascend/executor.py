"""Ascend model executor."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from adapdomain.utils.logging import get_logger

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclExecuteError

_log = get_logger(__name__)


class AscendExecutor:
    def warmup(self, *, model_id: int, input_dataset: Any, output_dataset: Any, runs: int = 5) -> None:
        for _ in range(runs):
            self.execute(model_id=model_id, input_dataset=input_dataset, output_dataset=output_dataset)

    def execute(self, *, model_id: int, input_dataset: Any, output_dataset: Any) -> float:
        acl = _import_acl()
        start = time.perf_counter()
        ret = acl.mdl.execute(model_id, input_dataset, output_dataset)
        elapsed_ms = (time.perf_counter() - start) * 1e3
        if ret != 0:
            raise AclExecuteError("mdl.execute failed", ret=ret, operation="mdl.execute")
        return elapsed_ms

    def benchmark(self, *, model_id: int, input_dataset: Any, output_dataset: Any, repeat: int) -> np.ndarray:
        latencies = np.empty(repeat, dtype=float)
        for i in range(repeat):
            latencies[i] = self.execute(model_id=model_id, input_dataset=input_dataset, output_dataset=output_dataset)
        return latencies
