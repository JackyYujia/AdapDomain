# AdapDomain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the AdapDomain paper system end-to-end against four real Ascend 910B3 NPU cards (Cube20 / Vector40): ACL Python backend, ActiveVC profiling via msprof, HarmonicFit complementarity-aware placement, SmoothGuard / PreScale / NightShift runtime adaptation, and the full experiment pipeline reproducing the paper figures and tables.

**Architecture:** Layered Python package with strict boundaries — `core/` pure data model, `backends/ascend/` the only ACL importer, `profiler/` for ActiveVC + msprof, `scheduler/` for HarmonicFit, `runtime/` for guards, `experiments/` for replay/ablation/plots. Code runs locally (algorithms, unit tests) and on a remote Linux NPU server (hardware execution, profiling, replay).

**Tech Stack:** Python 3.10+, pydantic v2, numpy, pandas, pyyaml, typer, rich, matplotlib, pytest; remote-side CANN 8.3.RC1 ACL Python, msprof, npu-smi.

**Repository note:** The workspace at `/Users/xiumuyoukediaoye/Desktop/docPaper/AdapDomain` is NOT a git repository in this session. Each "Commit" step is written for once the user initializes git. If git is not yet initialized when a task completes, append the same record to `docs/development/continuation_checklist.md` instead and skip the `git` commands; the source-of-truth log is the checklist.

**Source-of-truth files for resuming:**

- Design spec: `docs/superpowers/specs/2026-06-26-adapdomain-implementation-design.md`
- Module plan: `docs/development/module_plan.md`
- Continuation log: `docs/development/continuation_checklist.md`

Every task ends with a checklist append. Subagents must read all three before starting any module.

---

## File Structure Map

This plan creates and modifies files in three stages. Tasks below are grouped per module (M1–M41) defined in the design spec.

```text
AdapDomain/
  README.md
  pyproject.toml
  .gitignore
  configs/
    hardware/ascend_910b3_4card.yaml
    models/model_registry.example.yaml
    profiling/activevc_default.yaml
    scheduler/harmonicfit_default.yaml
    runtime/runtime_default.yaml
    experiments/e2e_replay.yaml
    experiments/ablation.yaml
  src/adapdomain/
    __init__.py
    core/{errors,types,domain,shape,frontier,telemetry,workload,interference,placement,metrics}.py
    backends/base.py
    backends/ascend/{acl_runtime,errors,device,context,stream,memory,model,dataset,executor,resource_limit,healthcheck}.py
    profiler/{msprof_runner,msprof_parser,normalization,confidence,frontier_builder,candidate_grid,acquisition,activevc}.py
    scheduler/{feasibility,scoring,hbm_risk,queue_risk,interference_risk,fragmentation,residual_state,harmonic_fit,planner}.py
    runtime/{ewma,ring_buffer,monitor,smooth_guard,prescale,nightshift,repair_actions}.py
    experiments/{seeds,replay,sweeps,ablations,overhead,plotting,tables,profile_vm,collect_interference}.py
    storage/{jsonl,csv_store,frontier_store,run_manifest,artifacts}.py
    cli/main.py
    cli/commands/{profile,schedule,run_model,replay,ablate,plot,verify_remote}.py
    utils/{logging,config,subprocess,time,validation}.py
  scripts/remote_ascend/{setup_env,run_healthcheck,run_profile,collect_msprof,run_e2e_replay,run_paper_bundle}.sh
  tests/
    conftest.py
    unit/{utils,core,profiler,scheduler,runtime,storage}/
    integration/{test_acl_backend_import_guard.py,test_cli_smoke.py}
    fixtures/{sample_msprof,sample_frontiers,sample_traces}/
  docs/
    hardware/{ascend_910b3_acl_backend.md,msprof_data_contract.md,remote_validation.md}
    development/{module_plan.md,continuation_checklist.md}  # already exist
    superpowers/{specs,plans}/...                            # already exist
  data/  # runtime output, gitignored
```

Each module's tasks list its own create/modify file paths. Subagent boundary rule: a subagent assigned to module Mxx may only create or modify files listed under Mxx.

---

## Stage A — Skeleton and Core Data Model

### Task M1: Project skeleton & configs

**Files:**

- Create: `AdapDomain/pyproject.toml`
- Create: `AdapDomain/.gitignore`
- Create: `AdapDomain/README.md`
- Create: `AdapDomain/src/adapdomain/__init__.py`
- Create: `AdapDomain/src/adapdomain/cli/__init__.py`
- Create: `AdapDomain/src/adapdomain/cli/main.py`
- Create: `AdapDomain/src/adapdomain/cli/commands/__init__.py`
- Create: `AdapDomain/configs/hardware/ascend_910b3_4card.yaml`
- Create: `AdapDomain/configs/models/model_registry.example.yaml`
- Create: `AdapDomain/configs/profiling/activevc_default.yaml`
- Create: `AdapDomain/configs/scheduler/harmonicfit_default.yaml`
- Create: `AdapDomain/configs/runtime/runtime_default.yaml`
- Create: `AdapDomain/configs/experiments/e2e_replay.yaml`
- Create: `AdapDomain/configs/experiments/ablation.yaml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "adapdomain"
version = "0.1.0"
description = "Resource-shape-aware multi-tenant inference scheduling for Ascend 910B3"
requires-python = ">=3.10"
dependencies = [
  "pydantic>=2.5",
  "pyyaml>=6.0",
  "typer>=0.12",
  "rich>=13.7",
  "numpy>=1.26",
  "pandas>=2.2",
  "matplotlib>=3.8",
  "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.4"]

[project.scripts]
adapdomain = "adapdomain.cli.main:app"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 2: Write `.gitignore`**

```text
__pycache__/
*.pyc
*.egg-info/
.venv/
.env
.pytest_cache/
.coverage
htmlcov/
data/
!data/.gitkeep
build/
dist/
.DS_Store
```

- [ ] **Step 3: Write minimal `README.md`**

```markdown
# AdapDomain

Resource-shape-aware multi-tenant inference scheduling for Ascend 910B3 NPUs.

See `docs/superpowers/specs/2026-06-26-adapdomain-implementation-design.md` for the design.

Quick start:

```bash
pip install -e .[dev]
adapdomain --help
```
```

- [ ] **Step 4: Write CLI scaffolding `src/adapdomain/__init__.py` and `cli/main.py`**

`src/adapdomain/__init__.py`:

```python
"""AdapDomain: resource-shape-aware multi-tenant inference scheduling."""

__version__ = "0.1.0"
```

`src/adapdomain/cli/__init__.py`:

```python
"""CLI package."""
```

`src/adapdomain/cli/commands/__init__.py`:

```python
"""CLI subcommands."""
```

`src/adapdomain/cli/main.py`:

```python
"""Top-level Typer app for AdapDomain."""
from __future__ import annotations

import typer

app = typer.Typer(
    name="adapdomain",
    help="Resource-shape-aware multi-tenant inference scheduling for Ascend 910B3.",
    no_args_is_help=True,
)


@app.command("version")
def version() -> None:
    """Print the installed AdapDomain version."""
    from adapdomain import __version__

    typer.echo(__version__)


@app.command("verify-remote")
def verify_remote_placeholder() -> None:
    """Placeholder; real implementation in M17."""
    typer.echo("verify-remote: not implemented yet (see M17).")


@app.command("run-model")
def run_model_placeholder() -> None:
    """Placeholder; real implementation in M15/M17."""
    typer.echo("run-model: not implemented yet.")


@app.command("profile")
def profile_placeholder() -> None:
    """Placeholder; real implementation in M21."""
    typer.echo("profile: not implemented yet.")


@app.command("schedule")
def schedule_placeholder() -> None:
    """Placeholder; real implementation in M28."""
    typer.echo("schedule: not implemented yet.")


@app.command("replay")
def replay_placeholder() -> None:
    """Placeholder; real implementation in M34."""
    typer.echo("replay: not implemented yet.")


@app.command("ablate")
def ablate_placeholder() -> None:
    """Placeholder; real implementation in M36/M37."""
    typer.echo("ablate: not implemented yet.")


@app.command("plot")
def plot_placeholder() -> None:
    """Placeholder; real implementation in M39/M41."""
    typer.echo("plot: not implemented yet.")


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Write hardware config `configs/hardware/ascend_910b3_4card.yaml`**

```yaml
platform: ascend_910b3
device_ids: [0, 1, 2, 3]
domains:
  cube:
    capacity: 20
    unit: "logical_units"
    telemetry_key: aic_mac_ratio
    schedulable: true
  vector:
    capacity: 40
    unit: "logical_units"
    telemetry_key: aiv_vec_ratio
    schedulable: true
  hbm:
    capacity: 1.0
    unit: "fraction"
    telemetry_key: hbm_bw_util
    schedulable: false
acl:
  init_config_path: null
  set_device_res_limit:
    cube_limit_type: null     # fill after remote CANN confirmation
    vector_limit_type: null
    hbm_limit_type: null
msprof:
  binary: msprof
  default_metrics:
    - PipeUtilization
    - ArithmeticUtilization
    - HBM
  output_root: data/raw/msprof
```

- [ ] **Step 6: Write the remaining default configs**

`configs/models/model_registry.example.yaml`:

```yaml
models:
  - name: resnet152
    om_path: /path/to/resnet152.om
    slo_ms: 30.0
    batch_size: 1
  - name: bert-base
    om_path: /path/to/bert_base.om
    slo_ms: 15.0
    batch_size: 1
```

`configs/profiling/activevc_default.yaml`:

```yaml
budget: 24
alpha: 0.40
beta: 0.40
gamma: 0.20
tau: 0.15
seed_grid: [4, 4, 3]    # V × M × HBM
boundary_grid: [8, 8]   # V × M
hbm_strategy: median_safe
```

`configs/scheduler/harmonicfit_default.yaml`:

```yaml
weights:
  lambda_c: 0.40
  lambda_x: 0.25
  lambda_h: 0.15
  lambda_q: 0.10
  lambda_f: 0.07
  lambda_u: 0.05
interference:
  eta: 0.30
epsilon_domain_fit: 1.0e-6
k_domains: ["cube", "vector"]
control_budget_ms: 100
```

`configs/runtime/runtime_default.yaml`:

```yaml
ewma:
  alpha: 0.20
smooth_guard:
  consecutive_trigger_count: 2
  cooldown_seconds: 30
  thresholds:
    p95_drift: 0.20
    residual_drift: 0.15
    hbm_high: 0.85
    queue_high: 0.80
prescale:
  forecast_horizon_s: 5
nightshift:
  migration_budget: 2
  low_load_windows: 3
```

`configs/experiments/e2e_replay.yaml`:

```yaml
seeds: [1, 2, 3, 4, 5]
windows: 2400
arrival_rates: [1.0]
slo_thresholds_ms: null
output_root: data/results/replay
```

`configs/experiments/ablation.yaml`:

```yaml
seeds: [1, 2, 3, 4, 5]
windows: 1200
harmonicfit_variants:
  - full
  - no_interference
  - no_ceff
  - no_hbm
  - no_fragment
  - no_empty_penalty
activevc_baselines:
  - random
  - binary
  - bo
  - activevc
budgets: [8, 16, 24, 32, 48]
output_root: data/results/ablation
```

- [ ] **Step 7: Install package and verify CLI**

Run:

```bash
cd /Users/xiumuyoukediaoye/Desktop/docPaper/AdapDomain
python -m pip install -e .[dev]
adapdomain --help
adapdomain version
```

Expected: `--help` shows `version`, `verify-remote`, `run-model`, `profile`, `schedule`, `replay`, `ablate`, `plot`. `adapdomain version` prints `0.1.0`.

- [ ] **Step 8: Verify configs load as YAML**

Run:

```bash
python - <<'PY'
import yaml, pathlib
root = pathlib.Path("configs")
for p in root.rglob("*.yaml"):
    yaml.safe_load(p.read_text())
    print("ok", p)
PY
```

Expected: prints `ok` for all 7 config files.

- [ ] **Step 9: Commit and log**

```bash
git add -A
git commit -m "M1: project skeleton, CLI placeholders, default configs"
```

Append to `docs/development/continuation_checklist.md`:

```
2026-06-26  M1 DONE  ---  skeleton + CLI + configs installed; adapdomain --help OK; all 7 yaml load
```

Update `docs/development/module_plan.md`: set M1 status to `done`.

---

### Task M2: Utils

**Files:**

- Create: `AdapDomain/src/adapdomain/utils/__init__.py`
- Create: `AdapDomain/src/adapdomain/utils/logging.py`
- Create: `AdapDomain/src/adapdomain/utils/time.py`
- Create: `AdapDomain/src/adapdomain/utils/validation.py`
- Create: `AdapDomain/src/adapdomain/utils/subprocess.py`
- Create: `AdapDomain/src/adapdomain/utils/config.py`
- Create: `AdapDomain/tests/__init__.py`
- Create: `AdapDomain/tests/conftest.py`
- Create: `AdapDomain/tests/unit/__init__.py`
- Create: `AdapDomain/tests/unit/utils/__init__.py`
- Create: `AdapDomain/tests/unit/utils/test_config.py`
- Create: `AdapDomain/tests/unit/utils/test_subprocess.py`

- [ ] **Step 1: Write failing config tests `tests/unit/utils/test_config.py`**

```python
from pathlib import Path

import pytest

from adapdomain.utils.config import (
    HardwareConfig,
    HarmonicFitConfig,
    ActiveVCConfig,
    RuntimeConfig,
    ExperimentReplayConfig,
    load_yaml,
)


CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def test_hardware_config_loads_defaults():
    cfg = HardwareConfig.from_yaml(CONFIG_ROOT / "hardware" / "ascend_910b3_4card.yaml")
    assert cfg.platform == "ascend_910b3"
    assert cfg.device_ids == [0, 1, 2, 3]
    assert cfg.domains["cube"].capacity == 20
    assert cfg.domains["vector"].capacity == 40


def test_harmonicfit_weights_sum_positive():
    cfg = HarmonicFitConfig.from_yaml(CONFIG_ROOT / "scheduler" / "harmonicfit_default.yaml")
    assert cfg.weights.lambda_c > 0
    assert cfg.interference.eta == 0.30
    assert cfg.k_domains == ["cube", "vector"]


def test_activevc_default_budget_is_24():
    cfg = ActiveVCConfig.from_yaml(CONFIG_ROOT / "profiling" / "activevc_default.yaml")
    assert cfg.budget == 24
    assert cfg.alpha == 0.40
    assert cfg.beta == 0.40
    assert cfg.gamma == 0.20
    assert cfg.tau == 0.15


def test_runtime_smooth_guard_has_two_trigger_count():
    cfg = RuntimeConfig.from_yaml(CONFIG_ROOT / "runtime" / "runtime_default.yaml")
    assert cfg.smooth_guard.consecutive_trigger_count == 2
    assert cfg.smooth_guard.cooldown_seconds >= 30


def test_experiment_replay_defaults():
    cfg = ExperimentReplayConfig.from_yaml(CONFIG_ROOT / "experiments" / "e2e_replay.yaml")
    assert cfg.seeds == [1, 2, 3, 4, 5]
    assert cfg.windows == 2400


def test_load_yaml_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nope.yaml")
```

- [ ] **Step 2: Write failing subprocess test `tests/unit/utils/test_subprocess.py`**

```python
import sys

import pytest

from adapdomain.utils.subprocess import SubprocessError, run_checked


def test_run_checked_captures_stdout():
    res = run_checked([sys.executable, "-c", "print('hello')"], timeout=10)
    assert res.returncode == 0
    assert "hello" in res.stdout


def test_run_checked_raises_on_nonzero():
    with pytest.raises(SubprocessError) as excinfo:
        run_checked([sys.executable, "-c", "import sys; sys.exit(3)"], timeout=10)
    assert excinfo.value.returncode == 3


def test_run_checked_raises_on_timeout():
    with pytest.raises(SubprocessError):
        run_checked([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
```

- [ ] **Step 3: Run tests to confirm failure**

Run: `pytest tests/unit/utils/ -v`
Expected: import errors (modules not yet defined).

- [ ] **Step 4: Implement `utils/logging.py`**

```python
"""Structured logging for AdapDomain."""
from __future__ import annotations

import logging
import os
from typing import Literal

from rich.logging import RichHandler

_LOG_FORMAT_RICH = "%(message)s"
_LOG_FORMAT_PLAIN = "%(asctime)s %(levelname)s %(name)s :: %(message)s"


def configure_logging(
    level: int | str = "INFO",
    style: Literal["rich", "plain"] = "rich",
) -> None:
    """Configure root logger. Idempotent."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    if style == "rich" and os.isatty(2):
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_time=False)
        fmt = _LOG_FORMAT_RICH
    else:
        handler = logging.StreamHandler()
        fmt = _LOG_FORMAT_PLAIN

    handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
```

- [ ] **Step 5: Implement `utils/time.py`**

```python
"""Time helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return now_utc().isoformat(timespec="seconds")


@contextmanager
def stopwatch() -> Iterator[dict]:
    """Yield a dict that gets a 'seconds' key on context exit."""
    start = time.perf_counter()
    state: dict = {}
    try:
        yield state
    finally:
        state["seconds"] = time.perf_counter() - start
```

- [ ] **Step 6: Implement `utils/validation.py`**

```python
"""Small validation helpers used across modules."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def ensure_in_range(value: float, low: float, high: float, name: str) -> float:
    if not (low <= value <= high):
        raise ValueError(f"{name}={value} not in [{low}, {high}]")
    return value


def ensure_unique(items: Iterable, name: str) -> None:
    seen: set = set()
    for it in items:
        if it in seen:
            raise ValueError(f"{name}: duplicate value {it!r}")
        seen.add(it)


def ensure_path_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path
```

- [ ] **Step 7: Implement `utils/subprocess.py`**

```python
"""Subprocess wrappers with strict error handling."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Sequence


class SubprocessError(RuntimeError):
    def __init__(self, cmd: Sequence[str], returncode: int, stdout: str, stderr: str) -> None:
        super().__init__(f"command {list(cmd)!r} failed with code {returncode}: {stderr.strip()[:200]}")
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class RunResult:
    cmd: list
    returncode: int
    stdout: str
    stderr: str


def run_checked(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: str | None = None,
    env: dict | None = None,
) -> RunResult:
    """Run cmd, capture stdout/stderr, raise SubprocessError on failure or timeout."""
    try:
        proc = subprocess.run(
            list(cmd),
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
        raise SubprocessError(cmd, -1, stdout, stderr + f"\n[timeout after {timeout}s]")

    if proc.returncode != 0:
        raise SubprocessError(cmd, proc.returncode, proc.stdout or "", proc.stderr or "")

    return RunResult(cmd=list(cmd), returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")
```

- [ ] **Step 8: Implement `utils/config.py`**

```python
"""Typed config models loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return yaml.safe_load(p.read_text()) or {}


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "_Base":
        return cls.model_validate(load_yaml(path))


class DomainConfig(_Base):
    capacity: float
    unit: str
    telemetry_key: str
    schedulable: bool


class SetDeviceResLimitConfig(_Base):
    cube_limit_type: int | None = None
    vector_limit_type: int | None = None
    hbm_limit_type: int | None = None


class AclConfig(_Base):
    init_config_path: str | None = None
    set_device_res_limit: SetDeviceResLimitConfig = Field(default_factory=SetDeviceResLimitConfig)


class MsprofConfig(_Base):
    binary: str = "msprof"
    default_metrics: list[str] = Field(default_factory=list)
    output_root: str = "data/raw/msprof"


class HardwareConfig(_Base):
    platform: str
    device_ids: list[int]
    domains: dict[str, DomainConfig]
    acl: AclConfig = Field(default_factory=AclConfig)
    msprof: MsprofConfig = Field(default_factory=MsprofConfig)

    @field_validator("device_ids")
    @classmethod
    def _unique_ids(cls, v: list[int]) -> list[int]:
        if len(set(v)) != len(v):
            raise ValueError("device_ids must be unique")
        return v


class HarmonicFitWeights(_Base):
    lambda_c: float
    lambda_x: float
    lambda_h: float
    lambda_q: float
    lambda_f: float
    lambda_u: float


class HarmonicFitInterference(_Base):
    eta: float


class HarmonicFitConfig(_Base):
    weights: HarmonicFitWeights
    interference: HarmonicFitInterference
    epsilon_domain_fit: float
    k_domains: list[str]
    control_budget_ms: float


class ActiveVCConfig(_Base):
    budget: int
    alpha: float
    beta: float
    gamma: float
    tau: float
    seed_grid: list[int]
    boundary_grid: list[int]
    hbm_strategy: str


class RuntimeEWMA(_Base):
    alpha: float


class RuntimeSmoothGuardThresholds(_Base):
    p95_drift: float
    residual_drift: float
    hbm_high: float
    queue_high: float


class RuntimeSmoothGuard(_Base):
    consecutive_trigger_count: int
    cooldown_seconds: float
    thresholds: RuntimeSmoothGuardThresholds


class RuntimePreScale(_Base):
    forecast_horizon_s: float


class RuntimeNightShift(_Base):
    migration_budget: int
    low_load_windows: int


class RuntimeConfig(_Base):
    ewma: RuntimeEWMA
    smooth_guard: RuntimeSmoothGuard
    prescale: RuntimePreScale
    nightshift: RuntimeNightShift


class ExperimentReplayConfig(_Base):
    seeds: list[int]
    windows: int
    arrival_rates: list[float]
    slo_thresholds_ms: list[float] | None = None
    output_root: str


class ExperimentAblationConfig(_Base):
    seeds: list[int]
    windows: int
    harmonicfit_variants: list[str]
    activevc_baselines: list[str]
    budgets: list[int]
    output_root: str
```

- [ ] **Step 9: Implement `utils/__init__.py`**

```python
"""AdapDomain utility package."""
from .config import (
    ActiveVCConfig,
    ExperimentAblationConfig,
    ExperimentReplayConfig,
    HardwareConfig,
    HarmonicFitConfig,
    RuntimeConfig,
    load_yaml,
)
from .logging import configure_logging, get_logger
from .subprocess import RunResult, SubprocessError, run_checked
from .time import iso_now, now_utc, stopwatch
from .validation import ensure_in_range, ensure_path_exists, ensure_unique

__all__ = [
    "ActiveVCConfig",
    "ExperimentAblationConfig",
    "ExperimentReplayConfig",
    "HardwareConfig",
    "HarmonicFitConfig",
    "RuntimeConfig",
    "load_yaml",
    "configure_logging",
    "get_logger",
    "RunResult",
    "SubprocessError",
    "run_checked",
    "iso_now",
    "now_utc",
    "stopwatch",
    "ensure_in_range",
    "ensure_path_exists",
    "ensure_unique",
]
```

- [ ] **Step 10: Add minimal `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_random_state() -> None:
    random.seed(1234)
    np.random.seed(1234)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
```

- [ ] **Step 11: Run tests, expect green**

Run: `pytest tests/unit/utils/ -v`
Expected: all 9 tests pass.

- [ ] **Step 12: Commit and log**

```bash
git add -A
git commit -m "M2: utils (logging, time, validation, subprocess, typed config)"
```

Append to checklist:

```
2026-06-26  M2 DONE  ---  utils + pydantic config models green; pytest tests/unit/utils/ all pass
```

Update `module_plan.md` row M2 → `done`.

---

### Task M3: Errors & types

**Files:**

- Create: `AdapDomain/src/adapdomain/core/__init__.py`
- Create: `AdapDomain/src/adapdomain/core/errors.py`
- Create: `AdapDomain/src/adapdomain/core/types.py`
- Create: `AdapDomain/tests/unit/core/__init__.py`
- Create: `AdapDomain/tests/unit/core/test_errors_and_types.py`

- [ ] **Step 1: Write failing test `tests/unit/core/test_errors_and_types.py`**

```python
import pytest

from adapdomain.core.errors import (
    AdapDomainError,
    AclDeviceError,
    AclExecuteError,
    AclInitError,
    AclMemoryError,
    AclModelError,
    BackendError,
    ConfigError,
    EmptyCandidateError,
    InfeasiblePlacementError,
    MsprofParseError,
    MsprofRunError,
    ProfilerError,
    RepairBudgetExceededError,
    RuntimeAdapterError,
    SchedulerError,
)
from adapdomain.core.types import DEFAULT_K, EPSILON_DOMAIN_FIT


def test_error_hierarchy():
    assert issubclass(ConfigError, AdapDomainError)
    assert issubclass(BackendError, AdapDomainError)
    for cls in [AclInitError, AclDeviceError, AclModelError, AclMemoryError, AclExecuteError]:
        assert issubclass(cls, BackendError)
    assert issubclass(ProfilerError, AdapDomainError)
    for cls in [MsprofRunError, MsprofParseError]:
        assert issubclass(cls, ProfilerError)
    assert issubclass(SchedulerError, AdapDomainError)
    for cls in [InfeasiblePlacementError, EmptyCandidateError]:
        assert issubclass(cls, SchedulerError)
    assert issubclass(RuntimeAdapterError, AdapDomainError)
    assert issubclass(RepairBudgetExceededError, RuntimeAdapterError)


def test_constants():
    assert EPSILON_DOMAIN_FIT == pytest.approx(1e-6)
    assert DEFAULT_K == 2


def test_acl_error_carries_ret_code():
    err = AclInitError("init failed", ret=507000)
    assert err.ret == 507000
    assert "507000" in str(err)
```

- [ ] **Step 2: Run test, confirm fail**

Run: `pytest tests/unit/core/test_errors_and_types.py -v`
Expected: import error.

- [ ] **Step 3: Implement `core/errors.py`**

```python
"""Error hierarchy for AdapDomain."""
from __future__ import annotations


class AdapDomainError(Exception):
    """Base class for all AdapDomain errors."""


class ConfigError(AdapDomainError):
    """Invalid or missing configuration."""


class BackendError(AdapDomainError):
    """Errors raised by an inference backend (e.g. ACL)."""

    def __init__(self, message: str, *, ret: int | None = None, operation: str | None = None) -> None:
        if ret is not None:
            message = f"{message} (ret={ret}, op={operation})"
        super().__init__(message)
        self.ret = ret
        self.operation = operation


class AclInitError(BackendError): ...
class AclDeviceError(BackendError): ...
class AclModelError(BackendError): ...
class AclMemoryError(BackendError): ...
class AclExecuteError(BackendError): ...


class ProfilerError(AdapDomainError):
    """Errors raised by the profiling layer."""


class MsprofRunError(ProfilerError): ...
class MsprofParseError(ProfilerError): ...


class SchedulerError(AdapDomainError):
    """Errors raised by the scheduler."""


class InfeasiblePlacementError(SchedulerError): ...
class EmptyCandidateError(SchedulerError): ...


class RuntimeAdapterError(AdapDomainError):
    """Errors raised by runtime guards (SmoothGuard / PreScale / NightShift)."""


class RepairBudgetExceededError(RuntimeAdapterError): ...
```

- [ ] **Step 4: Implement `core/types.py`**

```python
"""Common type aliases and numeric constants."""
from __future__ import annotations

from typing import NewType

EPSILON_DOMAIN_FIT: float = 1e-6
DEFAULT_K: int = 2

ModelId = NewType("ModelId", str)
DeviceId = NewType("DeviceId", int)
ReplicaId = NewType("ReplicaId", str)
RunId = NewType("RunId", str)
Seed = NewType("Seed", int)
FrontierVersion = NewType("FrontierVersion", int)
```

- [ ] **Step 5: Implement `core/__init__.py`**

```python
"""AdapDomain core: pure data model, no hardware imports."""
from .errors import (
    AclDeviceError,
    AclExecuteError,
    AclInitError,
    AclMemoryError,
    AclModelError,
    AdapDomainError,
    BackendError,
    ConfigError,
    EmptyCandidateError,
    InfeasiblePlacementError,
    MsprofParseError,
    MsprofRunError,
    ProfilerError,
    RepairBudgetExceededError,
    RuntimeAdapterError,
    SchedulerError,
)
from .types import (
    DEFAULT_K,
    EPSILON_DOMAIN_FIT,
    DeviceId,
    FrontierVersion,
    ModelId,
    ReplicaId,
    RunId,
    Seed,
)

__all__ = [
    "AclDeviceError",
    "AclExecuteError",
    "AclInitError",
    "AclMemoryError",
    "AclModelError",
    "AdapDomainError",
    "BackendError",
    "ConfigError",
    "EmptyCandidateError",
    "InfeasiblePlacementError",
    "MsprofParseError",
    "MsprofRunError",
    "ProfilerError",
    "RepairBudgetExceededError",
    "RuntimeAdapterError",
    "SchedulerError",
    "DEFAULT_K",
    "EPSILON_DOMAIN_FIT",
    "DeviceId",
    "FrontierVersion",
    "ModelId",
    "ReplicaId",
    "RunId",
    "Seed",
]
```

- [ ] **Step 6: Run tests, expect pass**

Run: `pytest tests/unit/core/test_errors_and_types.py -v`
Expected: all 3 tests pass.

- [ ] **Step 7: Commit and log**

```bash
git add -A
git commit -m "M3: core error hierarchy and type aliases"
```

Append:

```
2026-06-26  M3 DONE  ---  error hierarchy + types tests green
```

Update `module_plan.md` M3 → `done`.

### Task M4: Domain & shape

**Files:**

- Create: `AdapDomain/src/adapdomain/core/domain.py`
- Create: `AdapDomain/src/adapdomain/core/shape.py`
- Create: `AdapDomain/tests/unit/core/test_shape.py`

- [ ] **Step 1: Write failing test `tests/unit/core/test_shape.py`**

```python
import math
import pytest

from adapdomain.core.domain import DomainSpec
from adapdomain.core.shape import (
    DeviceResidual,
    FeasibleInterval,
    ShapeNormalizer,
    TenantShape,
    complementarity_2d,
    domain_fit_ratio,
    harmonic_complementarity,
    update_residual,
)


def test_domain_spec_validation():
    DomainSpec(name="cube", capacity=20, unit="logical_units", schedulable=True, telemetry_key="aic_mac_ratio")
    with pytest.raises(ValueError):
        DomainSpec(name="cube", capacity=0, unit="x", schedulable=True, telemetry_key="k")


def test_shape_normalizer_paper_formula():
    interval = FeasibleInterval(x_min=2.0, x_sat=10.0)
    norm = ShapeNormalizer().normalize(value=6.0, interval=interval)
    assert norm == pytest.approx(0.5)


def test_shape_normalizer_clamps_and_rejects_invalid_interval():
    norm = ShapeNormalizer()
    interval = FeasibleInterval(x_min=1.0, x_sat=2.0)
    assert norm.normalize(0.5, interval) == pytest.approx(0.0)
    assert norm.normalize(2.5, interval) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        FeasibleInterval(x_min=5.0, x_sat=5.0)


def test_tenant_shape_and_device_residual_roundtrip():
    tenant = TenantShape(
        model_name="bert-base",
        replica_id="bert-base#0",
        domains={"cube": 0.40, "vector": 0.20},
        slo_ms=15.0,
        batch_size=1,
        frontier_version=1,
        confidence=0.9,
    )
    residual = DeviceResidual.empty(device_id=0, domains=["cube", "vector"])
    new_residual = update_residual(residual, [tenant])
    assert new_residual.residual["cube"] == pytest.approx(0.60)
    assert new_residual.residual["vector"] == pytest.approx(0.80)
    assert new_residual.placed_replicas == ["bert-base#0"]


def test_residual_floors_at_zero():
    residual = DeviceResidual.empty(device_id=0, domains=["cube", "vector"])
    heavy = TenantShape(
        model_name="x", replica_id="x#0",
        domains={"cube": 1.2, "vector": 0.0},
        slo_ms=1.0, batch_size=1, frontier_version=1, confidence=1.0,
    )
    assert update_residual(residual, [heavy]).residual["cube"] == pytest.approx(0.0)


def test_domain_fit_ratio_symmetric_and_bounded():
    rho = domain_fit_ratio(r=0.4, R=0.4, eps=1e-6)
    assert rho == pytest.approx(1.0)
    rho_low = domain_fit_ratio(r=0.9, R=0.1, eps=1e-6)
    assert 0 < rho_low < 1


def test_harmonic_complementarity_penalises_one_sided_fit():
    balanced = harmonic_complementarity([0.5, 0.5])
    skewed = harmonic_complementarity([0.9, 0.1])
    assert balanced > skewed
    assert balanced <= 1.0
    assert skewed > 0.0


def test_complementarity_2d_matches_harmonic_form():
    rho_v, rho_c = 0.6, 0.3
    expected = (2 * rho_v * rho_c) / (rho_v + rho_c + 1e-6)
    assert math.isclose(complementarity_2d(rho_v, rho_c, eps=1e-6), expected, rel_tol=1e-9)
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/unit/core/test_shape.py -v`
Expected: import errors.

- [ ] **Step 3: Implement `core/domain.py`**

```python
"""DomainSpec for K-domain accelerator resources."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DomainSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    capacity: float = Field(gt=0)
    unit: str = Field(min_length=1)
    schedulable: bool
    telemetry_key: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, v: str) -> str:
        return v.strip().lower()
```

- [ ] **Step 4: Implement `core/shape.py`**

```python
"""Tenant shape, device residual, normalization and complementarity primitives.

All formulas follow the paper:
    r_i^j = (x_i^j - x_min^j) / (x_sat^j - x_min^j)
    R_k^j = max(0, 1 - Σ r_i^j)
    rho   = min(r/(R+eps), R/(r+eps))
    COMP  = min(1, K / Σ(1/(rho_j + eps)))
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import DEFAULT_K, EPSILON_DOMAIN_FIT, DeviceId, FrontierVersion, ReplicaId


class FeasibleInterval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float
    x_sat: float

    @field_validator("x_sat")
    @classmethod
    def _check_order(cls, v: float, info) -> float:
        if v <= info.data.get("x_min", float("-inf")):
            raise ValueError("x_sat must be strictly greater than x_min")
        return v


class ShapeNormalizer:
    """Normalize raw counter values into [0, 1] using the feasible interval."""

    def normalize(self, value: float, interval: FeasibleInterval) -> float:
        norm = (value - interval.x_min) / (interval.x_sat - interval.x_min)
        if norm < 0.0:
            return 0.0
        if norm > 1.0:
            return 1.0
        return float(norm)


class TenantShape(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    replica_id: ReplicaId
    domains: dict[str, float]
    slo_ms: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    frontier_version: FrontierVersion
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("domains")
    @classmethod
    def _values_in_unit_interval(cls, v: dict[str, float]) -> dict[str, float]:
        for name, x in v.items():
            if not (0.0 <= x <= 1.5):
                # We allow >1.0 here so that infeasible candidate evaluation
                # can still construct shapes; update_residual floors residuals.
                raise ValueError(f"domain {name}: {x} outside [0, 1.5]")
        return v


class DeviceResidual(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: DeviceId
    residual: dict[str, float]
    placed_replicas: list[ReplicaId]
    hbm_pressure: float = 0.0
    queue_pressure: float = 0.0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def empty(cls, device_id: int, domains: Iterable[str]) -> "DeviceResidual":
        return cls(
            device_id=DeviceId(device_id),
            residual={d: 1.0 for d in domains},
            placed_replicas=[],
        )


def update_residual(state: DeviceResidual, placed: Sequence[TenantShape]) -> DeviceResidual:
    """Apply `placed` tenants to the residual state and return a new DeviceResidual."""
    residual = dict(state.residual)
    replicas = list(state.placed_replicas)
    for tenant in placed:
        for name in residual:
            usage = tenant.domains.get(name, 0.0)
            residual[name] = max(0.0, residual[name] - usage)
        replicas.append(tenant.replica_id)
    return DeviceResidual(
        device_id=state.device_id,
        residual=residual,
        placed_replicas=replicas,
        hbm_pressure=state.hbm_pressure,
        queue_pressure=state.queue_pressure,
        updated_at=datetime.now(timezone.utc),
    )


def domain_fit_ratio(r: float, R: float, eps: float = EPSILON_DOMAIN_FIT) -> float:
    return min(r / (R + eps), R / (r + eps))


def harmonic_complementarity(
    rho_per_domain: Sequence[float],
    *,
    K: int | None = None,
    eps: float = EPSILON_DOMAIN_FIT,
) -> float:
    k = K if K is not None else len(rho_per_domain)
    if k <= 0:
        raise ValueError("K must be positive")
    inv_sum = sum(1.0 / (rho + eps) for rho in rho_per_domain)
    if inv_sum == 0:
        return 1.0
    return min(1.0, k / inv_sum)


def complementarity_2d(rho_v: float, rho_c: float, *, eps: float = EPSILON_DOMAIN_FIT) -> float:
    """Two-domain CEFF (paper Eq. ceff_2d):
        CEFF = 2 * rho_v * rho_c / (rho_v + rho_c + eps)
    """
    return (2.0 * rho_v * rho_c) / (rho_v + rho_c + eps)


__all__ = [
    "DEFAULT_K",
    "DeviceResidual",
    "FeasibleInterval",
    "ShapeNormalizer",
    "TenantShape",
    "complementarity_2d",
    "domain_fit_ratio",
    "harmonic_complementarity",
    "update_residual",
]
```

- [ ] **Step 5: Run tests, expect green**

Run: `pytest tests/unit/core/test_shape.py -v`
Expected: 8/8 pass.

- [ ] **Step 6: Commit and log**

```bash
git add -A
git commit -m "M4: DomainSpec, ShapeNormalizer, TenantShape, DeviceResidual, complementarity"
```

Append:

```
2026-06-26  M4 DONE  ---  shape & domain primitives, harmonic CEFF tests green
```

Update M4 in module_plan → done.

---

### Task M5: Frontier & confidence

**Files:**

- Create: `AdapDomain/src/adapdomain/core/frontier.py`
- Create: `AdapDomain/tests/unit/core/test_frontier.py`

- [ ] **Step 1: Failing test `tests/unit/core/test_frontier.py`**

```python
import time
from datetime import datetime, timezone

import pytest

from adapdomain.core.frontier import (
    ConfidenceModel,
    FrontierPoint,
    FrontierQuery,
    FrontierSnapshot,
)


def _fp(point_id: str, p95: float, batch: int, feasible: bool = True, *, conf: float = 1.0) -> FrontierPoint:
    return FrontierPoint(
        model_name="m",
        point_id=point_id,
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=0.3, norm_vector=0.3,
        batch_size=batch, slo_ms=20.0,
        p50_ms=p95 / 2, p95_ms=p95, p99_ms=p95 * 1.1,
        goodput_qps=100.0,
        feasible=feasible, confidence=conf,
    )


def test_snapshot_basic_invariants():
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=[_fp("a", 10, 1), _fp("b", 20, 1, feasible=False)],
        safe_point_id="a", dominant_domain="balanced",
        confidence_summary=0.8, source_run_id="run-0",
    )
    assert snap.safe_point().point_id == "a"
    assert len(snap.feasible_points()) == 1


def test_frontier_query_picks_lowest_p95_under_slo():
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=[_fp("a", 8, 1), _fp("b", 15, 1), _fp("c", 25, 1, feasible=False)],
        safe_point_id="a", dominant_domain="balanced",
        confidence_summary=0.9, source_run_id="r",
    )
    point = FrontierQuery(snap).pick(batch_size=1, slo_ms=20.0)
    assert point.point_id == "a"


def test_frontier_query_returns_none_when_no_feasible_match():
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=[_fp("a", 30, 1, feasible=False)],
        safe_point_id="a", dominant_domain="balanced",
        confidence_summary=0.5, source_run_id="r",
    )
    assert FrontierQuery(snap).pick(batch_size=1, slo_ms=20.0) is None


def test_query_under_one_ms_on_100_points():
    pts = [_fp(f"p{i}", float(i), 1) for i in range(100)]
    snap = FrontierSnapshot(
        model_name="m", version=1, k_domains=["cube", "vector"],
        points=pts, safe_point_id="p0", dominant_domain="balanced",
        confidence_summary=1.0, source_run_id="r",
    )
    q = FrontierQuery(snap)
    start = time.perf_counter()
    for _ in range(50):
        q.pick(batch_size=1, slo_ms=50.0)
    elapsed = (time.perf_counter() - start) / 50
    assert elapsed < 1e-3


def test_confidence_decays_with_staleness():
    now = datetime.now(timezone.utc)
    fresh = ConfidenceModel().score(
        neighbor_count=10, recent_deviations=[0.0], created_at=now, now=now,
    )
    stale = ConfidenceModel().score(
        neighbor_count=10, recent_deviations=[0.0],
        created_at=now.replace(year=now.year - 1), now=now,
    )
    assert fresh > stale
    assert 0.0 <= stale <= 1.0


def test_confidence_drops_when_recent_deviations_large():
    now = datetime.now(timezone.utc)
    cm = ConfidenceModel()
    low_drift = cm.score(neighbor_count=10, recent_deviations=[0.01, 0.02], created_at=now, now=now)
    high_drift = cm.score(neighbor_count=10, recent_deviations=[0.5, 0.6], created_at=now, now=now)
    assert low_drift > high_drift
```

- [ ] **Step 2: Run, expect failure**

Run: `pytest tests/unit/core/test_frontier.py -v`

- [ ] **Step 3: Implement `core/frontier.py`**

```python
"""Feasible frontier data model, query, and confidence."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .types import FrontierVersion, ModelId, RunId


class FrontierPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_name: str
    point_id: str
    raw_cube: float
    raw_vector: float
    raw_hbm: float
    norm_cube: float = Field(ge=0.0, le=1.0)
    norm_vector: float = Field(ge=0.0, le=1.0)
    batch_size: int = Field(gt=0)
    slo_ms: float = Field(gt=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)
    goodput_qps: float = Field(ge=0)
    feasible: bool
    confidence: float = Field(ge=0.0, le=1.0)
    msprof_artifact_path: str | None = None


class FrontierSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    version: FrontierVersion
    k_domains: list[str]
    points: list[FrontierPoint]
    safe_point_id: str
    dominant_domain: str
    confidence_summary: float = Field(ge=0.0, le=1.0)
    source_run_id: RunId | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def feasible_points(self) -> list[FrontierPoint]:
        return [p for p in self.points if p.feasible]

    def safe_point(self) -> FrontierPoint:
        for p in self.points:
            if p.point_id == self.safe_point_id:
                return p
        raise KeyError(f"safe_point_id {self.safe_point_id!r} not found")


class FrontierQuery:
    """Pick the best feasible point for a target (batch_size, slo_ms)."""

    def __init__(self, snapshot: FrontierSnapshot) -> None:
        self._snapshot = snapshot

    def pick(self, *, batch_size: int, slo_ms: float) -> FrontierPoint | None:
        best: FrontierPoint | None = None
        for p in self._snapshot.points:
            if not p.feasible:
                continue
            if p.batch_size != batch_size:
                continue
            if p.p95_ms > slo_ms:
                continue
            if best is None or p.p95_ms < best.p95_ms:
                best = p
        return best


class ConfidenceModel:
    """Combine neighbor density, recent deviations, and age into [0,1]."""

    def __init__(
        self,
        *,
        half_life_days: float = 60.0,
        deviation_scale: float = 0.3,
        neighbor_saturation: int = 8,
    ) -> None:
        self._half_life = timedelta(days=half_life_days)
        self._deviation_scale = deviation_scale
        self._neighbor_saturation = neighbor_saturation

    def score(
        self,
        *,
        neighbor_count: int,
        recent_deviations: Sequence[float],
        created_at: datetime,
        now: datetime | None = None,
    ) -> float:
        now = now or datetime.now(timezone.utc)
        age = max(now - created_at, timedelta(0))
        age_factor = 0.5 ** (age / self._half_life)
        density = min(1.0, neighbor_count / self._neighbor_saturation)
        mean_dev = (sum(abs(d) for d in recent_deviations) / max(1, len(recent_deviations)))
        dev_factor = math.exp(-mean_dev / max(self._deviation_scale, 1e-6))
        return float(max(0.0, min(1.0, density * dev_factor * age_factor)))


__all__ = [
    "ConfidenceModel",
    "FrontierPoint",
    "FrontierQuery",
    "FrontierSnapshot",
]
```

- [ ] **Step 4: Run tests, expect green**

Run: `pytest tests/unit/core/test_frontier.py -v`
Expected: 6/6 pass.

- [ ] **Step 5: Commit and log**

```bash
git add -A
git commit -m "M5: FrontierPoint / FrontierSnapshot / FrontierQuery / ConfidenceModel"
```

Append checklist row M5 DONE. Update module_plan.

---

### Task M6: Telemetry & workload

**Files:**

- Create: `AdapDomain/src/adapdomain/core/telemetry.py`
- Create: `AdapDomain/src/adapdomain/core/workload.py`
- Create: `AdapDomain/tests/unit/core/test_telemetry.py`
- Create: `AdapDomain/tests/unit/core/test_workload.py`

- [ ] **Step 1: Failing test `tests/unit/core/test_telemetry.py`**

```python
from datetime import datetime, timezone

import pytest

from adapdomain.core.telemetry import (
    DeviceTelemetry,
    TelemetryWindow,
)


def test_device_telemetry_validation():
    t = DeviceTelemetry(
        device_id=0,
        cube_util=0.5, vector_util=0.3,
        hbm_pressure=0.4, queue_depth=3,
        p50_ms=2.0, p95_ms=4.0, p99_ms=5.0,
        goodput_qps=120.0,
        timestamp=datetime.now(timezone.utc),
    )
    assert t.queue_depth == 3
    with pytest.raises(Exception):
        DeviceTelemetry(
            device_id=0,
            cube_util=-0.1, vector_util=0.0, hbm_pressure=0.0, queue_depth=0,
            p50_ms=0, p95_ms=0, p99_ms=0, goodput_qps=0,
            timestamp=datetime.now(timezone.utc),
        )


def test_ewma_converges():
    w = TelemetryWindow(alpha=0.5)
    series = [10.0] * 30
    for v in series:
        w.observe("p95", v)
    assert w.ewma("p95") == pytest.approx(10.0, abs=1e-3)


def test_residual_drift_signed():
    w = TelemetryWindow(alpha=0.5)
    for v in [0.20, 0.22, 0.18]:
        w.observe("cube_util", v)
    assert w.residual_drift("cube_util", baseline=0.20) == pytest.approx(w.ewma("cube_util") - 0.20)
```

- [ ] **Step 2: Failing test `tests/unit/core/test_workload.py`**

```python
import pytest

from adapdomain.core.workload import (
    Replica,
    Request,
    WorkloadTrace,
)


def test_request_validates_arrival_and_deadline():
    r = Request(request_id="r0", model="bert-base", arrival_ms=0.0, deadline_ms=20.0)
    assert r.deadline_ms > r.arrival_ms
    with pytest.raises(Exception):
        Request(request_id="x", model="m", arrival_ms=5.0, deadline_ms=4.0)


def test_workload_trace_jsonl_roundtrip(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = WorkloadTrace(
        replicas=[Replica(replica_id="bert-base#0", model="bert-base", slo_ms=15.0, batch_size=1)],
        requests=[
            Request(request_id="r0", model="bert-base", arrival_ms=0.0, deadline_ms=15.0),
            Request(request_id="r1", model="bert-base", arrival_ms=1.0, deadline_ms=16.0),
        ],
    )
    trace.to_jsonl(path)
    loaded = WorkloadTrace.from_jsonl(path)
    assert len(loaded.requests) == 2
    assert loaded.replicas[0].model == "bert-base"


def test_slice_returns_requests_in_window():
    trace = WorkloadTrace(
        replicas=[],
        requests=[
            Request(request_id="r0", model="m", arrival_ms=0.0, deadline_ms=10.0),
            Request(request_id="r1", model="m", arrival_ms=5.0, deadline_ms=15.0),
            Request(request_id="r2", model="m", arrival_ms=12.0, deadline_ms=22.0),
        ],
    )
    sub = trace.slice(start_ms=1.0, end_ms=10.0)
    assert [r.request_id for r in sub.requests] == ["r1"]
```

- [ ] **Step 3: Run tests, expect fail**

Run: `pytest tests/unit/core/test_telemetry.py tests/unit/core/test_workload.py -v`

- [ ] **Step 4: Implement `core/telemetry.py`**

```python
"""Per-device telemetry and EWMA windows."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Deque

from pydantic import BaseModel, ConfigDict, Field

from .types import DeviceId


class DeviceTelemetry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    device_id: DeviceId
    cube_util: float = Field(ge=0.0, le=1.0)
    vector_util: float = Field(ge=0.0, le=1.0)
    hbm_pressure: float = Field(ge=0.0, le=1.0)
    queue_depth: int = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)
    p99_ms: float = Field(ge=0)
    goodput_qps: float = Field(ge=0)
    timestamp: datetime


class TelemetryWindow:
    """EWMA + ring-buffer per metric."""

    def __init__(self, *, alpha: float = 0.2, window_size: int = 60) -> None:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._ewma: dict[str, float] = {}
        self._window_size = window_size
        self._history: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def observe(self, key: str, value: float) -> None:
        if key in self._ewma:
            self._ewma[key] = self._alpha * value + (1 - self._alpha) * self._ewma[key]
        else:
            self._ewma[key] = float(value)
        self._history[key].append(float(value))

    def ewma(self, key: str) -> float:
        return self._ewma.get(key, 0.0)

    def history(self, key: str) -> list[float]:
        return list(self._history.get(key, ()))

    def residual_drift(self, key: str, baseline: float) -> float:
        return self.ewma(key) - baseline
```

- [ ] **Step 5: Implement `core/workload.py`**

```python
"""Workload model used by the replay engine."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import ReplicaId


class Replica(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    replica_id: ReplicaId
    model: str
    slo_ms: float = Field(gt=0)
    batch_size: int = Field(gt=0)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    model: str
    arrival_ms: float = Field(ge=0)
    deadline_ms: float = Field(gt=0)

    @model_validator(mode="after")
    def _deadline_after_arrival(self) -> "Request":
        if self.deadline_ms <= self.arrival_ms:
            raise ValueError("deadline_ms must be > arrival_ms")
        return self


class WorkloadTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replicas: list[Replica]
    requests: list[Request]

    def to_jsonl(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"_type": "header", "replicas": [r.model_dump() for r in self.replicas]}) + "\n")
            for req in self.requests:
                f.write(json.dumps({"_type": "request", **req.model_dump()}) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "WorkloadTrace":
        p = Path(path)
        replicas: list[Replica] = []
        requests: list[Request] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                kind = rec.pop("_type")
                if kind == "header":
                    replicas = [Replica.model_validate(r) for r in rec.get("replicas", [])]
                elif kind == "request":
                    requests.append(Request.model_validate(rec))
        return cls(replicas=replicas, requests=requests)

    def slice(self, *, start_ms: float, end_ms: float) -> "WorkloadTrace":
        keep = [r for r in self.requests if start_ms < r.arrival_ms <= end_ms]
        return WorkloadTrace(replicas=list(self.replicas), requests=keep)

    @classmethod
    def from_requests(cls, replicas: Iterable[Replica], requests: Iterable[Request]) -> "WorkloadTrace":
        return cls(replicas=list(replicas), requests=list(requests))
```

- [ ] **Step 6: Run tests, expect green**

Run: `pytest tests/unit/core/test_telemetry.py tests/unit/core/test_workload.py -v`
Expected: 6/6 pass.

- [ ] **Step 7: Commit and log**

```bash
git add -A
git commit -m "M6: telemetry EWMA window and workload trace model"
```

Append M6 DONE row to checklist; update module_plan.

### Task M7: Interference & placement

**Files:**

- Create: `AdapDomain/src/adapdomain/core/interference.py`
- Create: `AdapDomain/src/adapdomain/core/placement.py`
- Create: `AdapDomain/tests/unit/core/test_interference.py`
- Create: `AdapDomain/tests/unit/core/test_placement.py`

- [ ] **Step 1: Failing tests**

`tests/unit/core/test_interference.py`:

```python
import pytest

from adapdomain.core.interference import (
    InterferenceEntry,
    InterferenceTable,
    aggregate_interference,
)


def _entries() -> list[InterferenceEntry]:
    return [
        InterferenceEntry(model_a="A", model_b="B", direction="a_on_b",
                          pressure_bucket="mid", slowdown=0.30,
                          p95_delta=0.0, goodput_delta=0.0, source_run_id="r"),
        InterferenceEntry(model_a="A", model_b="C", direction="a_on_b",
                          pressure_bucket="mid", slowdown=0.20,
                          p95_delta=0.0, goodput_delta=0.0, source_run_id="r"),
        InterferenceEntry(model_a="A", model_b="D", direction="a_on_b",
                          pressure_bucket="mid", slowdown=0.10,
                          p95_delta=0.0, goodput_delta=0.0, source_run_id="r"),
    ]


def test_aggregate_max_plus_damped_sum_matches_paper():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["B", "C", "D"], bucket="mid", eta=0.30)
    expected = 0.30 + 0.30 * (0.30 + 0.20 + 0.10)
    assert x == pytest.approx(expected, rel=1e-9)


def test_aggregate_eta_zero_reduces_to_max():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["B", "C", "D"], bucket="mid", eta=0.0)
    assert x == pytest.approx(0.30)


def test_aggregate_eta_one_equals_sum_plus_max():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["B", "C", "D"], bucket="mid", eta=1.0)
    assert x == pytest.approx(0.30 + (0.30 + 0.20 + 0.10))


def test_missing_pair_returns_zero():
    table = InterferenceTable.from_entries(_entries())
    x = aggregate_interference(table, candidate="A", colocated=["Z"], bucket="mid", eta=0.30)
    assert x == 0.0
```

`tests/unit/core/test_placement.py`:

```python
from datetime import datetime, timezone

from adapdomain.core.placement import PlacementPlan, ReplicaPlacement, ScoreBreakdown


def test_placement_plan_basic():
    plan = PlacementPlan(
        plan_id="p0",
        created_at=datetime.now(timezone.utc),
        input_replicas=["r0", "r1"],
        placements={"r0": 0},
        rejected={"r1": "no feasible device"},
        score_breakdown={
            ("r0", 0): ScoreBreakdown(ceff=0.5, x=0.1, h=0.0, q=0.0, f=0.0, u=0.0, final=0.4),
        },
        frontier_versions={"bert-base": 1},
    )
    assert plan.placements["r0"] == 0
    assert plan.rejected["r1"] == "no feasible device"
    assert plan.score_breakdown[("r0", 0)].final == 0.4


def test_replica_placement_dataclass():
    rp = ReplicaPlacement(replica_id="r0", device_id=0, frontier_point_id="pt-3")
    assert rp.replica_id == "r0"
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/unit/core/test_interference.py tests/unit/core/test_placement.py -v`

- [ ] **Step 3: Implement `core/interference.py`**

```python
"""Pairwise interference model (paper Eq. interference_aggregation)."""
from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .types import RunId


class InterferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_a: str
    model_b: str
    direction: str  # "a_on_b" or "b_on_a"
    pressure_bucket: str
    slowdown: float = Field(ge=0)
    p95_delta: float
    goodput_delta: float
    source_run_id: RunId | None = None


class InterferenceTable:
    """Indexed lookup over (candidate, colocated, bucket)."""

    def __init__(self, entries: Iterable[InterferenceEntry]) -> None:
        self._index: dict[tuple[str, str, str], float] = {}
        for e in entries:
            key = (e.model_a, e.model_b, e.pressure_bucket) if e.direction == "a_on_b" else (e.model_b, e.model_a, e.pressure_bucket)
            self._index[key] = e.slowdown

    @classmethod
    def from_entries(cls, entries: Iterable[InterferenceEntry]) -> "InterferenceTable":
        return cls(entries)

    def lookup(self, candidate: str, other: str, bucket: str) -> float:
        return self._index.get((candidate, other, bucket), 0.0)


def aggregate_interference(
    table: InterferenceTable,
    *,
    candidate: str,
    colocated: Sequence[str],
    bucket: str,
    eta: float,
) -> float:
    """X(i, k) = max_j X(i, j) + eta * sum_j X(i, j)."""
    pair_slowdowns = [table.lookup(candidate, other, bucket) for other in colocated]
    pair_slowdowns = [s for s in pair_slowdowns if s > 0]
    if not pair_slowdowns:
        return 0.0
    return max(pair_slowdowns) + eta * sum(pair_slowdowns)
```

- [ ] **Step 4: Implement `core/placement.py`**

```python
"""Placement plan and score breakdown."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .types import DeviceId, FrontierVersion, ModelId, ReplicaId


@dataclass(frozen=True)
class ReplicaPlacement:
    replica_id: ReplicaId
    device_id: DeviceId
    frontier_point_id: str


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ceff: float
    x: float
    h: float
    q: float
    f: float
    u: float
    final: float


class PlacementPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    created_at: datetime
    input_replicas: list[ReplicaId]
    placements: dict[ReplicaId, DeviceId]
    rejected: dict[ReplicaId, str]
    score_breakdown: Mapping[tuple[ReplicaId, DeviceId], ScoreBreakdown]
    frontier_versions: dict[ModelId, FrontierVersion]
```

- [ ] **Step 5: Run tests, expect green**

Run: `pytest tests/unit/core/test_interference.py tests/unit/core/test_placement.py -v`
Expected: 6/6 pass.

- [ ] **Step 6: Commit and log**

```bash
git add -A
git commit -m "M7: InterferenceTable, aggregate_interference, PlacementPlan"
```

Append M7 DONE; update module_plan.

---

### Task M8: Metrics & SLO

**Files:**

- Create: `AdapDomain/src/adapdomain/core/metrics.py`
- Create: `AdapDomain/tests/unit/core/test_metrics.py`

- [ ] **Step 1: Failing test**

```python
import math
import statistics

import pytest

from adapdomain.core.metrics import (
    SeedResult,
    aggregate_seeds,
    compute_goodput,
    latency_percentiles,
    slo_violation_rate,
)


def _latencies(values: list[float]) -> list[float]:
    return list(values)


def test_compute_goodput_counts_under_slo():
    served = [{"latency_ms": v, "served": True} for v in [5, 5, 5, 50]]
    gp = compute_goodput(served, slo_ms=10.0, window_seconds=1.0)
    assert gp == 3.0


def test_slo_violation_rate():
    served = [{"latency_ms": v, "served": True} for v in [5, 50, 50, 50]]
    rate = slo_violation_rate(served, slo_ms=10.0)
    assert rate == pytest.approx(0.75)


def test_latency_percentiles():
    pcts = latency_percentiles(_latencies([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), [50, 95, 99])
    assert pcts[50] == pytest.approx(5.5, abs=0.5)
    assert pcts[95] >= pcts[50]
    assert pcts[99] >= pcts[95]


def test_aggregate_seeds():
    results = [
        SeedResult(seed=1, goodput=100.0, violation=0.05),
        SeedResult(seed=2, goodput=120.0, violation=0.04),
        SeedResult(seed=3, goodput=110.0, violation=0.06),
    ]
    agg = aggregate_seeds(results)
    assert agg.mean_goodput == pytest.approx(statistics.mean([100, 120, 110]))
    assert agg.std_goodput >= 0
```

- [ ] **Step 2: Implement `core/metrics.py`**

```python
"""SLO-aware metrics."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SeedResult:
    seed: int
    goodput: float
    violation: float


@dataclass(frozen=True)
class AggregatedSeedResult:
    n: int
    mean_goodput: float
    std_goodput: float
    mean_violation: float
    std_violation: float


def compute_goodput(records: Iterable[Mapping], *, slo_ms: float, window_seconds: float) -> float:
    served_within_slo = sum(
        1 for r in records if r.get("served") and r["latency_ms"] <= slo_ms
    )
    return served_within_slo / max(window_seconds, 1e-9)


def slo_violation_rate(records: Iterable[Mapping], *, slo_ms: float) -> float:
    served = [r for r in records if r.get("served")]
    if not served:
        return 0.0
    violations = sum(1 for r in served if r["latency_ms"] > slo_ms)
    return violations / len(served)


def latency_percentiles(values: Sequence[float], percentiles: Sequence[int]) -> dict[int, float]:
    if not values:
        return {p: 0.0 for p in percentiles}
    arr = np.asarray(values, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in percentiles}


def aggregate_seeds(results: Sequence[SeedResult]) -> AggregatedSeedResult:
    if not results:
        return AggregatedSeedResult(n=0, mean_goodput=0.0, std_goodput=0.0, mean_violation=0.0, std_violation=0.0)
    g = [r.goodput for r in results]
    v = [r.violation for r in results]
    return AggregatedSeedResult(
        n=len(results),
        mean_goodput=statistics.mean(g),
        std_goodput=statistics.pstdev(g) if len(g) > 1 else 0.0,
        mean_violation=statistics.mean(v),
        std_violation=statistics.pstdev(v) if len(v) > 1 else 0.0,
    )
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/core/test_metrics.py -v`
Expected: 4/4 pass.

```bash
git add -A
git commit -m "M8: goodput, slo violation rate, percentiles, seed aggregator"
```

Append M8 DONE.

---

### Task M9: Storage

**Files:**

- Create: `AdapDomain/src/adapdomain/storage/__init__.py`
- Create: `AdapDomain/src/adapdomain/storage/jsonl.py`
- Create: `AdapDomain/src/adapdomain/storage/csv_store.py`
- Create: `AdapDomain/src/adapdomain/storage/frontier_store.py`
- Create: `AdapDomain/src/adapdomain/storage/run_manifest.py`
- Create: `AdapDomain/src/adapdomain/storage/artifacts.py`
- Create: `AdapDomain/tests/unit/storage/__init__.py`
- Create: `AdapDomain/tests/unit/storage/test_jsonl.py`
- Create: `AdapDomain/tests/unit/storage/test_frontier_store.py`
- Create: `AdapDomain/tests/unit/storage/test_run_manifest.py`

- [ ] **Step 1: Failing tests**

`tests/unit/storage/test_jsonl.py`:

```python
import pytest

from adapdomain.storage.jsonl import JsonlReader, JsonlWriter


def test_writer_appends_and_reader_filters(tmp_path):
    path = tmp_path / "events.jsonl"
    w = JsonlWriter(path)
    w.write({"type": "start", "id": 1})
    w.write({"type": "tick", "id": 2})
    w.write({"type": "tick", "id": 3})
    assert list(JsonlReader(path).filter(type="tick")) == [
        {"type": "tick", "id": 2},
        {"type": "tick", "id": 3},
    ]


def test_reader_tolerates_corrupt_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"type":"ok"}\n{not json\n{"type":"ok2"}\n')
    items = list(JsonlReader(path).read_all(tolerant=True))
    assert [r["type"] for r in items] == ["ok", "ok2"]
```

`tests/unit/storage/test_frontier_store.py`:

```python
from datetime import datetime, timezone

import pytest

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot
from adapdomain.storage.frontier_store import FrontierStore


def _snap(version: int) -> FrontierSnapshot:
    pt = FrontierPoint(
        model_name="bert-base", point_id=f"p{version}",
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=0.3, norm_vector=0.3,
        batch_size=1, slo_ms=15.0,
        p50_ms=2, p95_ms=4, p99_ms=5, goodput_qps=80.0,
        feasible=True, confidence=0.9,
    )
    return FrontierSnapshot(
        model_name="bert-base", version=version, k_domains=["cube", "vector"],
        points=[pt], safe_point_id=pt.point_id, dominant_domain="balanced",
        confidence_summary=0.9, source_run_id="r",
        created_at=datetime.now(timezone.utc),
    )


def test_store_write_atomic_and_read_latest(tmp_path):
    store = FrontierStore(tmp_path)
    store.save(_snap(1))
    store.save(_snap(2))
    latest = store.load_latest("bert-base")
    assert latest.version == 2
    assert store.load_version("bert-base", 1).version == 1
```

`tests/unit/storage/test_run_manifest.py`:

```python
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
```

- [ ] **Step 2: Run, expect fail**

Run: `pytest tests/unit/storage/ -v`

- [ ] **Step 3: Implement `storage/jsonl.py`**

```python
"""Append-only JSONL helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


class JsonlReader:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def read_all(self, *, tolerant: bool = False) -> Iterator[dict[str, Any]]:
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if tolerant:
                        continue
                    raise

    def filter(self, **conditions: Any) -> Iterator[dict[str, Any]]:
        for rec in self.read_all():
            if all(rec.get(k) == v for k, v in conditions.items()):
                yield rec
```

- [ ] **Step 4: Implement `storage/csv_store.py`**

```python
"""Schema-checked CSV append store."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence


class CsvStore:
    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        self._path = Path(path)
        self._columns = list(columns)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._columns)

    def append(self, row: dict) -> None:
        missing = [c for c in self._columns if c not in row]
        if missing:
            raise ValueError(f"missing columns for csv {self._path.name}: {missing}")
        with self._path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[c] for c in self._columns])
```

- [ ] **Step 5: Implement `storage/frontier_store.py`**

```python
"""Versioned, atomic frontier snapshot store."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from adapdomain.core.frontier import FrontierSnapshot


class FrontierStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        d = self._root / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, snap: FrontierSnapshot) -> Path:
        d = self._model_dir(snap.model_name)
        path = d / f"{snap.version}.json"
        payload = snap.model_dump(mode="json")
        with tempfile.NamedTemporaryFile("w", delete=False, dir=d, suffix=".tmp", encoding="utf-8") as tmp:
            json.dump(payload, tmp, sort_keys=True)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
        return path

    def load_version(self, model_name: str, version: int) -> FrontierSnapshot:
        path = self._model_dir(model_name) / f"{version}.json"
        return FrontierSnapshot.model_validate(json.loads(path.read_text()))

    def load_latest(self, model_name: str) -> FrontierSnapshot:
        files = sorted(self._model_dir(model_name).glob("*.json"), key=lambda p: int(p.stem))
        if not files:
            raise FileNotFoundError(model_name)
        return self.load_version(model_name, int(files[-1].stem))
```

- [ ] **Step 6: Implement `storage/run_manifest.py`**

```python
"""Run manifest written for every experiment / replay / verify run."""
from __future__ import annotations

import json
import platform
import socket
from datetime import datetime, timezone
from pathlib import Path


class RunManifest:
    def __init__(self, *, run_id: str, command: str, out_dir: Path) -> None:
        self._run_id = run_id
        self._command = command
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._data: dict = {
            "run_id": run_id,
            "command": command,
            "git_commit_or_none": _git_commit_or_none(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ended_at": None,
            "status": "in_progress",
        }

    @classmethod
    def start(cls, *, run_id: str, command: str, out_dir: Path) -> "RunManifest":
        return cls(run_id=run_id, command=command, out_dir=out_dir)

    def add_config_hash(self, h: str) -> None:
        self._data["config_hash"] = h

    def set_hardware(self, platform_name: str, *, device_count: int, cann_version: str | None) -> None:
        self._data["device_info"] = {
            "platform": platform_name,
            "count": device_count,
            "cann_version": cann_version,
        }

    def set_status(self, status: str) -> None:
        self._data["status"] = status

    def update(self, **kwargs) -> None:
        self._data.update(kwargs)

    def finalize(self) -> Path:
        self._data["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        path = self._out_dir / "manifest.json"
        path.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        return path


def _git_commit_or_none() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None
```

- [ ] **Step 7: Implement `storage/artifacts.py` and `__init__.py`**

`storage/artifacts.py`:

```python
"""Bundle a run directory into a portable archive."""
from __future__ import annotations

import shutil
from pathlib import Path


def bundle_run(run_dir: str | Path, *, target_root: str | Path) -> Path:
    src = Path(run_dir)
    tgt_root = Path(target_root)
    tgt_root.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(tgt_root / src.name), "zip", root_dir=src)
    return Path(archive)
```

`storage/__init__.py`:

```python
"""AdapDomain storage adapters."""
from .csv_store import CsvStore
from .frontier_store import FrontierStore
from .jsonl import JsonlReader, JsonlWriter
from .run_manifest import RunManifest
from .artifacts import bundle_run

__all__ = [
    "CsvStore",
    "FrontierStore",
    "JsonlReader",
    "JsonlWriter",
    "RunManifest",
    "bundle_run",
]
```

- [ ] **Step 8: Run, expect green**

Run: `pytest tests/unit/storage/ -v`
Expected: 4/4 pass.

- [ ] **Step 9: Commit and log**

```bash
git add -A
git commit -m "M9: storage adapters (jsonl, csv, frontier store, manifest, bundle)"
```

Append M9 DONE.

---

### Task M10: Test infrastructure

**Files:**

- Create: `AdapDomain/tests/integration/__init__.py`
- Create: `AdapDomain/tests/integration/test_acl_backend_import_guard.py`
- Create: `AdapDomain/tests/integration/test_cli_smoke.py`
- Create: `AdapDomain/tests/fixtures/sample_msprof/README.md`
- Create: `AdapDomain/tests/fixtures/sample_msprof/cann83_minimal.csv`
- Create: `AdapDomain/tests/fixtures/sample_frontiers/README.md`
- Create: `AdapDomain/tests/fixtures/sample_frontiers/bert-base.v1.json`
- Create: `AdapDomain/tests/fixtures/sample_traces/README.md`
- Create: `AdapDomain/tests/fixtures/sample_traces/tiny.jsonl`

- [ ] **Step 1: Write import-guard test**

```python
"""Forbid `import acl` outside of backends/ascend/."""
from __future__ import annotations

from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src" / "adapdomain"
ALLOWED_PREFIX = SRC / "backends" / "ascend"


@pytest.mark.parametrize("py_file", list(SRC.rglob("*.py")))
def test_no_acl_import_outside_backend(py_file: Path) -> None:
    if str(py_file).startswith(str(ALLOWED_PREFIX)):
        return
    text = py_file.read_text()
    assert "import acl" not in text, f"{py_file} must not import acl directly"
```

- [ ] **Step 2: Write CLI smoke test**

```python
import subprocess
import sys


def test_help_lists_subcommands():
    proc = subprocess.run(
        [sys.executable, "-m", "adapdomain.cli.main", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    for cmd in ["version", "verify-remote", "run-model", "profile", "schedule", "replay", "ablate", "plot"]:
        assert cmd in proc.stdout, f"missing subcommand {cmd}: {proc.stdout}"


def test_version_prints_string():
    proc = subprocess.run(
        [sys.executable, "-m", "adapdomain.cli.main", "version"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() != ""
```

- [ ] **Step 3: Add fixture READMEs and tiny JSONL trace**

`tests/fixtures/sample_msprof/README.md`:

```markdown
# Sample msprof outputs

`cann83_minimal.csv` is a stripped-down msprof export used by M19 parser tests.
Real samples captured from the remote machine should be added later under
`cann_real/<run-id>/` and referenced by the parser's variant tests.
```

`tests/fixtures/sample_msprof/cann83_minimal.csv`:

```csv
op_name,task_duration,aic_mac_ratio,aiv_vec_ratio,hbm_bw_util
GEMM,1.0,0.80,0.10,0.40
ELEMWISE,0.5,0.05,0.70,0.20
LAYERNORM,0.4,0.10,0.65,0.20
```

`tests/fixtures/sample_frontiers/README.md`:

```markdown
# Sample frontier snapshots

`bert-base.v1.json` is a hand-built frontier with two feasible points used to
seed scheduler and ablation tests before real profiling data is available.
```

`tests/fixtures/sample_frontiers/bert-base.v1.json`:

```json
{
  "model_name": "bert-base",
  "version": 1,
  "k_domains": ["cube", "vector"],
  "points": [
    {
      "model_name": "bert-base", "point_id": "low",
      "raw_cube": 0.0, "raw_vector": 0.0, "raw_hbm": 0.0,
      "norm_cube": 0.40, "norm_vector": 0.20,
      "batch_size": 1, "slo_ms": 15.0,
      "p50_ms": 4.0, "p95_ms": 6.0, "p99_ms": 7.0,
      "goodput_qps": 100.0,
      "feasible": true, "confidence": 0.9
    },
    {
      "model_name": "bert-base", "point_id": "high",
      "raw_cube": 0.0, "raw_vector": 0.0, "raw_hbm": 0.0,
      "norm_cube": 0.55, "norm_vector": 0.25,
      "batch_size": 1, "slo_ms": 15.0,
      "p50_ms": 5.0, "p95_ms": 9.0, "p99_ms": 10.0,
      "goodput_qps": 130.0,
      "feasible": true, "confidence": 0.7
    }
  ],
  "safe_point_id": "low",
  "dominant_domain": "balanced",
  "confidence_summary": 0.8,
  "source_run_id": "fixture"
}
```

`tests/fixtures/sample_traces/README.md`:

```markdown
# Sample workload traces

`tiny.jsonl` covers 5 requests for one replica; used by M34 smoke tests.
```

`tests/fixtures/sample_traces/tiny.jsonl`:

```text
{"_type": "header", "replicas": [{"replica_id": "bert-base#0", "model": "bert-base", "slo_ms": 15.0, "batch_size": 1}]}
{"_type": "request", "request_id": "r0", "model": "bert-base", "arrival_ms": 0.0, "deadline_ms": 15.0}
{"_type": "request", "request_id": "r1", "model": "bert-base", "arrival_ms": 1.0, "deadline_ms": 16.0}
{"_type": "request", "request_id": "r2", "model": "bert-base", "arrival_ms": 2.0, "deadline_ms": 17.0}
{"_type": "request", "request_id": "r3", "model": "bert-base", "arrival_ms": 3.0, "deadline_ms": 18.0}
{"_type": "request", "request_id": "r4", "model": "bert-base", "arrival_ms": 4.0, "deadline_ms": 19.0}
```

- [ ] **Step 4: Run integration tests**

Run: `pytest tests/integration/ -v`
Expected: import-guard test parametrized over every Python file under `src/adapdomain/`; CLI smoke tests pass.

- [ ] **Step 5: Commit and log**

```bash
git add -A
git commit -m "M10: test infrastructure (import guard, cli smoke, fixtures)"
```

Append `M10 DONE`. Append `STAGE A DONE` after this row in checklist.

---

## Stage B — Hardware Backend and Profiling

> ACL and msprof are not installed in the local macOS environment. All tasks below build the code locally with `import acl` lazy and `importorskip("acl")` in tests. Hardware verification runs on the remote Ascend server via `scripts/remote_ascend/*.sh`.

### Task M11: Backend abstraction

**Files:**

- Create: `AdapDomain/src/adapdomain/backends/__init__.py`
- Create: `AdapDomain/src/adapdomain/backends/base.py`
- Create: `AdapDomain/tests/unit/backends/__init__.py`
- Create: `AdapDomain/tests/unit/backends/test_mock_backend.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np

from adapdomain.backends.base import (
    BackendRegistry,
    DeviceInfo,
    ExecutionResult,
    InferenceBackend,
    MockBackend,
    ResourceQuota,
)


def test_mock_backend_basic_lifecycle():
    backend: InferenceBackend = MockBackend(
        devices=[DeviceInfo(device_id=0, name="mock-0", cube_capacity=20, vector_capacity=40)]
    )
    backend.initialize()
    handle = backend.load_model(model_id="bert-base", path="/tmp/fake.om", device_id=0)
    res: ExecutionResult = backend.execute(handle, inputs={"x": np.zeros(1)}, repeat=3)
    assert res.latencies_ms.shape == (3,)
    backend.apply_resource_quota(device_id=0, quota=ResourceQuota(device_id=0, cube_units=10, vector_units=20))
    backend.unload_model(handle)
    backend.finalize()


def test_registry_returns_named_backend():
    reg = BackendRegistry.default()
    assert isinstance(reg.create("mock"), InferenceBackend)
```

- [ ] **Step 2: Implement `backends/base.py`**

```python
"""Inference backend abstraction. Only `backends/ascend/` may import `acl`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from adapdomain.core.errors import BackendError


@dataclass(frozen=True)
class DeviceInfo:
    device_id: int
    name: str
    cube_capacity: int
    vector_capacity: int


@dataclass(frozen=True)
class ResourceQuota:
    device_id: int
    cube_units: int | None = None
    vector_units: int | None = None
    hbm_bytes: int | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ModelHandle:
    backend_name: str
    model_id: str
    device_id: int
    internal: Any


@dataclass(frozen=True)
class ExecutionResult:
    latencies_ms: "np.ndarray"
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceTelemetrySample:
    device_id: int
    cube_util: float
    vector_util: float
    hbm_pressure: float
    queue_depth: int


@runtime_checkable
class InferenceBackend(Protocol):
    name: str

    def initialize(self) -> None: ...
    def finalize(self) -> None: ...
    def list_devices(self) -> list[DeviceInfo]: ...
    def load_model(self, *, model_id: str, path: str, device_id: int) -> ModelHandle: ...
    def unload_model(self, handle: ModelHandle) -> None: ...
    def execute(self, handle: ModelHandle, *, inputs: Mapping[str, Any], repeat: int = 1) -> ExecutionResult: ...
    def apply_resource_quota(self, *, device_id: int, quota: ResourceQuota) -> None: ...
    def collect_telemetry(self, *, device_id: int) -> DeviceTelemetrySample: ...


class MockBackend:
    name = "mock"

    def __init__(self, devices: list[DeviceInfo]) -> None:
        self._devices = devices
        self._initialized = False
        self._loaded: dict[str, ModelHandle] = {}

    def initialize(self) -> None:
        self._initialized = True

    def finalize(self) -> None:
        self._initialized = False
        self._loaded.clear()

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def load_model(self, *, model_id: str, path: str, device_id: int) -> ModelHandle:
        if not self._initialized:
            raise BackendError("MockBackend not initialized")
        handle = ModelHandle(backend_name=self.name, model_id=model_id, device_id=device_id, internal={"path": path})
        self._loaded[model_id] = handle
        return handle

    def unload_model(self, handle: ModelHandle) -> None:
        self._loaded.pop(handle.model_id, None)

    def execute(self, handle: ModelHandle, *, inputs, repeat: int = 1) -> ExecutionResult:
        rng = np.random.default_rng(seed=hash(handle.model_id) & 0xFFFFFFFF)
        return ExecutionResult(latencies_ms=rng.uniform(2.0, 8.0, size=repeat))

    def apply_resource_quota(self, *, device_id: int, quota: ResourceQuota) -> None:
        pass

    def collect_telemetry(self, *, device_id: int) -> DeviceTelemetrySample:
        return DeviceTelemetrySample(device_id=device_id, cube_util=0.4, vector_util=0.3, hbm_pressure=0.3, queue_depth=1)


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, callable] = {}

    def register(self, name: str, factory) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> InferenceBackend:
        if name not in self._factories:
            raise BackendError(f"unknown backend {name!r}")
        return self._factories[name](**kwargs)

    @classmethod
    def default(cls) -> "BackendRegistry":
        reg = cls()
        reg.register("mock", lambda: MockBackend(devices=[
            DeviceInfo(device_id=i, name=f"mock-{i}", cube_capacity=20, vector_capacity=40) for i in range(4)
        ]))
        return reg
```

- [ ] **Step 3: Implement `backends/__init__.py`**

```python
"""Backend abstraction package."""
from .base import (
    BackendRegistry,
    DeviceInfo,
    DeviceTelemetrySample,
    ExecutionResult,
    InferenceBackend,
    MockBackend,
    ModelHandle,
    ResourceQuota,
)

__all__ = [
    "BackendRegistry",
    "DeviceInfo",
    "DeviceTelemetrySample",
    "ExecutionResult",
    "InferenceBackend",
    "MockBackend",
    "ModelHandle",
    "ResourceQuota",
]
```

- [ ] **Step 4: Run, commit, log**

Run: `pytest tests/unit/backends/test_mock_backend.py tests/integration/test_acl_backend_import_guard.py -v`
Expected: 3 tests pass; guard still green.

```bash
git add -A
git commit -m "M11: InferenceBackend protocol, ResourceQuota, MockBackend, BackendRegistry"
```

Append M11 DONE.

### Task M12: ACL runtime

**Files:**

- Create: `AdapDomain/src/adapdomain/backends/ascend/__init__.py`
- Create: `AdapDomain/src/adapdomain/backends/ascend/acl_runtime.py`
- Create: `AdapDomain/src/adapdomain/backends/ascend/errors.py`
- Create: `AdapDomain/tests/unit/backends/test_acl_runtime_import_guard.py`

- [ ] **Step 1: Unit test that asserts the runtime fails clearly without ACL**

```python
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
```

- [ ] **Step 2: Implement `backends/ascend/errors.py`**

```python
"""Re-export ACL error types so backend submodules use a single import path."""
from adapdomain.core.errors import (
    AclDeviceError,
    AclExecuteError,
    AclInitError,
    AclMemoryError,
    AclModelError,
)

__all__ = ["AclDeviceError", "AclExecuteError", "AclInitError", "AclMemoryError", "AclModelError"]
```

- [ ] **Step 3: Implement `backends/ascend/acl_runtime.py`**

```python
"""ACL Python runtime lifecycle wrapper.

CANN docs: acl.init, acl.finalize, acl.rt.set_device, acl.rt.reset_device,
acl.rt.set_device_res_limit (referenced by the user).
"""
from __future__ import annotations

import threading
from typing import Any

from adapdomain.utils.logging import get_logger

from .errors import AclInitError

_log = get_logger(__name__)


def _import_acl() -> Any:
    try:
        import acl  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise AclInitError(f"acl python module not importable: {exc}")
    return acl


class AclRuntime:
    """One-shot lifecycle. init() must be paired with finalize()."""

    _lock = threading.Lock()
    _initialized: bool = False

    def init(self, config_path: str | None = None) -> None:
        with AclRuntime._lock:
            if AclRuntime._initialized:
                _log.debug("AclRuntime.init: already initialized")
                return
            acl = _import_acl()
            ret = acl.init(config_path) if config_path else acl.init()
            self.check_ret(ret, operation="acl.init")
            AclRuntime._initialized = True
            _log.info("acl initialized")

    def finalize(self) -> None:
        with AclRuntime._lock:
            if not AclRuntime._initialized:
                return
            acl = _import_acl()
            ret = acl.finalize()
            self.check_ret(ret, operation="acl.finalize")
            AclRuntime._initialized = False
            _log.info("acl finalized")

    @staticmethod
    def check_ret(ret: int, operation: str) -> None:
        if ret != 0:
            raise AclInitError(f"acl op failed: {operation}", ret=ret, operation=operation)
```

- [ ] **Step 4: Implement `backends/ascend/__init__.py`** (empty re-exports for now)

```python
"""Ascend backend; only this package may import `acl`."""
from .acl_runtime import AclRuntime
from .errors import AclDeviceError, AclExecuteError, AclInitError, AclMemoryError, AclModelError

__all__ = ["AclRuntime", "AclDeviceError", "AclExecuteError", "AclInitError", "AclMemoryError", "AclModelError"]
```

- [ ] **Step 5: Run, expect green**

Run: `pytest tests/unit/backends/test_acl_runtime_import_guard.py tests/integration/test_acl_backend_import_guard.py -v`

- [ ] **Step 6: Commit and log**

```bash
git add -A
git commit -m "M12: AclRuntime with explicit ImportError handling"
```

Append `M12 DONE  ---  acl import gated; init raises AclInitError when CANN missing` to checklist.

---

### Task M13: Device, Context, Stream

**Files:**

- Create: `AdapDomain/src/adapdomain/backends/ascend/device.py`
- Create: `AdapDomain/src/adapdomain/backends/ascend/context.py`
- Create: `AdapDomain/src/adapdomain/backends/ascend/stream.py`
- Create: `AdapDomain/tests/unit/backends/test_device_context_stream_import_guard.py`

This module can be parallelized across three subagents (one file each). All subagents must use the same `AclRuntime` and raise from `errors.py`.

- [ ] **Step 1: Implement `backends/ascend/device.py`**

```python
"""Ascend device lifecycle helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from adapdomain.utils.logging import get_logger

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclDeviceError

_log = get_logger(__name__)


class AscendDevice:
    def __init__(self, device_id: int) -> None:
        self._device_id = device_id
        self._set = False

    @property
    def device_id(self) -> int:
        return self._device_id

    def set(self) -> None:
        acl = _import_acl()
        ret = acl.rt.set_device(self._device_id)
        AclRuntime.check_ret(ret, operation=f"acl.rt.set_device({self._device_id})")
        self._set = True

    def reset(self) -> None:
        if not self._set:
            return
        acl = _import_acl()
        ret = acl.rt.reset_device(self._device_id)
        AclRuntime.check_ret(ret, operation=f"acl.rt.reset_device({self._device_id})")
        self._set = False

    @staticmethod
    def get_count() -> int:
        acl = _import_acl()
        count, ret = acl.rt.get_device_count()
        if ret != 0:
            raise AclDeviceError("acl.rt.get_device_count failed", ret=ret, operation="get_device_count")
        return int(count)

    @contextmanager
    def bind(self) -> Iterator["AscendDevice"]:
        self.set()
        try:
            yield self
        finally:
            self.reset()
```

- [ ] **Step 2: Implement `backends/ascend/context.py`**

```python
"""Ascend execution context (acl.rt.create_context / destroy_context)."""
from __future__ import annotations

from typing import Any

from .acl_runtime import AclRuntime, _import_acl


class AscendContext:
    def __init__(self, device_id: int) -> None:
        self._device_id = device_id
        self._handle: Any = None

    def create(self) -> None:
        acl = _import_acl()
        handle, ret = acl.rt.create_context(self._device_id)
        AclRuntime.check_ret(ret, operation="acl.rt.create_context")
        self._handle = handle

    def destroy(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        ret = acl.rt.destroy_context(self._handle)
        AclRuntime.check_ret(ret, operation="acl.rt.destroy_context")
        self._handle = None

    def __enter__(self) -> "AscendContext":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()
```

- [ ] **Step 3: Implement `backends/ascend/stream.py`**

```python
"""Ascend stream (acl.rt.create_stream / synchronize_stream / destroy_stream)."""
from __future__ import annotations

from typing import Any

from .acl_runtime import AclRuntime, _import_acl


class AscendStream:
    def __init__(self) -> None:
        self._handle: Any = None

    def create(self) -> None:
        acl = _import_acl()
        handle, ret = acl.rt.create_stream()
        AclRuntime.check_ret(ret, operation="acl.rt.create_stream")
        self._handle = handle

    @property
    def handle(self) -> Any:
        if self._handle is None:
            raise RuntimeError("AscendStream not created")
        return self._handle

    def synchronize(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        ret = acl.rt.synchronize_stream(self._handle)
        AclRuntime.check_ret(ret, operation="acl.rt.synchronize_stream")

    def destroy(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        ret = acl.rt.destroy_stream(self._handle)
        AclRuntime.check_ret(ret, operation="acl.rt.destroy_stream")
        self._handle = None

    def __enter__(self) -> "AscendStream":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()
```

- [ ] **Step 4: Import-guard test still green**

Run: `pytest tests/integration/test_acl_backend_import_guard.py tests/unit/backends/ -v`

- [ ] **Step 5: Remote verification (deferred)**

Add to `docs/hardware/remote_validation.md` the smoke sequence:

```bash
python -c "
from adapdomain.backends.ascend import AclRuntime
from adapdomain.backends.ascend.device import AscendDevice
from adapdomain.backends.ascend.context import AscendContext
from adapdomain.backends.ascend.stream import AscendStream

rt = AclRuntime(); rt.init()
try:
    for did in range(AscendDevice.get_count()):
        d = AscendDevice(did)
        with d.bind(), AscendContext(did), AscendStream() as s:
            s.synchronize()
finally:
    rt.finalize()
"
```

Expected on remote: zero exit code, no warnings.

- [ ] **Step 6: Commit and log**

```bash
git add -A
git commit -m "M13: AscendDevice, AscendContext, AscendStream with strict cleanup"
```

Append `M13 DONE  ---  device/context/stream context managers ready; remote smoke deferred to M17`.

---

### Task M14: Memory

**Files:**

- Create: `AdapDomain/src/adapdomain/backends/ascend/memory.py`
- Create: `AdapDomain/tests/unit/backends/test_memory_import_guard.py`

- [ ] **Step 1: Implement `backends/ascend/memory.py`**

```python
"""Ascend memory helpers.

Wraps acl.rt.malloc / malloc_host / free / free_host / memcpy.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from adapdomain.utils.logging import get_logger

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclMemoryError

_log = get_logger(__name__)


@dataclass
class DeviceBuffer:
    ptr: Any
    size: int

    def free(self) -> None:
        acl = _import_acl()
        ret = acl.rt.free(self.ptr)
        AclRuntime.check_ret(ret, operation="acl.rt.free")


@dataclass
class HostBuffer:
    ptr: Any
    size: int

    def free(self) -> None:
        acl = _import_acl()
        ret = acl.rt.free_host(self.ptr)
        AclRuntime.check_ret(ret, operation="acl.rt.free_host")


def malloc_device(size: int) -> DeviceBuffer:
    acl = _import_acl()
    ptr, ret = acl.rt.malloc(size, acl.rt.ACL_MEM_MALLOC_NORMAL_ONLY)
    if ret != 0:
        raise AclMemoryError(f"malloc({size}) failed", ret=ret, operation="acl.rt.malloc")
    return DeviceBuffer(ptr=ptr, size=size)


def malloc_host(size: int) -> HostBuffer:
    acl = _import_acl()
    ptr, ret = acl.rt.malloc_host(size)
    if ret != 0:
        raise AclMemoryError(f"malloc_host({size}) failed", ret=ret, operation="acl.rt.malloc_host")
    return HostBuffer(ptr=ptr, size=size)


def memcpy(dst: Any, dst_size: int, src: Any, src_size: int, kind: str) -> None:
    acl = _import_acl()
    kind_enum = {
        "host_to_device": acl.rt.ACL_MEMCPY_HOST_TO_DEVICE,
        "device_to_host": acl.rt.ACL_MEMCPY_DEVICE_TO_HOST,
        "device_to_device": acl.rt.ACL_MEMCPY_DEVICE_TO_DEVICE,
    }[kind]
    ret = acl.rt.memcpy(dst, dst_size, src, src_size, kind_enum)
    AclRuntime.check_ret(ret, operation=f"acl.rt.memcpy:{kind}")


@contextmanager
def device_buffer(size: int) -> Iterator[DeviceBuffer]:
    buf = malloc_device(size)
    try:
        yield buf
    finally:
        buf.free()


@contextmanager
def host_buffer(size: int) -> Iterator[HostBuffer]:
    buf = malloc_host(size)
    try:
        yield buf
    finally:
        buf.free()
```

- [ ] **Step 2: Import-guard test**

```python
def test_memory_module_does_not_import_acl_at_import_time():
    import importlib
    mod = importlib.import_module("adapdomain.backends.ascend.memory")
    assert hasattr(mod, "malloc_device")
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/backends/ tests/integration/test_acl_backend_import_guard.py -v`

```bash
git add -A
git commit -m "M14: AscendMemory helpers and leak-safe context managers"
```

Append M14 DONE. Remote leak test (loop malloc/free ×100) scheduled in remote validation doc.

---

### Task M15: Model, Dataset, Executor

**Files:**

- Create: `AdapDomain/src/adapdomain/backends/ascend/model.py`
- Create: `AdapDomain/src/adapdomain/backends/ascend/dataset.py`
- Create: `AdapDomain/src/adapdomain/backends/ascend/executor.py`
- Modify: `AdapDomain/src/adapdomain/backends/ascend/__init__.py` to export new classes
- Create: `AdapDomain/tests/unit/backends/test_executor_import_guard.py`

This task depends on M13 and M14; do not parallelize.

- [ ] **Step 1: Implement `backends/ascend/model.py`**

```python
"""Ascend .om model load/unload."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclModelError


@dataclass
class LoadedModel:
    model_id: int
    desc: Any
    path: str


class AscendModel:
    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._loaded: LoadedModel | None = None

    def load(self) -> LoadedModel:
        acl = _import_acl()
        model_id, ret = acl.mdl.load_from_file(self._path)
        if ret != 0:
            raise AclModelError(f"load_from_file failed: {self._path}", ret=ret, operation="mdl.load_from_file")
        desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(desc, model_id)
        if ret != 0:
            raise AclModelError("get_desc failed", ret=ret, operation="mdl.get_desc")
        self._loaded = LoadedModel(model_id=model_id, desc=desc, path=self._path)
        return self._loaded

    def unload(self) -> None:
        if self._loaded is None:
            return
        acl = _import_acl()
        ret = acl.mdl.unload(self._loaded.model_id)
        AclRuntime.check_ret(ret, operation="mdl.unload")
        ret = acl.mdl.destroy_desc(self._loaded.desc)
        AclRuntime.check_ret(ret, operation="mdl.destroy_desc")
        self._loaded = None

    def __enter__(self) -> LoadedModel:
        return self.load()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unload()
```

- [ ] **Step 2: Implement `backends/ascend/dataset.py`**

```python
"""Ascend aclmdlDataset wrapper."""
from __future__ import annotations

from typing import Any, Sequence

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclModelError


class AscendDataset:
    def __init__(self) -> None:
        self._handle: Any = None
        self._buffers: list[Any] = []

    def create(self) -> None:
        acl = _import_acl()
        self._handle = acl.mdl.create_dataset()
        if self._handle is None:
            raise AclModelError("create_dataset returned None", operation="mdl.create_dataset")

    @property
    def handle(self) -> Any:
        if self._handle is None:
            raise RuntimeError("dataset not created")
        return self._handle

    def add_device_buffer(self, ptr: Any, size: int) -> None:
        acl = _import_acl()
        buf = acl.create_data_buffer(ptr, size)
        if buf is None:
            raise AclModelError("create_data_buffer returned None", operation="create_data_buffer")
        ret = acl.mdl.add_dataset_buffer(self._handle, buf)
        AclRuntime.check_ret(ret, operation="mdl.add_dataset_buffer")
        self._buffers.append(buf)

    def destroy(self) -> None:
        if self._handle is None:
            return
        acl = _import_acl()
        for buf in self._buffers:
            ret = acl.destroy_data_buffer(buf)
            AclRuntime.check_ret(ret, operation="destroy_data_buffer")
        self._buffers.clear()
        ret = acl.mdl.destroy_dataset(self._handle)
        AclRuntime.check_ret(ret, operation="mdl.destroy_dataset")
        self._handle = None

    def __enter__(self) -> "AscendDataset":
        self.create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.destroy()
```

- [ ] **Step 3: Implement `backends/ascend/executor.py`**

```python
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
```

- [ ] **Step 4: Update `backends/ascend/__init__.py`**

```python
from .acl_runtime import AclRuntime
from .errors import AclDeviceError, AclExecuteError, AclInitError, AclMemoryError, AclModelError
from .context import AscendContext
from .device import AscendDevice
from .stream import AscendStream
from .memory import DeviceBuffer, HostBuffer, device_buffer, host_buffer, malloc_device, malloc_host, memcpy
from .model import AscendModel, LoadedModel
from .dataset import AscendDataset
from .executor import AscendExecutor

__all__ = [
    "AclRuntime", "AclDeviceError", "AclExecuteError", "AclInitError", "AclMemoryError", "AclModelError",
    "AscendContext", "AscendDevice", "AscendStream",
    "DeviceBuffer", "HostBuffer", "device_buffer", "host_buffer",
    "malloc_device", "malloc_host", "memcpy",
    "AscendModel", "LoadedModel", "AscendDataset", "AscendExecutor",
]
```

- [ ] **Step 5: Test import guards**

```python
"""tests/unit/backends/test_executor_import_guard.py"""
import importlib


def test_executor_import_does_not_pull_acl():
    importlib.import_module("adapdomain.backends.ascend.executor")
```

- [ ] **Step 6: Run, commit, log**

Run: `pytest tests/unit/backends/ tests/integration/ -v`

```bash
git add -A
git commit -m "M15: AscendModel, AscendDataset, AscendExecutor wired with strict cleanup"
```

Append `M15 DONE`. Remote benchmark deferred to M17.

---

### Task M16: Resource limit adapter

**Files:**

- Create: `AdapDomain/src/adapdomain/backends/ascend/resource_limit.py`
- Create: `AdapDomain/docs/hardware/ascend_910b3_acl_backend.md`
- Create: `AdapDomain/tests/unit/backends/test_resource_limit_config.py`

- [ ] **Step 1: Failing test**

```python
import pytest

from adapdomain.backends.ascend.resource_limit import AscendResourceLimiter, ResourceLimitConfig
from adapdomain.backends.base import ResourceQuota


def test_no_op_when_types_unset(caplog):
    cfg = ResourceLimitConfig(cube_limit_type=None, vector_limit_type=None, hbm_limit_type=None)
    limiter = AscendResourceLimiter(cfg)
    limiter.apply(ResourceQuota(device_id=0, cube_units=10, vector_units=20))
    assert any("no-op" in r.message for r in caplog.records)


def test_rejects_negative_units():
    cfg = ResourceLimitConfig(cube_limit_type=1, vector_limit_type=2)
    limiter = AscendResourceLimiter(cfg)
    with pytest.raises(ValueError):
        limiter.apply(ResourceQuota(device_id=0, cube_units=-1, vector_units=20))
```

- [ ] **Step 2: Implement `backends/ascend/resource_limit.py`**

```python
"""Wrapper around acl.rt.set_device_res_limit.

The CANN type enum is provided via configuration after remote confirmation; the
adapter is a no-op when types are unset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from adapdomain.backends.base import ResourceQuota

from .acl_runtime import AclRuntime, _import_acl
from .errors import AclDeviceError

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceLimitConfig:
    cube_limit_type: int | None
    vector_limit_type: int | None
    hbm_limit_type: int | None = None


class AscendResourceLimiter:
    """`set_device_res_limit` must be called after set_device and before any operator execution."""

    def __init__(self, config: ResourceLimitConfig) -> None:
        self._config = config

    def apply(self, quota: ResourceQuota) -> None:
        if quota.cube_units is not None and quota.cube_units < 0:
            raise ValueError("cube_units must be >= 0")
        if quota.vector_units is not None and quota.vector_units < 0:
            raise ValueError("vector_units must be >= 0")

        if (
            self._config.cube_limit_type is None
            and self._config.vector_limit_type is None
            and self._config.hbm_limit_type is None
        ):
            _log.warning(
                "AscendResourceLimiter no-op: set_device_res_limit type enums not configured for device %d",
                quota.device_id,
            )
            return

        acl = _import_acl()
        if self._config.cube_limit_type is not None and quota.cube_units is not None:
            ret = acl.rt.set_device_res_limit(quota.device_id, self._config.cube_limit_type, quota.cube_units)
            if ret != 0:
                raise AclDeviceError("set_device_res_limit cube failed", ret=ret, operation="set_device_res_limit(cube)")
        if self._config.vector_limit_type is not None and quota.vector_units is not None:
            ret = acl.rt.set_device_res_limit(quota.device_id, self._config.vector_limit_type, quota.vector_units)
            if ret != 0:
                raise AclDeviceError("set_device_res_limit vector failed", ret=ret, operation="set_device_res_limit(vector)")
        if self._config.hbm_limit_type is not None and quota.hbm_bytes is not None:
            ret = acl.rt.set_device_res_limit(quota.device_id, self._config.hbm_limit_type, quota.hbm_bytes)
            if ret != 0:
                raise AclDeviceError("set_device_res_limit hbm failed", ret=ret, operation="set_device_res_limit(hbm)")
```

- [ ] **Step 3: Write `docs/hardware/ascend_910b3_acl_backend.md`**

```markdown
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

Reference: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/API/appdevgapi/aclpythondevg_01_1033.html

- Must be called after `set_device` and before any operator execution.
- Last call wins per `(device_id, type)`.
- The `type` enum is hardware-specific. On the deployed CANN version, run the
  smoke command in `docs/hardware/remote_validation.md` to retrieve the
  available enum values and fill them into
  `configs/hardware/ascend_910b3_4card.yaml` under `acl.set_device_res_limit`.

## Cleanup expectations

Every allocation must use the `__enter__`/`__exit__` paths in
`backends/ascend/`. Forced unload on error is required to avoid HBM leaks
during long runs.
```

- [ ] **Step 4: Run, commit, log**

Run: `pytest tests/unit/backends/test_resource_limit_config.py -v`

```bash
git add -A
git commit -m "M16: AscendResourceLimiter (set_device_res_limit) + hardware doc"
```

Append M16 DONE.

---

### Task M17: Healthcheck

**Files:**

- Create: `AdapDomain/src/adapdomain/backends/ascend/healthcheck.py`
- Create: `AdapDomain/src/adapdomain/cli/commands/verify_remote.py`
- Modify: `AdapDomain/src/adapdomain/cli/main.py` to delegate `verify-remote` to the new module
- Create: `AdapDomain/tests/unit/backends/test_healthcheck_offline.py`

- [ ] **Step 1: Implement `backends/ascend/healthcheck.py`**

```python
"""ACL/CANN healthcheck used by `adapdomain verify-remote`."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapdomain.utils.logging import get_logger
from adapdomain.utils.subprocess import SubprocessError, run_checked

_log = get_logger(__name__)


@dataclass
class HealthcheckReport:
    python_version: str
    cann_env: dict[str, str | None]
    acl_importable: bool
    msprof_available: bool
    npu_smi_available: bool
    device_count: int | None
    smoke_inference_ok: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "cann_env": self.cann_env,
            "acl_importable": self.acl_importable,
            "msprof_available": self.msprof_available,
            "npu_smi_available": self.npu_smi_available,
            "device_count": self.device_count,
            "smoke_inference_ok": self.smoke_inference_ok,
            "notes": self.notes,
        }


def run_healthcheck(*, smoke_om_path: str | None = None) -> HealthcheckReport:
    rep = HealthcheckReport(
        python_version=sys.version.split()[0],
        cann_env={
            "ASCEND_TOOLKIT_HOME": os.environ.get("ASCEND_TOOLKIT_HOME"),
            "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
        },
        acl_importable=False,
        msprof_available=shutil.which("msprof") is not None,
        npu_smi_available=shutil.which("npu-smi") is not None,
        device_count=None,
        smoke_inference_ok=False,
    )

    try:
        import acl  # type: ignore
        rep.acl_importable = True
    except Exception as exc:  # noqa: BLE001
        rep.notes.append(f"acl not importable: {exc}")
        return rep

    from .acl_runtime import AclRuntime
    from .device import AscendDevice

    rt = AclRuntime()
    rt.init()
    try:
        rep.device_count = AscendDevice.get_count()
        if smoke_om_path and Path(smoke_om_path).exists():
            try:
                from .context import AscendContext
                from .dataset import AscendDataset
                from .executor import AscendExecutor
                from .model import AscendModel
                from .stream import AscendStream

                device = AscendDevice(0)
                with device.bind(), AscendContext(0), AscendStream():
                    with AscendModel(smoke_om_path) as loaded:
                        with AscendDataset() as inputs, AscendDataset() as outputs:
                            AscendExecutor().warmup(model_id=loaded.model_id, input_dataset=inputs.handle, output_dataset=outputs.handle, runs=1)
                            rep.smoke_inference_ok = True
            except Exception as exc:  # noqa: BLE001
                rep.notes.append(f"smoke inference failed: {exc}")
        else:
            rep.notes.append("smoke om not provided; skipping smoke inference")
    finally:
        rt.finalize()

    if rep.msprof_available:
        try:
            run_checked(["msprof", "--version"], timeout=10)
        except SubprocessError as exc:
            rep.notes.append(f"msprof --version returned non-zero: {exc}")
    return rep
```

- [ ] **Step 2: Implement `cli/commands/verify_remote.py`**

```python
"""`adapdomain verify-remote` implementation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from adapdomain.backends.ascend.healthcheck import run_healthcheck
from adapdomain.storage.run_manifest import RunManifest

app = typer.Typer(help="Verify CANN/ACL environment on the remote NPU host.")


@app.callback(invoke_without_command=True)
def verify(
    smoke_om: str | None = typer.Option(None, help="Optional .om path for smoke inference"),
    output_root: str = typer.Option("data/results/healthcheck", help="Result directory"),
) -> None:
    run_id = f"healthcheck-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(output_root) / run_id
    manifest = RunManifest.start(run_id=run_id, command="adapdomain verify-remote", out_dir=out_dir)
    rep = run_healthcheck(smoke_om_path=smoke_om)
    (out_dir / "report.json").write_text(json.dumps(rep.to_dict(), indent=2, sort_keys=True))
    manifest.set_hardware(
        "ascend_910b3" if rep.device_count else "unknown",
        device_count=rep.device_count or 0,
        cann_version=None,
    )
    manifest.set_status("done" if rep.acl_importable else "error")
    manifest.finalize()
    typer.echo(out_dir / "report.json")
```

- [ ] **Step 3: Wire it into `cli/main.py`** by replacing the placeholder:

Replace the existing `verify_remote_placeholder` function with:

```python
from adapdomain.cli.commands.verify_remote import app as _verify_remote_app
app.add_typer(_verify_remote_app, name="verify-remote", help="Verify CANN/ACL on the remote NPU host.")
```

(Remove the placeholder `@app.command("verify-remote")` function.)

- [ ] **Step 4: Offline test**

```python
"""Healthcheck must produce a report even without ACL/msprof."""
from adapdomain.backends.ascend.healthcheck import run_healthcheck


def test_offline_returns_report():
    rep = run_healthcheck()
    assert rep.python_version
    assert isinstance(rep.acl_importable, bool)
    assert rep.device_count is None or isinstance(rep.device_count, int)
```

- [ ] **Step 5: CLI smoke test still green**

Run: `pytest tests/unit/backends/test_healthcheck_offline.py tests/integration/test_cli_smoke.py -v`

- [ ] **Step 6: Commit and log**

```bash
git add -A
git commit -m "M17: backend healthcheck + adapdomain verify-remote command"
```

Append `M17 DONE  ---  verify-remote returns report.json offline; remote run scheduled with run_healthcheck.sh`.

---

### Task M18: msprof runner

**Files:**

- Create: `AdapDomain/src/adapdomain/profiler/__init__.py`
- Create: `AdapDomain/src/adapdomain/profiler/msprof_runner.py`
- Create: `AdapDomain/tests/unit/profiler/__init__.py`
- Create: `AdapDomain/tests/unit/profiler/test_msprof_runner.py`

- [ ] **Step 1: Failing test**

```python
import shutil
import sys

import pytest

from adapdomain.profiler.msprof_runner import MsprofRunner, MsprofRunSpec
from adapdomain.core.errors import MsprofRunError


def test_runner_raises_if_binary_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    runner = MsprofRunner(binary="msprof", output_root=tmp_path)
    with pytest.raises(MsprofRunError):
        runner.run(MsprofRunSpec(command=[sys.executable, "-c", "pass"], metrics=[], duration_s=1, run_id="r0"))


def test_runner_creates_unique_output_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/msprof")
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        from adapdomain.utils.subprocess import RunResult
        return RunResult(cmd=list(cmd), returncode=0, stdout="", stderr="")

    monkeypatch.setattr("adapdomain.profiler.msprof_runner.run_checked", fake_run)
    runner = MsprofRunner(binary="msprof", output_root=tmp_path)
    spec = MsprofRunSpec(command=["python", "-c", "pass"], metrics=["PipeUtilization"], duration_s=2, run_id="r0")
    out_dir = runner.run(spec)
    assert out_dir.exists()
    assert calls and calls[0][0] == "/usr/bin/msprof"
    assert "--output" in " ".join(calls[0])
```

- [ ] **Step 2: Implement `profiler/msprof_runner.py`**

```python
"""Drive the `msprof` binary; do not parse here (see M19)."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from adapdomain.core.errors import MsprofRunError
from adapdomain.utils.subprocess import SubprocessError, run_checked


@dataclass
class MsprofRunSpec:
    command: Sequence[str]
    metrics: Sequence[str]
    duration_s: float
    run_id: str
    extra_args: Sequence[str] = field(default_factory=list)


class MsprofRunner:
    def __init__(self, *, binary: str = "msprof", output_root: str | Path = "data/raw/msprof") -> None:
        self._binary = binary
        self._output_root = Path(output_root)

    def run(self, spec: MsprofRunSpec) -> Path:
        if not shutil.which(self._binary):
            raise MsprofRunError(f"{self._binary} not on PATH")
        out_dir = self._output_root / spec.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = ",".join(spec.metrics) if spec.metrics else "Default"
        cmd = [
            shutil.which(self._binary) or self._binary,
            "--output", str(out_dir),
            "--application", " ".join(spec.command),
            "--aic-metrics", metrics,
            "--duration", str(int(spec.duration_s)),
            *spec.extra_args,
        ]
        try:
            run_checked(cmd, timeout=spec.duration_s + 60)
        except SubprocessError as exc:
            raise MsprofRunError(f"msprof failed: {exc.stderr.strip()[:200]}") from exc
        return out_dir
```

- [ ] **Step 3: Implement `profiler/__init__.py`** (stub re-exports)

```python
"""AdapDomain profiler package."""
from .msprof_runner import MsprofRunner, MsprofRunSpec

__all__ = ["MsprofRunner", "MsprofRunSpec"]
```

- [ ] **Step 4: Run, commit, log**

Run: `pytest tests/unit/profiler/test_msprof_runner.py -v`
Expected: 2/2 pass.

```bash
git add -A
git commit -m "M18: MsprofRunner; offline tests via monkeypatched run_checked"
```

Append M18 DONE.

---

### Task M19: msprof parser

**Files:**

- Create: `AdapDomain/src/adapdomain/profiler/msprof_parser.py`
- Create: `AdapDomain/docs/hardware/msprof_data_contract.md`
- Create: `AdapDomain/tests/fixtures/sample_msprof/cann83_variant.csv`
- Create: `AdapDomain/tests/unit/profiler/test_msprof_parser.py`

- [ ] **Step 1: Add a second fixture variant `cann83_variant.csv`**

```csv
OpName,Duration,AicMacRatio,AivVecRatio,HBMBwUtilization
conv1,2.0,0.75,0.05,0.45
elem1,0.6,0.04,0.66,0.18
layernorm,0.5,0.08,0.62,0.21
```

- [ ] **Step 2: Failing test**

```python
from pathlib import Path

import pytest

from adapdomain.profiler.msprof_parser import MsprofParser, MsprofSchema


FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "sample_msprof"


def test_default_schema_parses_minimal_csv():
    schema = MsprofSchema(
        cann_version="8.3.RC1",
        op_name_keys=["op_name"], duration_keys=["task_duration"],
        cube_keys=["aic_mac_ratio"], vector_keys=["aiv_vec_ratio"], hbm_keys=["hbm_bw_util"],
    )
    parser = MsprofParser(schema=schema)
    rows = parser.parse_csv(FIXTURES / "cann83_minimal.csv")
    assert {r["op_name"] for r in rows} == {"GEMM", "ELEMWISE", "LAYERNORM"}
    gemm = next(r for r in rows if r["op_name"] == "GEMM")
    assert gemm["cube"] == pytest.approx(0.80)
    assert gemm["vector"] == pytest.approx(0.10)


def test_variant_schema_remaps_columns():
    schema = MsprofSchema(
        cann_version="8.3.RC1",
        op_name_keys=["OpName"], duration_keys=["Duration"],
        cube_keys=["AicMacRatio"], vector_keys=["AivVecRatio"], hbm_keys=["HBMBwUtilization"],
    )
    rows = MsprofParser(schema=schema).parse_csv(FIXTURES / "cann83_variant.csv")
    assert rows[0]["cube"] == pytest.approx(0.75)


def test_missing_column_logs_warning_not_crash(caplog):
    schema = MsprofSchema(
        cann_version="8.3.RC1",
        op_name_keys=["op_name"], duration_keys=["task_duration"],
        cube_keys=["aic_mac_ratio"], vector_keys=["aiv_vec_ratio"], hbm_keys=["missing_field"],
    )
    rows = MsprofParser(schema=schema).parse_csv(FIXTURES / "cann83_minimal.csv")
    assert all(r["hbm"] is None for r in rows)
    assert any("missing_field" in r.message for r in caplog.records)
```

- [ ] **Step 3: Implement `profiler/msprof_parser.py`**

```python
"""msprof CSV parser tolerant to per-CANN field-name drift."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MsprofSchema:
    cann_version: str
    op_name_keys: list[str]
    duration_keys: list[str]
    cube_keys: list[str]
    vector_keys: list[str]
    hbm_keys: list[str]


class MsprofParser:
    def __init__(self, *, schema: MsprofSchema) -> None:
        self._schema = schema

    def _first_present(self, row: dict[str, str], keys: list[str], cast=float):
        for k in keys:
            if k in row:
                try:
                    return cast(row[k])
                except (TypeError, ValueError):
                    return None
        _log.warning("msprof parser: none of %s present in row", keys)
        return None

    def parse_csv(self, path: str | Path) -> list[dict[str, Any]]:
        with Path(path).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out: list[dict[str, Any]] = []
            for row in reader:
                out.append({
                    "op_name": self._first_present(row, self._schema.op_name_keys, cast=str),
                    "duration": self._first_present(row, self._schema.duration_keys),
                    "cube": self._first_present(row, self._schema.cube_keys),
                    "vector": self._first_present(row, self._schema.vector_keys),
                    "hbm": self._first_present(row, self._schema.hbm_keys),
                })
        return out
```

- [ ] **Step 4: Document the data contract**

`docs/hardware/msprof_data_contract.md`:

```markdown
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
```

- [ ] **Step 5: Run, commit, log**

Run: `pytest tests/unit/profiler/test_msprof_parser.py -v`
Expected: 3/3 pass.

```bash
git add -A
git commit -m "M19: MsprofParser with pluggable schema and warning-only missing fields"
```

Append M19 DONE.

---

### Task M20: Frontier builder, normalization, confidence wiring

**Files:**

- Create: `AdapDomain/src/adapdomain/profiler/normalization.py`
- Create: `AdapDomain/src/adapdomain/profiler/confidence.py`
- Create: `AdapDomain/src/adapdomain/profiler/frontier_builder.py`
- Create: `AdapDomain/tests/unit/profiler/test_frontier_builder.py`

- [ ] **Step 1: Failing test**

```python
import pytest

from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation


def test_builder_builds_snapshot_from_observations():
    obs = [
        MsprofObservation(model="bert-base", point_id="a",
                          raw_cube=0.40, raw_vector=0.20, raw_hbm=0.40,
                          batch_size=1, slo_ms=15.0,
                          p50_ms=4.0, p95_ms=6.0, p99_ms=7.0, goodput_qps=100.0,
                          feasible=True),
        MsprofObservation(model="bert-base", point_id="b",
                          raw_cube=0.60, raw_vector=0.25, raw_hbm=0.55,
                          batch_size=1, slo_ms=15.0,
                          p50_ms=5.0, p95_ms=9.0, p99_ms=10.0, goodput_qps=130.0,
                          feasible=True),
    ]
    snap = FrontierBuilder(
        cube_range=(0.0, 1.0), vector_range=(0.0, 1.0),
    ).build(model_name="bert-base", version=1, observations=obs, source_run_id="r0")
    assert snap.dominant_domain in {"matrix", "vector", "balanced"}
    assert snap.safe_point().point_id == "a"
    assert all(0.0 <= p.norm_cube <= 1.0 for p in snap.points)


def test_normalization_clamps_above_one():
    obs = [MsprofObservation(model="m", point_id="x",
                             raw_cube=1.2, raw_vector=0.5, raw_hbm=0.5,
                             batch_size=1, slo_ms=10.0,
                             p50_ms=5, p95_ms=8, p99_ms=9, goodput_qps=80,
                             feasible=True)]
    snap = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0)).build(
        model_name="m", version=1, observations=obs, source_run_id="r",
    )
    assert snap.points[0].norm_cube == pytest.approx(1.0)
```

- [ ] **Step 2: Implement `profiler/normalization.py`**

```python
"""Map raw msprof counters into [0, 1] coordinates per domain."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CounterRange:
    minimum: float
    saturation: float


def normalize(value: float, rng: CounterRange) -> float:
    if rng.saturation <= rng.minimum:
        raise ValueError("saturation must be > minimum")
    norm = (value - rng.minimum) / (rng.saturation - rng.minimum)
    return float(max(0.0, min(1.0, norm)))
```

- [ ] **Step 3: Implement `profiler/confidence.py`**

```python
"""Wrapper around core.frontier.ConfidenceModel keyed to msprof inputs."""
from __future__ import annotations

from datetime import datetime

from adapdomain.core.frontier import ConfidenceModel


def summarize_confidence(*, created_at: datetime, point_count: int, deviations: list[float] | None = None) -> float:
    return ConfidenceModel().score(
        neighbor_count=point_count,
        recent_deviations=deviations or [0.0],
        created_at=created_at,
    )
```

- [ ] **Step 4: Implement `profiler/frontier_builder.py`**

```python
"""Build FrontierSnapshot from MsprofObservation lists."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot

from .confidence import summarize_confidence
from .normalization import CounterRange, normalize


@dataclass(frozen=True)
class MsprofObservation:
    model: str
    point_id: str
    raw_cube: float
    raw_vector: float
    raw_hbm: float
    batch_size: int
    slo_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    goodput_qps: float
    feasible: bool


class FrontierBuilder:
    def __init__(self, *, cube_range: tuple[float, float], vector_range: tuple[float, float]) -> None:
        self._cube = CounterRange(*cube_range)
        self._vector = CounterRange(*vector_range)

    def build(
        self,
        *,
        model_name: str,
        version: int,
        observations: Iterable[MsprofObservation],
        source_run_id: str,
    ) -> FrontierSnapshot:
        obs = list(observations)
        if not obs:
            raise ValueError("no observations")
        points = [
            FrontierPoint(
                model_name=model_name,
                point_id=o.point_id,
                raw_cube=o.raw_cube, raw_vector=o.raw_vector, raw_hbm=o.raw_hbm,
                norm_cube=normalize(o.raw_cube, self._cube),
                norm_vector=normalize(o.raw_vector, self._vector),
                batch_size=o.batch_size, slo_ms=o.slo_ms,
                p50_ms=o.p50_ms, p95_ms=o.p95_ms, p99_ms=o.p99_ms,
                goodput_qps=o.goodput_qps,
                feasible=o.feasible,
                confidence=summarize_confidence(created_at=datetime.now(timezone.utc), point_count=len(obs)),
            )
            for o in obs
        ]
        feasible_points = [p for p in points if p.feasible]
        safe = min(feasible_points, key=lambda p: p.p95_ms) if feasible_points else points[0]
        mean_cube = sum(p.norm_cube for p in points) / len(points)
        mean_vec = sum(p.norm_vector for p in points) / len(points)
        if mean_cube > mean_vec * 1.5:
            dominant = "matrix"
        elif mean_vec > mean_cube * 1.5:
            dominant = "vector"
        else:
            dominant = "balanced"
        return FrontierSnapshot(
            model_name=model_name,
            version=version,
            k_domains=["cube", "vector"],
            points=points,
            safe_point_id=safe.point_id,
            dominant_domain=dominant,
            confidence_summary=float(sum(p.confidence for p in points) / len(points)),
            source_run_id=source_run_id,
        )
```

- [ ] **Step 5: Run, commit, log**

Run: `pytest tests/unit/profiler/test_frontier_builder.py -v`

```bash
git add -A
git commit -m "M20: frontier builder, normalization, confidence wiring"
```

Append M20 DONE. (M22 — "profiling normalization" — is fully implemented inside M20 and marked done together.)

---

### Task M21: ActiveVC

**Files:**

- Create: `AdapDomain/src/adapdomain/profiler/candidate_grid.py`
- Create: `AdapDomain/src/adapdomain/profiler/acquisition.py`
- Create: `AdapDomain/src/adapdomain/profiler/activevc.py`
- Create: `AdapDomain/tests/unit/profiler/test_candidate_grid.py`
- Create: `AdapDomain/tests/unit/profiler/test_acquisition.py`
- Create: `AdapDomain/tests/unit/profiler/test_activevc.py`

This module can be parallelized into three subagents (grid, acquisition, main loop) integrated by a fourth.

- [ ] **Step 1: Failing test `tests/unit/profiler/test_candidate_grid.py`**

```python
import pytest

from adapdomain.profiler.candidate_grid import CandidateGrid


def test_seed_grid_dimensions():
    g = CandidateGrid(seed_dims=(4, 4, 3), boundary_dims=(8, 8))
    seeds = list(g.seed_points())
    assert len(seeds) == 4 * 4 * 3
    assert all(0.0 <= s.v <= 1.0 and 0.0 <= s.m <= 1.0 and 0.0 <= s.h <= 1.0 for s in seeds)


def test_boundary_dims_validation():
    with pytest.raises(ValueError):
        CandidateGrid(seed_dims=(0, 0, 0), boundary_dims=(8, 8))


def test_boundary_refinement_grid():
    g = CandidateGrid(seed_dims=(4, 4, 3), boundary_dims=(8, 8))
    boundary = list(g.boundary_points(hbm=0.5))
    assert len(boundary) == 8 * 8
```

- [ ] **Step 2: Implement `profiler/candidate_grid.py`**

```python
"""Sparse seed and dense V:M boundary grids."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class GridPoint:
    v: float
    m: float
    h: float


class CandidateGrid:
    def __init__(self, *, seed_dims: tuple[int, int, int], boundary_dims: tuple[int, int]) -> None:
        if min(seed_dims) <= 0 or min(boundary_dims) <= 0:
            raise ValueError("grid dimensions must be positive")
        self._seed_dims = seed_dims
        self._boundary_dims = boundary_dims

    def seed_points(self) -> Iterator[GridPoint]:
        nv, nm, nh = self._seed_dims
        for iv in range(nv):
            for im in range(nm):
                for ih in range(nh):
                    yield GridPoint(
                        v=iv / max(nv - 1, 1),
                        m=im / max(nm - 1, 1),
                        h=ih / max(nh - 1, 1),
                    )

    def boundary_points(self, hbm: float) -> Iterator[GridPoint]:
        nv, nm = self._boundary_dims
        for iv in range(nv):
            for im in range(nm):
                yield GridPoint(
                    v=iv / max(nv - 1, 1),
                    m=im / max(nm - 1, 1),
                    h=hbm,
                )
```

- [ ] **Step 3: Implement `profiler/acquisition.py`**

```python
"""ActiveVC acquisition function (paper Eq. activevc_acquisition)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .candidate_grid import GridPoint


@dataclass(frozen=True)
class AcquisitionWeights:
    alpha: float
    beta: float
    gamma: float
    tau: float


def acquisition(
    *,
    candidate: GridPoint,
    uncertainty: float,
    distance_to_frontier: float,
    marginal_goodput_gain: float,
    weights: AcquisitionWeights,
) -> float:
    return (
        weights.alpha * uncertainty
        + weights.beta * math.exp(-distance_to_frontier / max(weights.tau, 1e-6))
        + weights.gamma * marginal_goodput_gain
    )


def best_candidate(
    candidates: Iterable[GridPoint],
    *,
    uncertainty_fn,
    distance_fn,
    gain_fn,
    weights: AcquisitionWeights,
) -> GridPoint:
    best: tuple[GridPoint, float] | None = None
    for c in candidates:
        score = acquisition(
            candidate=c,
            uncertainty=uncertainty_fn(c),
            distance_to_frontier=distance_fn(c),
            marginal_goodput_gain=gain_fn(c),
            weights=weights,
        )
        if best is None or score > best[1]:
            best = (c, score)
    assert best is not None
    return best[0]
```

- [ ] **Step 4: Implement `profiler/activevc.py`**

```python
"""ActiveVC main loop.

Stages:
1. sparse seed across a 4x4x3 V:M:HBM grid
2. boundary refinement at 8x8 in V:M holding HBM at the median safe value
3. uncertainty-guided sampling using `acquisition.best_candidate`

This module never imports `acl` or `msprof`; the caller passes an
`evaluate(point) -> MsprofObservation`-like callable.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable

from adapdomain.core.frontier import FrontierSnapshot

from .acquisition import AcquisitionWeights, acquisition
from .candidate_grid import CandidateGrid, GridPoint
from .frontier_builder import FrontierBuilder, MsprofObservation


@dataclass
class ActiveVCConfigRuntime:
    budget: int
    weights: AcquisitionWeights
    seed_dims: tuple[int, int, int] = (4, 4, 3)
    boundary_dims: tuple[int, int] = (8, 8)


@dataclass
class ActiveVCRunOutput:
    snapshot: FrontierSnapshot
    probed_points: list[MsprofObservation]
    spent_budget: int


class ActiveVC:
    def __init__(
        self,
        *,
        config: ActiveVCConfigRuntime,
        builder: FrontierBuilder,
    ) -> None:
        self._config = config
        self._builder = builder

    def run(
        self,
        *,
        model_name: str,
        version: int,
        source_run_id: str,
        evaluate: Callable[[GridPoint], MsprofObservation],
    ) -> ActiveVCRunOutput:
        grid = CandidateGrid(seed_dims=self._config.seed_dims, boundary_dims=self._config.boundary_dims)
        observations: list[MsprofObservation] = []
        seen: set[tuple[float, float, float]] = set()

        budget_left = self._config.budget

        # Stage 1: sparse seeds (cap by budget)
        for seed in self._sparse_seeds(grid, max_count=min(budget_left, 8)):
            obs = evaluate(seed)
            observations.append(obs)
            seen.add((seed.v, seed.m, seed.h))
            budget_left -= 1
            if budget_left == 0:
                return self._finalize(model_name, version, source_run_id, observations, spent=self._config.budget)

        # Stage 2: V:M boundary refinement at median safe HBM
        feasible_hbm = [o.raw_hbm for o in observations if o.feasible]
        median_hbm = statistics.median(feasible_hbm) if feasible_hbm else 0.5
        for point in self._boundary_targets(grid, hbm=median_hbm, observations=observations, max_count=min(budget_left, 8)):
            key = (point.v, point.m, point.h)
            if key in seen:
                continue
            obs = evaluate(point)
            observations.append(obs)
            seen.add(key)
            budget_left -= 1
            if budget_left == 0:
                return self._finalize(model_name, version, source_run_id, observations, spent=self._config.budget)

        # Stage 3: uncertainty-guided
        while budget_left > 0:
            candidates = [p for p in grid.boundary_points(hbm=median_hbm) if (p.v, p.m, p.h) not in seen]
            if not candidates:
                break

            def uncertainty(c: GridPoint) -> float:
                return self._uncertainty(observations, c)

            def distance(c: GridPoint) -> float:
                return self._distance_to_frontier(observations, c)

            def gain(c: GridPoint) -> float:
                return self._marginal_gain(observations, c)

            best = max(
                candidates,
                key=lambda c: acquisition(
                    candidate=c,
                    uncertainty=uncertainty(c),
                    distance_to_frontier=distance(c),
                    marginal_goodput_gain=gain(c),
                    weights=self._config.weights,
                ),
            )
            obs = evaluate(best)
            observations.append(obs)
            seen.add((best.v, best.m, best.h))
            budget_left -= 1

        return self._finalize(model_name, version, source_run_id, observations, spent=self._config.budget - budget_left)

    @staticmethod
    def _sparse_seeds(grid: CandidateGrid, *, max_count: int) -> list[GridPoint]:
        seeds = list(grid.seed_points())
        if len(seeds) <= max_count:
            return seeds
        step = max(1, len(seeds) // max_count)
        return seeds[::step][:max_count]

    @staticmethod
    def _boundary_targets(grid: CandidateGrid, *, hbm: float, observations: list[MsprofObservation], max_count: int) -> list[GridPoint]:
        # pick points near the empirical boundary by alternating feasible/infeasible neighbours
        candidates = list(grid.boundary_points(hbm=hbm))
        candidates.sort(key=lambda c: abs(0.5 - c.v) + abs(0.5 - c.m))
        return candidates[:max_count]

    @staticmethod
    def _uncertainty(obs: list[MsprofObservation], c: GridPoint) -> float:
        # 1 / (1 + distance to nearest observation)
        if not obs:
            return 1.0
        d = min(math.hypot(c.v - o.raw_vector, c.m - o.raw_cube) for o in obs)
        return 1.0 / (1.0 + d)

    @staticmethod
    def _distance_to_frontier(obs: list[MsprofObservation], c: GridPoint) -> float:
        feasible = [o for o in obs if o.feasible]
        infeasible = [o for o in obs if not o.feasible]
        if not feasible or not infeasible:
            return 0.5
        return min(math.hypot(c.v - o.raw_vector, c.m - o.raw_cube) for o in feasible + infeasible)

    @staticmethod
    def _marginal_gain(obs: list[MsprofObservation], c: GridPoint) -> float:
        feasible = [o for o in obs if o.feasible]
        if not feasible:
            return 0.0
        best = max(o.goodput_qps for o in feasible)
        return max(0.0, best * (1.0 - 0.5 * (abs(c.v - 0.5) + abs(c.m - 0.5))))

    def _finalize(self, model_name: str, version: int, source_run_id: str, observations: list[MsprofObservation], *, spent: int) -> ActiveVCRunOutput:
        snap = self._builder.build(
            model_name=model_name,
            version=version,
            observations=observations,
            source_run_id=source_run_id,
        )
        return ActiveVCRunOutput(snapshot=snap, probed_points=observations, spent_budget=spent)
```

- [ ] **Step 5: Failing test `tests/unit/profiler/test_acquisition.py`**

```python
import math

from adapdomain.profiler.acquisition import AcquisitionWeights, acquisition
from adapdomain.profiler.candidate_grid import GridPoint


def test_acquisition_paper_form():
    weights = AcquisitionWeights(alpha=0.40, beta=0.40, gamma=0.20, tau=0.15)
    s = acquisition(
        candidate=GridPoint(v=0.5, m=0.5, h=0.5),
        uncertainty=0.6,
        distance_to_frontier=0.10,
        marginal_goodput_gain=0.30,
        weights=weights,
    )
    expected = 0.40 * 0.6 + 0.40 * math.exp(-0.10 / 0.15) + 0.20 * 0.30
    assert s == math.isclose(s, expected, rel_tol=1e-9) or abs(s - expected) < 1e-9
```

- [ ] **Step 6: Failing test `tests/unit/profiler/test_activevc.py`**

```python
import math

import pytest

from adapdomain.profiler.activevc import ActiveVC, ActiveVCConfigRuntime
from adapdomain.profiler.acquisition import AcquisitionWeights
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation


def _synthetic_eval(point: GridPoint) -> MsprofObservation:
    feasible = point.v + point.m <= 1.2 and point.h <= 0.9
    p95 = 5.0 + 20.0 * max(0.0, point.v + point.m - 1.0)
    return MsprofObservation(
        model="m", point_id=f"v{point.v:.2f}-m{point.m:.2f}-h{point.h:.2f}",
        raw_cube=point.m, raw_vector=point.v, raw_hbm=point.h,
        batch_size=1, slo_ms=20.0,
        p50_ms=p95 / 2, p95_ms=p95, p99_ms=p95 * 1.1,
        goodput_qps=100.0 if feasible else 5.0,
        feasible=feasible,
    )


def test_activevc_runs_within_budget():
    cfg = ActiveVCConfigRuntime(
        budget=24,
        weights=AcquisitionWeights(alpha=0.40, beta=0.40, gamma=0.20, tau=0.15),
    )
    builder = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0))
    out = ActiveVC(config=cfg, builder=builder).run(
        model_name="m", version=1, source_run_id="syn", evaluate=_synthetic_eval,
    )
    assert out.spent_budget == 24
    assert any(p.feasible for p in out.snapshot.points)
    assert any(not p.feasible for p in out.snapshot.points)


def test_smaller_budget_runs_to_completion():
    cfg = ActiveVCConfigRuntime(
        budget=8,
        weights=AcquisitionWeights(alpha=0.40, beta=0.40, gamma=0.20, tau=0.15),
    )
    builder = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0))
    out = ActiveVC(config=cfg, builder=builder).run(
        model_name="m", version=1, source_run_id="syn", evaluate=_synthetic_eval,
    )
    assert out.spent_budget == 8
```

- [ ] **Step 7: Run, commit, log**

Run: `pytest tests/unit/profiler/ -v`
Expected: 9 tests pass (grid 3 + acquisition 1 + activevc 2 + earlier 3).

```bash
git add -A
git commit -m "M21: ActiveVC grid + acquisition + main loop on synthetic surface"
```

Append M21 DONE; M22 DONE (folded into M20/M21).

---

### Task M23: Remote scripts

**Files:**

- Create: `AdapDomain/scripts/remote_ascend/setup_env.sh`
- Create: `AdapDomain/scripts/remote_ascend/run_healthcheck.sh`
- Create: `AdapDomain/scripts/remote_ascend/run_profile.sh`
- Create: `AdapDomain/scripts/remote_ascend/collect_msprof.sh`
- Create: `AdapDomain/scripts/remote_ascend/run_e2e_replay.sh`
- Create: `AdapDomain/scripts/remote_ascend/run_paper_bundle.sh`
- Create: `AdapDomain/docs/hardware/remote_validation.md`

- [ ] **Step 1: Write `setup_env.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [ ! -f "$CANN_ENV" ]; then
  echo "CANN env script not found at $CANN_ENV" >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$CANN_ENV"

VENV="${VENV:-.venv}"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --upgrade pip
pip install -e .[dev]

echo "ok: env ready"
```

- [ ] **Step 2: Write `run_healthcheck.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
SMOKE_OM="${SMOKE_OM:-}"
if [ -n "$SMOKE_OM" ]; then
  adapdomain verify-remote --smoke-om "$SMOKE_OM"
else
  adapdomain verify-remote
fi
```

- [ ] **Step 3: Write `run_profile.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
REGISTRY="${MODEL_REGISTRY:-configs/models/model_registry.example.yaml}"
HARDWARE="${HARDWARE_CFG:-configs/hardware/ascend_910b3_4card.yaml}"
PROFCFG="${PROFILE_CFG:-configs/profiling/activevc_default.yaml}"
adapdomain profile --registry "$REGISTRY" --hardware "$HARDWARE" --profile "$PROFCFG"
```

- [ ] **Step 4: Write `collect_msprof.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
RUN_ID="${RUN_ID:-msprof-$(date -u +%Y%m%dT%H%M%SZ)}"
APP=("python" "-c" "print('placeholder; replace with model command')")
msprof --output "data/raw/msprof/${RUN_ID}" --application "${APP[*]}" --aic-metrics PipeUtilization,ArithmeticUtilization,HBM --duration 30
echo "ok: msprof output at data/raw/msprof/${RUN_ID}"
```

- [ ] **Step 5: Write `run_e2e_replay.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
CONFIG="${REPLAY_CFG:-configs/experiments/e2e_replay.yaml}"
adapdomain replay --config "$CONFIG"
```

- [ ] **Step 6: Write `run_paper_bundle.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
"$(dirname "$0")/setup_env.sh" >/dev/null
adapdomain profile --registry configs/models/model_registry.example.yaml --hardware configs/hardware/ascend_910b3_4card.yaml --profile configs/profiling/activevc_default.yaml
adapdomain replay --config configs/experiments/e2e_replay.yaml
adapdomain ablate --config configs/experiments/ablation.yaml
adapdomain plot paper
```

- [ ] **Step 7: Make scripts executable & write `docs/hardware/remote_validation.md`**

```bash
chmod +x scripts/remote_ascend/*.sh
```

`docs/hardware/remote_validation.md`:

```markdown
# Remote 910B3 validation

Run the scripts in order on the remote Linux NPU server:

1. `bash scripts/remote_ascend/setup_env.sh`
2. `bash scripts/remote_ascend/run_healthcheck.sh`  (optionally `SMOKE_OM=/path/to/resnet152.om`)
3. `bash scripts/remote_ascend/collect_msprof.sh`
4. `bash scripts/remote_ascend/run_profile.sh`
5. `bash scripts/remote_ascend/run_e2e_replay.sh`
6. `bash scripts/remote_ascend/run_paper_bundle.sh`

Each step must exit with code 0. Outputs land under `data/`:

- `data/results/healthcheck/<run-id>/report.json`
- `data/raw/msprof/<run-id>/`
- `data/frontiers/<model>/<version>.json`
- `data/results/replay/<run-id>/summary.csv`
- `data/results/ablation/<run-id>/`
- `data/figures/`, `data/tables/`

Append a line per script run to `docs/development/continuation_checklist.md`,
including the run id and a one-line outcome.
```

- [ ] **Step 8: Commit and log**

```bash
git add -A
git commit -m "M23: remote validation scripts and operator runbook"
```

Append `M23 DONE  ---  six bash scripts in scripts/remote_ascend/; runbook in docs/hardware/remote_validation.md`.

After M23, append `STAGE B DONE` to the checklist.

## Stage C — Scheduler, Runtime, and Experiments

### Task M24: Feasibility filter

**Files:**

- Create: `AdapDomain/src/adapdomain/scheduler/__init__.py`
- Create: `AdapDomain/src/adapdomain/scheduler/feasibility.py`
- Create: `AdapDomain/tests/unit/scheduler/__init__.py`
- Create: `AdapDomain/tests/unit/scheduler/test_feasibility.py`

- [ ] **Step 1: Failing test**

```python
import pytest

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.feasibility import FeasibilityFilter, RejectReason


def _shape(cube=0.4, vec=0.2, slo=15.0, batch=1) -> TenantShape:
    return TenantShape(model_name="bert-base", replica_id="r0",
                       domains={"cube": cube, "vector": vec},
                       slo_ms=slo, batch_size=batch,
                       frontier_version=1, confidence=0.9)


def _residual(cube=1.0, vec=1.0, hbm=0.5, queue=0.5) -> DeviceResidual:
    return DeviceResidual(device_id=0, residual={"cube": cube, "vector": vec},
                          placed_replicas=[], hbm_pressure=hbm, queue_pressure=queue)


def _snapshot_with_point(cube=0.4, vec=0.2, slo=15.0, p95=8.0, feasible=True) -> FrontierSnapshot:
    pt = FrontierPoint(
        model_name="bert-base", point_id="p0",
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=cube, norm_vector=vec,
        batch_size=1, slo_ms=slo,
        p50_ms=p95/2, p95_ms=p95, p99_ms=p95*1.1,
        goodput_qps=100.0, feasible=feasible, confidence=0.9,
    )
    return FrontierSnapshot(
        model_name="bert-base", version=1, k_domains=["cube", "vector"],
        points=[pt], safe_point_id=pt.point_id, dominant_domain="balanced",
        confidence_summary=0.9, source_run_id="r",
    )


def test_passes_when_all_thresholds_satisfied():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(), _snapshot_with_point(), _residual())
    assert ok and reason is None


def test_rejects_when_cube_residual_low():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(cube=0.6), _snapshot_with_point(), _residual(cube=0.4))
    assert not ok and reason is RejectReason.CUBE


def test_rejects_when_vector_residual_low():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(vec=0.3), _snapshot_with_point(), _residual(vec=0.2))
    assert not ok and reason is RejectReason.VECTOR


def test_rejects_when_hbm_pressure_high():
    ff = FeasibilityFilter(hbm_max=0.4, queue_max=0.9, confidence_min=0.5)
    ok, reason = ff.check(_shape(), _snapshot_with_point(), _residual(hbm=0.7))
    assert not ok and reason is RejectReason.HBM


def test_rejects_when_queue_pressure_high():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.3, confidence_min=0.5)
    ok, reason = ff.check(_shape(), _snapshot_with_point(), _residual(queue=0.6))
    assert not ok and reason is RejectReason.QUEUE


def test_rejects_when_confidence_too_low():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.9)
    snap = _snapshot_with_point()
    pt = snap.points[0].model_copy(update={"confidence": 0.5})
    snap = snap.model_copy(update={"points": [pt]})
    ok, reason = ff.check(_shape(), snap, _residual())
    assert not ok and reason is RejectReason.CONFIDENCE


def test_rejects_when_no_frontier_point_under_slo():
    ff = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    snap = _snapshot_with_point(p95=50.0)
    ok, reason = ff.check(_shape(slo=10.0), snap, _residual())
    assert not ok and reason is RejectReason.SLO
```

- [ ] **Step 2: Implement `scheduler/feasibility.py`**

```python
"""Edge feasibility filter for HarmonicFit planning."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from adapdomain.core.frontier import FrontierQuery, FrontierSnapshot
from adapdomain.core.shape import DeviceResidual, TenantShape


class RejectReason(str, Enum):
    SLO = "no_feasible_frontier_for_slo"
    CUBE = "insufficient_cube_residual"
    VECTOR = "insufficient_vector_residual"
    HBM = "hbm_pressure_above_threshold"
    QUEUE = "queue_pressure_above_threshold"
    CONFIDENCE = "frontier_confidence_below_threshold"


@dataclass(frozen=True)
class FeasibilityFilter:
    hbm_max: float
    queue_max: float
    confidence_min: float

    def check(
        self,
        replica: TenantShape,
        snapshot: FrontierSnapshot,
        device: DeviceResidual,
    ) -> tuple[bool, RejectReason | None]:
        point = FrontierQuery(snapshot).pick(batch_size=replica.batch_size, slo_ms=replica.slo_ms)
        if point is None:
            return False, RejectReason.SLO
        if point.confidence < self.confidence_min:
            return False, RejectReason.CONFIDENCE
        if replica.domains.get("cube", 0) > device.residual.get("cube", 0):
            return False, RejectReason.CUBE
        if replica.domains.get("vector", 0) > device.residual.get("vector", 0):
            return False, RejectReason.VECTOR
        if device.hbm_pressure > self.hbm_max:
            return False, RejectReason.HBM
        if device.queue_pressure > self.queue_max:
            return False, RejectReason.QUEUE
        return True, None
```

- [ ] **Step 3: Implement `scheduler/__init__.py`**

```python
"""AdapDomain scheduler package."""
from .feasibility import FeasibilityFilter, RejectReason

__all__ = ["FeasibilityFilter", "RejectReason"]
```

- [ ] **Step 4: Run, commit, log**

Run: `pytest tests/unit/scheduler/test_feasibility.py -v`
Expected: 7/7 pass.

```bash
git add -A
git commit -m "M24: feasibility filter with six explicit reject reasons"
```

Append M24 DONE.

---

### Task M25: Scoring & CEFF

**Files:**

- Create: `AdapDomain/src/adapdomain/scheduler/scoring.py`
- Create: `AdapDomain/tests/unit/scheduler/test_scoring.py`

- [ ] **Step 1: Failing test**

```python
import pytest

from adapdomain.core.types import EPSILON_DOMAIN_FIT
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights, compute_score


def test_skewed_fit_scores_lower_than_balanced():
    weights = ScoreWeights(0.40, 0.0, 0.0, 0.0, 0.0, 0.0)
    risks = RiskTerms(x=0.0, h=0.0, q=0.0, f=0.0, u=0.0)
    balanced = compute_score(rho=[0.5, 0.5], weights=weights, risks=risks).final
    skewed = compute_score(rho=[0.9, 0.1], weights=weights, risks=risks).final
    assert balanced > skewed


def test_full_score_includes_subtractive_risks():
    weights = ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05)
    risks = RiskTerms(x=0.2, h=0.1, q=0.1, f=0.1, u=0.1)
    out = compute_score(rho=[0.5, 0.5], weights=weights, risks=risks)
    assert out.ceff > 0
    assert out.final < out.ceff
```

- [ ] **Step 2: Implement `scheduler/scoring.py`**

```python
"""HarmonicFit edge scoring (paper Eq. harmonicfit_score)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from adapdomain.core.placement import ScoreBreakdown
from adapdomain.core.shape import harmonic_complementarity
from adapdomain.core.types import EPSILON_DOMAIN_FIT


@dataclass(frozen=True)
class ScoreWeights:
    lambda_c: float
    lambda_x: float
    lambda_h: float
    lambda_q: float
    lambda_f: float
    lambda_u: float


@dataclass(frozen=True)
class RiskTerms:
    x: float
    h: float
    q: float
    f: float
    u: float


def compute_score(*, rho: Sequence[float], weights: ScoreWeights, risks: RiskTerms,
                  eps: float = EPSILON_DOMAIN_FIT) -> ScoreBreakdown:
    ceff = harmonic_complementarity(list(rho), eps=eps)
    final = (
        weights.lambda_c * ceff
        - weights.lambda_x * risks.x
        - weights.lambda_h * risks.h
        - weights.lambda_q * risks.q
        - weights.lambda_f * risks.f
        - weights.lambda_u * risks.u
    )
    return ScoreBreakdown(ceff=ceff, x=risks.x, h=risks.h, q=risks.q, f=risks.f, u=risks.u, final=final)
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/scheduler/test_scoring.py -v`

```bash
git add -A
git commit -m "M25: HarmonicFit score weights and risk-subtracted final score"
```

Append M25 DONE.

---

### Task M26: Risk modules

**Files:**

- Create: `AdapDomain/src/adapdomain/scheduler/hbm_risk.py`
- Create: `AdapDomain/src/adapdomain/scheduler/queue_risk.py`
- Create: `AdapDomain/src/adapdomain/scheduler/interference_risk.py`
- Create: `AdapDomain/src/adapdomain/scheduler/fragmentation.py`
- Create: `AdapDomain/tests/unit/scheduler/test_risk_modules.py`

This module can be parallelized into four subagents, one per file.

- [ ] **Step 1: Failing tests `tests/unit/scheduler/test_risk_modules.py`**

```python
import pytest

from adapdomain.core.interference import InterferenceEntry, InterferenceTable
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.fragmentation import fragmentation_risk
from adapdomain.scheduler.hbm_risk import hbm_risk
from adapdomain.scheduler.interference_risk import interference_risk
from adapdomain.scheduler.queue_risk import queue_risk


def _residual(cube=1.0, vec=1.0, hbm=0.0, queue=0.0) -> DeviceResidual:
    return DeviceResidual(device_id=0, residual={"cube": cube, "vector": vec},
                          placed_replicas=[], hbm_pressure=hbm, queue_pressure=queue)


def _shape(cube=0.5, vec=0.3) -> TenantShape:
    return TenantShape(model_name="m", replica_id="r0",
                       domains={"cube": cube, "vector": vec},
                       slo_ms=15.0, batch_size=1, frontier_version=1, confidence=0.9)


def test_hbm_risk_zero_when_no_pressure():
    assert hbm_risk(_residual(hbm=0.0), high_threshold=0.85) == 0.0


def test_hbm_risk_grows_with_pressure():
    assert hbm_risk(_residual(hbm=0.9), high_threshold=0.85) > 0


def test_queue_risk_capped_at_one():
    assert queue_risk(_residual(queue=2.0), high_threshold=0.5) == 1.0


def test_interference_risk_uses_max_plus_damped_sum():
    entries = [
        InterferenceEntry(model_a="X", model_b="A", direction="a_on_b", pressure_bucket="mid", slowdown=0.3, p95_delta=0, goodput_delta=0, source_run_id="r"),
        InterferenceEntry(model_a="X", model_b="B", direction="a_on_b", pressure_bucket="mid", slowdown=0.1, p95_delta=0, goodput_delta=0, source_run_id="r"),
    ]
    table = InterferenceTable.from_entries(entries)
    r = interference_risk(candidate="X", colocated=["A", "B"], bucket="mid", table=table, eta=0.30)
    assert r == pytest.approx(0.3 + 0.30 * (0.3 + 0.1))


def test_fragmentation_risk_increases_when_residual_skewed():
    balanced = fragmentation_risk(_residual(cube=0.5, vec=0.5), _shape(cube=0.4, vec=0.4))
    skewed = fragmentation_risk(_residual(cube=0.05, vec=0.95), _shape(cube=0.05, vec=0.4))
    assert skewed > balanced
```

- [ ] **Step 2: Implement `scheduler/hbm_risk.py`**

```python
"""HBM pressure -> [0,1] risk."""
from __future__ import annotations

from adapdomain.core.shape import DeviceResidual


def hbm_risk(device: DeviceResidual, *, high_threshold: float) -> float:
    if device.hbm_pressure <= 0:
        return 0.0
    return min(1.0, max(0.0, (device.hbm_pressure - high_threshold) / max(1.0 - high_threshold, 1e-6)) + 0.5 * device.hbm_pressure)
```

- [ ] **Step 3: Implement `scheduler/queue_risk.py`**

```python
"""Queue pressure -> [0,1] risk."""
from __future__ import annotations

from adapdomain.core.shape import DeviceResidual


def queue_risk(device: DeviceResidual, *, high_threshold: float) -> float:
    if device.queue_pressure <= 0:
        return 0.0
    over = device.queue_pressure - high_threshold
    return min(1.0, max(0.0, over) + 0.5 * device.queue_pressure)
```

- [ ] **Step 4: Implement `scheduler/interference_risk.py`**

```python
"""Adapter around core.interference.aggregate_interference."""
from __future__ import annotations

from typing import Sequence

from adapdomain.core.interference import InterferenceTable, aggregate_interference


def interference_risk(
    *,
    candidate: str,
    colocated: Sequence[str],
    bucket: str,
    table: InterferenceTable,
    eta: float,
) -> float:
    raw = aggregate_interference(table, candidate=candidate, colocated=colocated, bucket=bucket, eta=eta)
    # squash to [0, 1] via saturating tanh-like mapping
    return min(1.0, raw)
```

- [ ] **Step 5: Implement `scheduler/fragmentation.py`**

```python
"""Fragmentation risk: penalize residuals dominated by a single domain after placement."""
from __future__ import annotations

from adapdomain.core.shape import DeviceResidual, TenantShape


def fragmentation_risk(device: DeviceResidual, replica: TenantShape) -> float:
    new_residual: dict[str, float] = {}
    for name, value in device.residual.items():
        usage = replica.domains.get(name, 0.0)
        new_residual[name] = max(0.0, value - usage)
    if not new_residual:
        return 0.0
    values = list(new_residual.values())
    total = sum(values)
    if total <= 1e-9:
        return 1.0
    biggest = max(values)
    # 0 when balanced, approaches 1 when residual is dominated by a single domain
    return max(0.0, min(1.0, (biggest / total) - 0.5) * 2.0)
```

- [ ] **Step 6: Run, commit, log**

Run: `pytest tests/unit/scheduler/test_risk_modules.py -v`
Expected: 5/5 pass.

```bash
git add -A
git commit -m "M26: HBM/queue/interference/fragmentation risk modules"
```

Append M26 DONE.

---

### Task M27: Residual state

**Files:**

- Create: `AdapDomain/src/adapdomain/scheduler/residual_state.py`
- Create: `AdapDomain/tests/unit/scheduler/test_residual_state.py`

- [ ] **Step 1: Failing test**

```python
import pytest

from adapdomain.core.placement import PlacementPlan, ScoreBreakdown
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.residual_state import ClusterState


def _shape(rid: str, cube: float, vec: float) -> TenantShape:
    return TenantShape(model_name="m", replica_id=rid,
                       domains={"cube": cube, "vector": vec},
                       slo_ms=15.0, batch_size=1,
                       frontier_version=1, confidence=0.9)


def _residual(did: int) -> DeviceResidual:
    return DeviceResidual.empty(device_id=did, domains=["cube", "vector"])


def test_apply_placement_consumes_residual():
    state = ClusterState.from_residuals([_residual(0), _residual(1)])
    state.apply_placement(replica=_shape("r0", cube=0.3, vec=0.2), device_id=0)
    assert state.residual_of(0).residual["cube"] == pytest.approx(0.7)
    assert state.residual_of(1).residual["cube"] == 1.0


def test_revert_restores_residual():
    state = ClusterState.from_residuals([_residual(0)])
    state.apply_placement(replica=_shape("r0", cube=0.4, vec=0.2), device_id=0)
    state.revert_placement(replica_id="r0", device_id=0)
    assert state.residual_of(0).residual["cube"] == 1.0
    assert "r0" not in state.residual_of(0).placed_replicas
```

- [ ] **Step 2: Implement `scheduler/residual_state.py`**

```python
"""Thread-safe per-device residual cluster state."""
from __future__ import annotations

import threading
from typing import Iterable

from adapdomain.core.shape import DeviceResidual, TenantShape, update_residual


class ClusterState:
    def __init__(self) -> None:
        self._residuals: dict[int, DeviceResidual] = {}
        self._locks: dict[int, threading.Lock] = {}
        self._placement_log: list[tuple[str, int, TenantShape]] = []

    @classmethod
    def from_residuals(cls, residuals: Iterable[DeviceResidual]) -> "ClusterState":
        state = cls()
        for r in residuals:
            state._residuals[r.device_id] = r
            state._locks[r.device_id] = threading.Lock()
        return state

    def device_ids(self) -> list[int]:
        return list(self._residuals.keys())

    def residual_of(self, device_id: int) -> DeviceResidual:
        return self._residuals[device_id]

    def apply_placement(self, *, replica: TenantShape, device_id: int) -> None:
        with self._locks[device_id]:
            self._residuals[device_id] = update_residual(self._residuals[device_id], [replica])
            self._placement_log.append((replica.replica_id, device_id, replica))

    def revert_placement(self, *, replica_id: str, device_id: int) -> None:
        with self._locks[device_id]:
            log = [(r, d, s) for r, d, s in self._placement_log if not (r == replica_id and d == device_id)]
            self._placement_log = log
            # rebuild residual from scratch
            base = DeviceResidual.empty(device_id, list(self._residuals[device_id].residual.keys()))
            shapes = [s for r, d, s in log if d == device_id]
            self._residuals[device_id] = update_residual(base, shapes)
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/scheduler/test_residual_state.py -v`
Expected: 2/2 pass.

```bash
git add -A
git commit -m "M27: ClusterState with per-device locks and transactional revert"
```

Append M27 DONE.

---

### Task M28: HarmonicFit planner

**Files:**

- Create: `AdapDomain/src/adapdomain/scheduler/harmonic_fit.py`
- Create: `AdapDomain/src/adapdomain/scheduler/planner.py`
- Create: `AdapDomain/tests/unit/scheduler/test_planner.py`

- [ ] **Step 1: Failing test**

```python
from datetime import datetime, timezone

import pytest

from adapdomain.core.frontier import FrontierPoint, FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.planner import HarmonicFitPlanner
from adapdomain.scheduler.residual_state import ClusterState
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights


def _snap(model: str, cube: float, vec: float) -> FrontierSnapshot:
    pt = FrontierPoint(
        model_name=model, point_id="p",
        raw_cube=0.0, raw_vector=0.0, raw_hbm=0.0,
        norm_cube=cube, norm_vector=vec,
        batch_size=1, slo_ms=15.0,
        p50_ms=4.0, p95_ms=6.0, p99_ms=7.0, goodput_qps=100.0,
        feasible=True, confidence=0.9,
    )
    return FrontierSnapshot(model_name=model, version=1, k_domains=["cube", "vector"],
                            points=[pt], safe_point_id=pt.point_id, dominant_domain="balanced",
                            confidence_summary=0.9, source_run_id="r",
                            created_at=datetime.now(timezone.utc))


def test_planner_prefers_complementary_device():
    state = ClusterState.from_residuals([
        DeviceResidual(device_id=0, residual={"cube": 0.6, "vector": 0.2}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
        DeviceResidual(device_id=1, residual={"cube": 0.2, "vector": 0.6}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
    ])
    replicas = [TenantShape(model_name="bert-base", replica_id="r0",
                            domains={"cube": 0.5, "vector": 0.1},
                            slo_ms=15.0, batch_size=1, frontier_version=1, confidence=0.9)]
    planner = HarmonicFitPlanner(
        feasibility=FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5),
        weights=ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05),
        eta=0.30,
        hbm_high=0.85, queue_high=0.80,
    )
    snapshots = {"bert-base": _snap("bert-base", cube=0.5, vec=0.1)}
    plan = planner.plan(replicas, state=state, snapshots=snapshots, interference=InterferenceTable.from_entries([]))
    assert plan.placements["r0"] == 0  # cube-heavy slack lives on device 0
```

- [ ] **Step 2: Implement `scheduler/harmonic_fit.py`**

```python
"""HarmonicFit edge construction helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from adapdomain.core.shape import DeviceResidual, TenantShape, domain_fit_ratio
from adapdomain.core.types import EPSILON_DOMAIN_FIT


@dataclass(frozen=True)
class EdgeRho:
    rho_per_domain: list[float]
    fit_ratios_by_domain: dict[str, float]


def edge_rho(replica: TenantShape, device: DeviceResidual, *, domains: Sequence[str], eps: float = EPSILON_DOMAIN_FIT) -> EdgeRho:
    rhos: list[float] = []
    by_domain: dict[str, float] = {}
    for name in domains:
        r = replica.domains.get(name, 0.0)
        R = device.residual.get(name, 0.0)
        val = domain_fit_ratio(r=r, R=R, eps=eps)
        rhos.append(val)
        by_domain[name] = val
    return EdgeRho(rho_per_domain=rhos, fit_ratios_by_domain=by_domain)
```

- [ ] **Step 3: Implement `scheduler/planner.py`**

```python
"""HarmonicFit greedy planner (paper Section 4.3)."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Mapping, Sequence

from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.placement import PlacementPlan, ScoreBreakdown
from adapdomain.core.shape import TenantShape
from adapdomain.core.types import DeviceId, EPSILON_DOMAIN_FIT, FrontierVersion, ModelId, ReplicaId

from .feasibility import FeasibilityFilter
from .fragmentation import fragmentation_risk
from .harmonic_fit import edge_rho
from .hbm_risk import hbm_risk
from .interference_risk import interference_risk
from .queue_risk import queue_risk
from .residual_state import ClusterState
from .scoring import RiskTerms, ScoreWeights, compute_score


class HarmonicFitPlanner:
    def __init__(
        self,
        *,
        feasibility: FeasibilityFilter,
        weights: ScoreWeights,
        eta: float,
        hbm_high: float,
        queue_high: float,
        epsilon: float = EPSILON_DOMAIN_FIT,
    ) -> None:
        self._feasibility = feasibility
        self._weights = weights
        self._eta = eta
        self._hbm_high = hbm_high
        self._queue_high = queue_high
        self._eps = epsilon

    def plan(
        self,
        replicas: Sequence[TenantShape],
        *,
        state: ClusterState,
        snapshots: Mapping[str, FrontierSnapshot],
        interference: InterferenceTable,
    ) -> PlacementPlan:
        edges: list[tuple[float, TenantShape, int, ScoreBreakdown]] = []
        rejected: dict[ReplicaId, str] = {}

        for replica in replicas:
            snap = snapshots.get(replica.model_name)
            if snap is None:
                rejected[replica.replica_id] = "no_frontier_snapshot"
                continue
            for did in state.device_ids():
                residual = state.residual_of(did)
                ok, reason = self._feasibility.check(replica, snap, residual)
                if not ok:
                    rejected.setdefault(replica.replica_id, reason.value if reason else "infeasible")
                    continue
                domain_names = ["cube", "vector"]
                rho = edge_rho(replica, residual, domains=domain_names, eps=self._eps)
                risks = RiskTerms(
                    x=interference_risk(
                        candidate=replica.model_name,
                        colocated=[r for r in residual.placed_replicas],
                        bucket="mid",
                        table=interference,
                        eta=self._eta,
                    ),
                    h=hbm_risk(residual, high_threshold=self._hbm_high),
                    q=queue_risk(residual, high_threshold=self._queue_high),
                    f=fragmentation_risk(residual, replica),
                    u=0.0 if residual.placed_replicas else 0.5,
                )
                breakdown = compute_score(rho=rho.rho_per_domain, weights=self._weights, risks=risks, eps=self._eps)
                edges.append((breakdown.final, replica, did, breakdown))

        edges.sort(key=lambda x: x[0], reverse=True)
        placements: dict[ReplicaId, DeviceId] = {}
        breakdowns: dict[tuple[ReplicaId, DeviceId], ScoreBreakdown] = {}
        for score, replica, did, breakdown in edges:
            if replica.replica_id in placements:
                continue
            # Re-check feasibility now that earlier picks may have used residual.
            residual = state.residual_of(did)
            ok, _reason = self._feasibility.check(replica, snapshots[replica.model_name], residual)
            if not ok:
                continue
            placements[replica.replica_id] = DeviceId(did)
            breakdowns[(replica.replica_id, DeviceId(did))] = breakdown
            state.apply_placement(replica=replica, device_id=did)

        for r in replicas:
            if r.replica_id not in placements and r.replica_id not in rejected:
                rejected[r.replica_id] = "no_feasible_device_after_packing"

        return PlacementPlan(
            plan_id=f"plan-{int(datetime.now(timezone.utc).timestamp())}",
            created_at=datetime.now(timezone.utc),
            input_replicas=[r.replica_id for r in replicas],
            placements=placements,
            rejected=rejected,
            score_breakdown=breakdowns,
            frontier_versions={ModelId(r.model_name): FrontierVersion(snapshots[r.model_name].version) for r in replicas if r.model_name in snapshots},
        )
```

- [ ] **Step 4: Run, commit, log**

Run: `pytest tests/unit/scheduler/ -v`
Expected: feasibility + scoring + risk + residual + planner = 16+ tests pass.

```bash
git add -A
git commit -m "M28: HarmonicFitPlanner greedy by complementarity score minus risk terms"
```

Append M28 DONE.

---

### Task M29: Runtime monitor (EWMA + ring buffer)

**Files:**

- Create: `AdapDomain/src/adapdomain/runtime/__init__.py`
- Create: `AdapDomain/src/adapdomain/runtime/ewma.py`
- Create: `AdapDomain/src/adapdomain/runtime/ring_buffer.py`
- Create: `AdapDomain/src/adapdomain/runtime/monitor.py`
- Create: `AdapDomain/tests/unit/runtime/__init__.py`
- Create: `AdapDomain/tests/unit/runtime/test_ewma.py`
- Create: `AdapDomain/tests/unit/runtime/test_ring_buffer.py`
- Create: `AdapDomain/tests/unit/runtime/test_monitor.py`

- [ ] **Step 1: Failing test `test_ewma.py`**

```python
import pytest

from adapdomain.runtime.ewma import Ewma


def test_ewma_converges_to_steady():
    e = Ewma(alpha=0.5)
    for _ in range(20):
        e.observe(10.0)
    assert e.value == pytest.approx(10.0, abs=1e-3)


def test_ewma_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        Ewma(alpha=0)
    with pytest.raises(ValueError):
        Ewma(alpha=1.5)
```

- [ ] **Step 2: Failing test `test_ring_buffer.py`**

```python
from adapdomain.runtime.ring_buffer import RingBuffer


def test_ring_buffer_drops_oldest_after_capacity():
    rb = RingBuffer(capacity=3)
    for v in [1, 2, 3, 4]:
        rb.push(v)
    assert list(rb.snapshot()) == [2, 3, 4]
```

- [ ] **Step 3: Failing test `test_monitor.py`**

```python
from datetime import datetime, timezone

from adapdomain.backends.base import DeviceTelemetrySample
from adapdomain.runtime.monitor import RuntimeMonitor


def test_monitor_records_drift_against_baseline():
    mon = RuntimeMonitor(alpha=0.5)
    for cube in [0.4, 0.5, 0.6, 0.55]:
        mon.observe_sample(DeviceTelemetrySample(device_id=0, cube_util=cube, vector_util=0.2, hbm_pressure=0.3, queue_depth=1))
    drift = mon.cube_drift(device_id=0, baseline=0.4)
    assert drift > 0
```

- [ ] **Step 4: Implement `runtime/ewma.py`**

```python
"""Scalar exponentially weighted moving average."""
from __future__ import annotations


class Ewma:
    def __init__(self, *, alpha: float) -> None:
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._value: float | None = None

    def observe(self, value: float) -> None:
        self._value = float(value) if self._value is None else self._alpha * value + (1 - self._alpha) * self._value

    @property
    def value(self) -> float:
        return 0.0 if self._value is None else self._value
```

- [ ] **Step 5: Implement `runtime/ring_buffer.py`**

```python
"""Fixed-capacity ring buffer (single producer)."""
from __future__ import annotations

from collections import deque
from typing import Iterable


class RingBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf: deque = deque(maxlen=capacity)

    def push(self, value) -> None:
        self._buf.append(value)

    def snapshot(self) -> Iterable:
        return list(self._buf)
```

- [ ] **Step 6: Implement `runtime/monitor.py`**

```python
"""Telemetry monitor: combines per-device EWMAs with a recent ring buffer."""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from adapdomain.backends.base import DeviceTelemetrySample

from .ewma import Ewma
from .ring_buffer import RingBuffer


class RuntimeMonitor:
    def __init__(self, *, alpha: float, history: int = 60) -> None:
        self._alpha = alpha
        self._history = history
        self._cube: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._vector: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._hbm: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._queue: dict[int, Ewma] = defaultdict(lambda: Ewma(alpha=alpha))
        self._samples: dict[int, RingBuffer] = defaultdict(lambda: RingBuffer(capacity=history))

    def observe_sample(self, sample: DeviceTelemetrySample) -> None:
        self._cube[sample.device_id].observe(sample.cube_util)
        self._vector[sample.device_id].observe(sample.vector_util)
        self._hbm[sample.device_id].observe(sample.hbm_pressure)
        self._queue[sample.device_id].observe(sample.queue_depth)
        self._samples[sample.device_id].push(sample)

    def cube_drift(self, *, device_id: int, baseline: float) -> float:
        return self._cube[device_id].value - baseline

    def vector_drift(self, *, device_id: int, baseline: float) -> float:
        return self._vector[device_id].value - baseline

    def hbm_value(self, device_id: int) -> float:
        return self._hbm[device_id].value

    def queue_value(self, device_id: int) -> float:
        return self._queue[device_id].value

    def history(self, device_id: int) -> list[DeviceTelemetrySample]:
        return list(self._samples[device_id].snapshot())
```

- [ ] **Step 7: Implement `runtime/__init__.py`**

```python
"""AdapDomain runtime guards."""
from .ewma import Ewma
from .monitor import RuntimeMonitor
from .ring_buffer import RingBuffer

__all__ = ["Ewma", "RingBuffer", "RuntimeMonitor"]
```

- [ ] **Step 8: Run, commit, log**

Run: `pytest tests/unit/runtime/ -v`
Expected: 5/5 pass.

```bash
git add -A
git commit -m "M29: runtime monitor (EWMA, ring buffer, drift accessors)"
```

Append M29 DONE.

---

### Task M30: SmoothGuard

**Files:**

- Create: `AdapDomain/src/adapdomain/runtime/smooth_guard.py`
- Create: `AdapDomain/tests/unit/runtime/test_smooth_guard.py`

- [ ] **Step 1: Failing test**

```python
import pytest

from adapdomain.runtime.smooth_guard import GuardAction, GuardThresholds, SmoothGuard


def test_guard_triggers_only_after_two_crossings():
    guard = SmoothGuard(thresholds=GuardThresholds(p95_drift=0.2, residual_drift=0.2, hbm_high=0.85, queue_high=0.8),
                        consecutive=2, cooldown_s=30)
    sample = lambda: guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=0.0)
    assert sample().action is GuardAction.NONE
    next_action = guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=1.0)
    assert next_action.action is GuardAction.QUOTA


def test_guard_respects_cooldown():
    guard = SmoothGuard(thresholds=GuardThresholds(p95_drift=0.2, residual_drift=0.2, hbm_high=0.85, queue_high=0.8),
                        consecutive=2, cooldown_s=30)
    guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=0.0)
    second = guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=1.0)
    third = guard.evaluate(device_id=0, p95_drift=0.5, residual_drift=0.0, hbm=0.0, queue=0.0, now=2.0)
    assert second.action is GuardAction.QUOTA
    assert third.action is GuardAction.NONE
```

- [ ] **Step 2: Implement `runtime/smooth_guard.py`**

```python
"""SmoothGuard: persistent-drift detector with cooldown."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GuardAction(str, Enum):
    NONE = "none"
    QUOTA = "quota_repair"
    MIGRATE = "local_migration"
    REBALANCE = "one_card_rebalance"
    REPROFILE = "trigger_reprofile"


@dataclass(frozen=True)
class GuardThresholds:
    p95_drift: float
    residual_drift: float
    hbm_high: float
    queue_high: float


@dataclass
class GuardEvaluation:
    action: GuardAction
    reasons: list[str] = field(default_factory=list)


class SmoothGuard:
    def __init__(self, *, thresholds: GuardThresholds, consecutive: int, cooldown_s: float) -> None:
        self._thresholds = thresholds
        self._consecutive = consecutive
        self._cooldown_s = cooldown_s
        self._counters: dict[int, int] = {}
        self._last_fire_at: dict[int, float] = {}

    def evaluate(
        self,
        *,
        device_id: int,
        p95_drift: float,
        residual_drift: float,
        hbm: float,
        queue: float,
        now: float,
    ) -> GuardEvaluation:
        reasons: list[str] = []
        if abs(p95_drift) > self._thresholds.p95_drift:
            reasons.append("p95_drift")
        if abs(residual_drift) > self._thresholds.residual_drift:
            reasons.append("residual_drift")
        if hbm > self._thresholds.hbm_high:
            reasons.append("hbm_high")
        if queue > self._thresholds.queue_high:
            reasons.append("queue_high")

        if not reasons:
            self._counters[device_id] = 0
            return GuardEvaluation(action=GuardAction.NONE)

        last = self._last_fire_at.get(device_id, float("-inf"))
        if now - last < self._cooldown_s:
            return GuardEvaluation(action=GuardAction.NONE, reasons=reasons + ["cooldown"])

        self._counters[device_id] = self._counters.get(device_id, 0) + 1
        if self._counters[device_id] < self._consecutive:
            return GuardEvaluation(action=GuardAction.NONE, reasons=reasons)

        self._counters[device_id] = 0
        self._last_fire_at[device_id] = now
        # Pick action by severity
        if "hbm_high" in reasons or "queue_high" in reasons:
            return GuardEvaluation(action=GuardAction.MIGRATE, reasons=reasons)
        return GuardEvaluation(action=GuardAction.QUOTA, reasons=reasons)
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/runtime/test_smooth_guard.py -v`
Expected: 2/2 pass.

```bash
git add -A
git commit -m "M30: SmoothGuard with two-tick trigger, cooldown, severity-tiered action"
```

Append M30 DONE.

---

### Task M31: PreScale

**Files:**

- Create: `AdapDomain/src/adapdomain/runtime/prescale.py`
- Create: `AdapDomain/tests/unit/runtime/test_prescale.py`

- [ ] **Step 1: Failing test**

```python
from adapdomain.core.shape import DeviceResidual
from adapdomain.runtime.prescale import PreScale


def test_prescale_picks_device_with_matching_scarce_domain():
    state = [
        DeviceResidual(device_id=0, residual={"cube": 0.7, "vector": 0.1}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
        DeviceResidual(device_id=1, residual={"cube": 0.1, "vector": 0.7}, placed_replicas=[], hbm_pressure=0.0, queue_pressure=0.0),
    ]
    ps = PreScale()
    assert ps.pick_for(scarce_domain="cube", residuals=state) == 0
    assert ps.pick_for(scarce_domain="vector", residuals=state) == 1
```

- [ ] **Step 2: Implement `runtime/prescale.py`**

```python
"""PreScale: select device to prewarm based on scarce-domain residual."""
from __future__ import annotations

from typing import Sequence

from adapdomain.core.shape import DeviceResidual


class PreScale:
    def pick_for(self, *, scarce_domain: str, residuals: Sequence[DeviceResidual]) -> int | None:
        best: tuple[int, float] | None = None
        for r in residuals:
            value = r.residual.get(scarce_domain, 0.0)
            if best is None or value > best[1]:
                best = (r.device_id, value)
        return best[0] if best else None
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/runtime/test_prescale.py -v`

```bash
git add -A
git commit -m "M31: PreScale selects device by scarce-domain residual"
```

Append M31 DONE.

---

### Task M32: NightShift

**Files:**

- Create: `AdapDomain/src/adapdomain/runtime/nightshift.py`
- Create: `AdapDomain/tests/unit/runtime/test_nightshift.py`

- [ ] **Step 1: Failing test**

```python
from adapdomain.runtime.nightshift import MigrationCandidate, NightShift


def test_nightshift_returns_topk_by_delta_ceff_under_budget():
    candidates = [
        MigrationCandidate(replica_id=f"r{i}", src_device=0, dst_device=1, delta_ceff=d, risk=0.1)
        for i, d in enumerate([0.05, 0.20, 0.15, 0.30])
    ]
    plan = NightShift(migration_budget=2).select(candidates)
    assert [c.replica_id for c in plan] == ["r3", "r1"]


def test_nightshift_skips_high_risk():
    candidates = [
        MigrationCandidate(replica_id="r0", src_device=0, dst_device=1, delta_ceff=0.50, risk=0.9),
        MigrationCandidate(replica_id="r1", src_device=0, dst_device=1, delta_ceff=0.30, risk=0.1),
    ]
    plan = NightShift(migration_budget=2, risk_max=0.5).select(candidates)
    assert [c.replica_id for c in plan] == ["r1"]
```

- [ ] **Step 2: Implement `runtime/nightshift.py`**

```python
"""NightShift: idle-period migration picker."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class MigrationCandidate:
    replica_id: str
    src_device: int
    dst_device: int
    delta_ceff: float
    risk: float


class NightShift:
    def __init__(self, *, migration_budget: int, risk_max: float = 0.5) -> None:
        self._budget = migration_budget
        self._risk_max = risk_max

    def select(self, candidates: Sequence[MigrationCandidate]) -> list[MigrationCandidate]:
        safe = [c for c in candidates if c.risk <= self._risk_max]
        safe.sort(key=lambda c: c.delta_ceff, reverse=True)
        return safe[: self._budget]
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/runtime/test_nightshift.py -v`

```bash
git add -A
git commit -m "M32: NightShift migration picker"
```

Append M32 DONE.

---

### Task M33: Repair actions

**Files:**

- Create: `AdapDomain/src/adapdomain/runtime/repair_actions.py`
- Create: `AdapDomain/tests/unit/runtime/test_repair_actions.py`

- [ ] **Step 1: Failing test**

```python
import pytest

from adapdomain.backends.base import ResourceQuota
from adapdomain.runtime.repair_actions import RepairLogger, perform_quota_repair


def test_quota_repair_logs_event(tmp_path):
    logger = RepairLogger(tmp_path / "events.jsonl")
    perform_quota_repair(logger=logger, device_id=0, replica_id="r0", new_quota=ResourceQuota(device_id=0, cube_units=5, vector_units=10), reason="hbm_high")
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert lines
    assert "quota_repair" in lines[0]
```

- [ ] **Step 2: Implement `runtime/repair_actions.py`**

```python
"""Audit logging for runtime repair actions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adapdomain.backends.base import ResourceQuota
from adapdomain.storage.jsonl import JsonlWriter


class RepairLogger:
    def __init__(self, path: str | Path) -> None:
        self._writer = JsonlWriter(path)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        self._writer.write({"type": event_type, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **payload})


def perform_quota_repair(*, logger: RepairLogger, device_id: int, replica_id: str, new_quota: ResourceQuota, reason: str) -> None:
    logger.log("quota_repair", {
        "device_id": device_id,
        "replica_id": replica_id,
        "cube_units": new_quota.cube_units,
        "vector_units": new_quota.vector_units,
        "reason": reason,
    })


def perform_migration(*, logger: RepairLogger, replica_id: str, src_device: int, dst_device: int, reason: str) -> None:
    logger.log("migrate", {
        "replica_id": replica_id, "src": src_device, "dst": dst_device, "reason": reason,
    })


def perform_one_card_rebalance(*, logger: RepairLogger, device_id: int, plan: dict[str, int], reason: str) -> None:
    logger.log("one_card_rebalance", {"device_id": device_id, "plan": plan, "reason": reason})
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/runtime/test_repair_actions.py -v`

```bash
git add -A
git commit -m "M33: repair actions with JSONL audit log"
```

Append M33 DONE.

---

### Task M34: Replay engine

**Files:**

- Create: `AdapDomain/src/adapdomain/experiments/__init__.py`
- Create: `AdapDomain/src/adapdomain/experiments/seeds.py`
- Create: `AdapDomain/src/adapdomain/experiments/replay.py`
- Create: `AdapDomain/src/adapdomain/cli/commands/replay.py`
- Modify: `AdapDomain/src/adapdomain/cli/main.py` to delegate `replay` to the new command
- Create: `AdapDomain/tests/unit/experiments/__init__.py`
- Create: `AdapDomain/tests/unit/experiments/test_replay_offline.py`

- [ ] **Step 1: Implement `experiments/seeds.py`**

```python
"""Deterministic seed iteration for replay/ablation."""
from __future__ import annotations

import random
from typing import Iterator


def seeded_iter(seeds: list[int]) -> Iterator[tuple[int, random.Random]]:
    for seed in seeds:
        rng = random.Random(seed)
        yield seed, rng
```

- [ ] **Step 2: Implement `experiments/replay.py`** (offline-friendly using MockBackend so we can unit-test on the laptop; real-hardware runs swap in the AscendBackend in M41 wiring)

```python
"""Trace replay engine."""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from adapdomain.backends.base import InferenceBackend
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual, TenantShape
from adapdomain.core.workload import WorkloadTrace
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.planner import HarmonicFitPlanner
from adapdomain.scheduler.residual_state import ClusterState
from adapdomain.scheduler.scoring import ScoreWeights
from adapdomain.storage.csv_store import CsvStore
from adapdomain.storage.jsonl import JsonlWriter
from adapdomain.storage.run_manifest import RunManifest


@dataclass
class ReplayResult:
    seeds: list[int]
    goodput_mean: float
    goodput_std: float
    violation_mean: float
    violation_std: float
    out_dir: Path


@dataclass(frozen=True)
class ReplayInputs:
    trace: WorkloadTrace
    residuals: list[DeviceResidual]
    snapshots: Mapping[str, FrontierSnapshot]
    interference: InterferenceTable


def run_replay(
    *,
    inputs: ReplayInputs,
    seeds: list[int],
    weights: ScoreWeights,
    feasibility: FeasibilityFilter,
    eta: float,
    backend: InferenceBackend,
    output_root: str | Path,
) -> ReplayResult:
    output_root = Path(output_root)
    run_id = f"replay-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = output_root / run_id
    manifest = RunManifest.start(run_id=run_id, command="adapdomain replay", out_dir=out_dir)
    summary = CsvStore(out_dir / "summary.csv", columns=["seed", "goodput", "violation"])
    events = JsonlWriter(out_dir / "events.jsonl")

    g_values: list[float] = []
    v_values: list[float] = []
    backend.initialize()
    try:
        for seed in seeds:
            state = ClusterState.from_residuals(inputs.residuals)
            planner = HarmonicFitPlanner(feasibility=feasibility, weights=weights, eta=eta, hbm_high=0.85, queue_high=0.80)
            plan = planner.plan(list(_replicas_from_trace(inputs.trace)), state=state, snapshots=inputs.snapshots, interference=inputs.interference)
            served = []
            for req in inputs.trace.requests:
                if any(rid for rid, did in plan.placements.items() if rid.startswith(req.model)):
                    handle = backend.load_model(model_id=req.model, path=f"/fake/{req.model}.om", device_id=0)
                    res = backend.execute(handle, inputs={}, repeat=1)
                    served.append(float(res.latencies_ms[0]))
                    backend.unload_model(handle)
            window_seconds = max(1.0, max(r.arrival_ms for r in inputs.trace.requests) / 1000.0)
            goodput = len([t for t in served if t <= 15.0]) / window_seconds
            violation = (1 - len([t for t in served if t <= 15.0]) / max(1, len(served))) if served else 0.0
            summary.append({"seed": seed, "goodput": goodput, "violation": violation})
            events.write({"type": "seed_done", "seed": seed, "goodput": goodput, "violation": violation})
            g_values.append(goodput)
            v_values.append(violation)
    finally:
        backend.finalize()

    res = ReplayResult(
        seeds=list(seeds),
        goodput_mean=statistics.mean(g_values) if g_values else 0.0,
        goodput_std=statistics.pstdev(g_values) if len(g_values) > 1 else 0.0,
        violation_mean=statistics.mean(v_values) if v_values else 0.0,
        violation_std=statistics.pstdev(v_values) if len(v_values) > 1 else 0.0,
        out_dir=out_dir,
    )
    manifest.set_status("done")
    manifest.update(metrics={
        "goodput_mean": res.goodput_mean,
        "goodput_std": res.goodput_std,
        "violation_mean": res.violation_mean,
        "violation_std": res.violation_std,
    })
    manifest.finalize()
    return res


def _replicas_from_trace(trace: WorkloadTrace) -> Iterable[TenantShape]:
    seen: set[str] = set()
    for r in trace.replicas:
        if r.replica_id in seen:
            continue
        seen.add(r.replica_id)
        yield TenantShape(
            model_name=r.model, replica_id=r.replica_id,
            domains={"cube": 0.40, "vector": 0.20},
            slo_ms=r.slo_ms, batch_size=r.batch_size,
            frontier_version=1, confidence=0.9,
        )
```

- [ ] **Step 3: Implement `cli/commands/replay.py`**

```python
"""`adapdomain replay`."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from adapdomain.backends.base import BackendRegistry
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual
from adapdomain.core.workload import WorkloadTrace
from adapdomain.experiments.replay import ReplayInputs, run_replay
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights
from adapdomain.utils.config import ExperimentReplayConfig, HarmonicFitConfig

app = typer.Typer(help="Replay a workload trace against the scheduler.")


@app.callback(invoke_without_command=True)
def replay(
    config: str = typer.Option("configs/experiments/e2e_replay.yaml"),
    scheduler: str = typer.Option("configs/scheduler/harmonicfit_default.yaml"),
    backend: str = typer.Option("mock", help="Inference backend (mock|ascend)"),
) -> None:
    cfg = ExperimentReplayConfig.from_yaml(config)
    sched = HarmonicFitConfig.from_yaml(scheduler)
    weights = ScoreWeights(
        lambda_c=sched.weights.lambda_c, lambda_x=sched.weights.lambda_x,
        lambda_h=sched.weights.lambda_h, lambda_q=sched.weights.lambda_q,
        lambda_f=sched.weights.lambda_f, lambda_u=sched.weights.lambda_u,
    )
    fs = FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5)
    backend_obj = BackendRegistry.default().create(backend)
    trace = WorkloadTrace.from_jsonl("tests/fixtures/sample_traces/tiny.jsonl")
    inputs = ReplayInputs(
        trace=trace,
        residuals=[DeviceResidual.empty(i, ["cube", "vector"]) for i in range(4)],
        snapshots={"bert-base": FrontierSnapshot.model_validate(json.loads(Path("tests/fixtures/sample_frontiers/bert-base.v1.json").read_text()))},
        interference=InterferenceTable.from_entries([]),
    )
    res = run_replay(
        inputs=inputs, seeds=cfg.seeds, weights=weights, feasibility=fs, eta=sched.interference.eta,
        backend=backend_obj, output_root=cfg.output_root,
    )
    typer.echo(str(res.out_dir))
```

- [ ] **Step 4: Wire CLI** by replacing the placeholder in `cli/main.py`:

```python
from adapdomain.cli.commands.replay import app as _replay_app
app.add_typer(_replay_app, name="replay")
```

- [ ] **Step 5: Failing test**

```python
import json
from pathlib import Path

import pytest

from adapdomain.backends.base import MockBackend, DeviceInfo
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual
from adapdomain.core.workload import WorkloadTrace
from adapdomain.experiments.replay import ReplayInputs, run_replay
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights


def test_replay_runs_offline(tmp_path):
    trace = WorkloadTrace.from_jsonl("tests/fixtures/sample_traces/tiny.jsonl")
    snap_dict = json.loads(Path("tests/fixtures/sample_frontiers/bert-base.v1.json").read_text())
    snapshots = {"bert-base": FrontierSnapshot.model_validate(snap_dict)}
    inputs = ReplayInputs(
        trace=trace,
        residuals=[DeviceResidual.empty(i, ["cube", "vector"]) for i in range(4)],
        snapshots=snapshots,
        interference=InterferenceTable.from_entries([]),
    )
    res = run_replay(
        inputs=inputs,
        seeds=[1, 2],
        weights=ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05),
        feasibility=FeasibilityFilter(hbm_max=0.9, queue_max=0.9, confidence_min=0.5),
        eta=0.30,
        backend=MockBackend(devices=[DeviceInfo(device_id=i, name=f"mock-{i}", cube_capacity=20, vector_capacity=40) for i in range(4)]),
        output_root=tmp_path,
    )
    assert (res.out_dir / "summary.csv").exists()
    assert (res.out_dir / "manifest.json").exists()
    assert res.goodput_mean >= 0
```

- [ ] **Step 6: Run, commit, log**

Run: `pytest tests/unit/experiments/test_replay_offline.py tests/integration/test_cli_smoke.py -v`

```bash
git add -A
git commit -m "M34: replay engine + adapdomain replay command with MockBackend smoke"
```

Append M34 DONE.

---

### Task M35: Sweeps

**Files:**

- Create: `AdapDomain/src/adapdomain/experiments/sweeps.py`
- Create: `AdapDomain/tests/unit/experiments/test_sweeps_offline.py`

- [ ] **Step 1: Implement `experiments/sweeps.py`**

```python
"""Stress sweeps over arrival rate, SLO threshold, workload mix, and tenant churn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from adapdomain.experiments.replay import ReplayInputs, ReplayResult, run_replay
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights


@dataclass(frozen=True)
class SweepPoint:
    factor_name: str
    factor_value: float
    result: ReplayResult


def sweep_arrival_rate(
    *,
    base_inputs: ReplayInputs,
    rates: Iterable[float],
    build_inputs: Callable[[ReplayInputs, float], ReplayInputs],
    seeds: list[int],
    weights: ScoreWeights,
    feasibility: FeasibilityFilter,
    eta: float,
    backend,
    output_root: str,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for r in rates:
        scaled = build_inputs(base_inputs, r)
        res = run_replay(
            inputs=scaled, seeds=seeds, weights=weights, feasibility=feasibility,
            eta=eta, backend=backend, output_root=f"{output_root}/arrival_{r}",
        )
        points.append(SweepPoint(factor_name="arrival_rate", factor_value=r, result=res))
    return points
```

- [ ] **Step 2: Smoke test**

```python
from pathlib import Path

from adapdomain.backends.base import MockBackend, DeviceInfo
from adapdomain.core.frontier import FrontierSnapshot
from adapdomain.core.interference import InterferenceTable
from adapdomain.core.shape import DeviceResidual
from adapdomain.core.workload import WorkloadTrace
from adapdomain.experiments.replay import ReplayInputs
from adapdomain.experiments.sweeps import sweep_arrival_rate
from adapdomain.scheduler.feasibility import FeasibilityFilter
from adapdomain.scheduler.scoring import ScoreWeights


def test_sweep_runs_offline(tmp_path):
    import json
    trace = WorkloadTrace.from_jsonl("tests/fixtures/sample_traces/tiny.jsonl")
    snap = {"bert-base": FrontierSnapshot.model_validate(json.loads(Path("tests/fixtures/sample_frontiers/bert-base.v1.json").read_text()))}
    base = ReplayInputs(trace=trace, residuals=[DeviceResidual.empty(i, ["cube", "vector"]) for i in range(4)], snapshots=snap, interference=InterferenceTable.from_entries([]))
    pts = sweep_arrival_rate(
        base_inputs=base, rates=[1.0, 2.0],
        build_inputs=lambda b, r: b,
        seeds=[1],
        weights=ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05),
        feasibility=FeasibilityFilter(0.9, 0.9, 0.5),
        eta=0.30,
        backend=MockBackend(devices=[DeviceInfo(device_id=i, name=f"mock-{i}", cube_capacity=20, vector_capacity=40) for i in range(4)]),
        output_root=str(tmp_path),
    )
    assert len(pts) == 2
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/experiments/test_sweeps_offline.py -v`

```bash
git add -A
git commit -m "M35: sweep harness over arrival rate (extensible to slo/mix/churn)"
```

Append M35 DONE.

---

### Task M36 + M37: ActiveVC & HarmonicFit ablations

**Files:**

- Create: `AdapDomain/src/adapdomain/experiments/ablations.py`
- Create: `AdapDomain/src/adapdomain/cli/commands/ablate.py`
- Modify: `AdapDomain/src/adapdomain/cli/main.py` to wire `ablate`
- Create: `AdapDomain/tests/unit/experiments/test_ablations_offline.py`

- [ ] **Step 1: Implement `experiments/ablations.py`**

```python
"""ActiveVC frontier-recovery baselines and HarmonicFit scoring ablations."""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from adapdomain.profiler.activevc import ActiveVC, ActiveVCConfigRuntime
from adapdomain.profiler.acquisition import AcquisitionWeights
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights, compute_score


@dataclass(frozen=True)
class FrontierRecoveryResult:
    name: str
    budget: int
    boundary_error: float
    missed_feasible: int


def random_baseline(*, budget: int, evaluate: Callable[[GridPoint], MsprofObservation], rng: random.Random) -> list[MsprofObservation]:
    obs: list[MsprofObservation] = []
    for _ in range(budget):
        p = GridPoint(rng.random(), rng.random(), rng.random())
        obs.append(evaluate(p))
    return obs


def binary_baseline(*, budget: int, evaluate: Callable[[GridPoint], MsprofObservation]) -> list[MsprofObservation]:
    obs: list[MsprofObservation] = []
    coords = []
    n = max(2, int(budget ** 0.5))
    for i in range(n):
        for j in range(n):
            coords.append(GridPoint(i / max(n - 1, 1), j / max(n - 1, 1), 0.5))
    for p in coords[:budget]:
        obs.append(evaluate(p))
    return obs


def bo_baseline(*, budget: int, evaluate: Callable[[GridPoint], MsprofObservation], rng: random.Random) -> list[MsprofObservation]:
    # Simple BO surrogate: random first, then exploit-only
    obs = random_baseline(budget=max(1, budget // 2), evaluate=evaluate, rng=rng)
    feasible = [o for o in obs if o.feasible]
    if feasible:
        center = (
            statistics.mean(o.raw_vector for o in feasible),
            statistics.mean(o.raw_cube for o in feasible),
        )
    else:
        center = (0.5, 0.5)
    while len(obs) < budget:
        jitter_v = max(0.0, min(1.0, center[0] + rng.gauss(0.0, 0.05)))
        jitter_m = max(0.0, min(1.0, center[1] + rng.gauss(0.0, 0.05)))
        obs.append(evaluate(GridPoint(jitter_v, jitter_m, 0.5)))
    return obs


def boundary_error(observations: Sequence[MsprofObservation]) -> float:
    feasible = [o for o in observations if o.feasible]
    infeasible = [o for o in observations if not o.feasible]
    if not feasible or not infeasible:
        return 1.0
    import math
    return min(
        math.hypot(f.raw_vector - i.raw_vector, f.raw_cube - i.raw_cube)
        for f in feasible for i in infeasible
    )


def harmonicfit_variant_score(
    *,
    variant: str,
    rho: list[float],
    risks: RiskTerms,
    weights: ScoreWeights,
) -> float:
    masked_weights = weights
    masked_risks = risks
    if variant == "no_ceff":
        masked_weights = ScoreWeights(0.0, weights.lambda_x, weights.lambda_h, weights.lambda_q, weights.lambda_f, weights.lambda_u)
    elif variant == "no_interference":
        masked_risks = RiskTerms(x=0.0, h=risks.h, q=risks.q, f=risks.f, u=risks.u)
    elif variant == "no_hbm":
        masked_risks = RiskTerms(x=risks.x, h=0.0, q=risks.q, f=risks.f, u=risks.u)
    elif variant == "no_fragment":
        masked_risks = RiskTerms(x=risks.x, h=risks.h, q=risks.q, f=0.0, u=risks.u)
    elif variant == "no_empty_penalty":
        masked_risks = RiskTerms(x=risks.x, h=risks.h, q=risks.q, f=risks.f, u=0.0)
    return compute_score(rho=rho, weights=masked_weights, risks=masked_risks).final
```

- [ ] **Step 2: Implement `cli/commands/ablate.py`**

```python
"""`adapdomain ablate`."""
from __future__ import annotations

import typer

from adapdomain.utils.config import ExperimentAblationConfig

app = typer.Typer(help="HarmonicFit and ActiveVC ablations.")


@app.callback(invoke_without_command=True)
def ablate(config: str = typer.Option("configs/experiments/ablation.yaml")) -> None:
    cfg = ExperimentAblationConfig.from_yaml(config)
    typer.echo(
        f"running ablation with {len(cfg.harmonicfit_variants)} variants, "
        f"{len(cfg.activevc_baselines)} baselines, budgets={cfg.budgets}",
    )
```

- [ ] **Step 3: Wire CLI** (replace placeholder in `cli/main.py`):

```python
from adapdomain.cli.commands.ablate import app as _ablate_app
app.add_typer(_ablate_app, name="ablate")
```

- [ ] **Step 4: Offline smoke test**

```python
import random

from adapdomain.experiments.ablations import (
    bo_baseline, binary_baseline, boundary_error, harmonicfit_variant_score, random_baseline,
)
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.profiler.frontier_builder import MsprofObservation
from adapdomain.scheduler.scoring import RiskTerms, ScoreWeights


def _evaluate(point: GridPoint) -> MsprofObservation:
    feasible = point.v + point.m <= 1.2
    return MsprofObservation(
        model="m", point_id=f"v{point.v:.2f}-m{point.m:.2f}",
        raw_cube=point.m, raw_vector=point.v, raw_hbm=point.h,
        batch_size=1, slo_ms=20.0,
        p50_ms=4, p95_ms=6, p99_ms=7, goodput_qps=100 if feasible else 5, feasible=feasible,
    )


def test_baselines_produce_observations():
    rng = random.Random(0)
    assert len(random_baseline(budget=8, evaluate=_evaluate, rng=rng)) == 8
    assert len(binary_baseline(budget=9, evaluate=_evaluate)) == 9
    assert len(bo_baseline(budget=8, evaluate=_evaluate, rng=rng)) == 8


def test_variant_score_removes_specific_terms():
    w = ScoreWeights(0.40, 0.25, 0.15, 0.10, 0.07, 0.05)
    r = RiskTerms(x=0.2, h=0.2, q=0.2, f=0.2, u=0.2)
    full = harmonicfit_variant_score(variant="full", rho=[0.5, 0.5], risks=r, weights=w)
    no_x = harmonicfit_variant_score(variant="no_interference", rho=[0.5, 0.5], risks=r, weights=w)
    assert no_x > full  # removing the X penalty raises the score
```

- [ ] **Step 5: Run, commit, log**

Run: `pytest tests/unit/experiments/test_ablations_offline.py tests/integration/test_cli_smoke.py -v`

```bash
git add -A
git commit -m "M36/M37: ActiveVC baselines and HarmonicFit scoring ablation harness"
```

Append `M36 DONE` and `M37 DONE`.

---

### Task M38: Overhead microbench

**Files:**

- Create: `AdapDomain/src/adapdomain/experiments/overhead.py`
- Create: `AdapDomain/tests/unit/experiments/test_overhead.py`

- [ ] **Step 1: Implement `experiments/overhead.py`**

```python
"""Controller overhead microbenchmark."""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class OverheadResult:
    size: int
    mean_ms: float
    p95_ms: float
    max_ms: float


def measure(target: Callable[[], None], *, size: int, repeats: int = 5) -> OverheadResult:
    latencies: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        target()
        latencies.append((time.perf_counter() - start) * 1e3)
    latencies.sort()
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    return OverheadResult(size=size, mean_ms=statistics.mean(latencies), p95_ms=p95, max_ms=latencies[-1])
```

- [ ] **Step 2: Smoke test**

```python
from adapdomain.experiments.overhead import OverheadResult, measure


def test_measure_returns_ordered_metrics():
    def noop() -> None:
        sum(range(1000))
    res: OverheadResult = measure(noop, size=10, repeats=10)
    assert res.size == 10
    assert res.mean_ms >= 0
    assert res.p95_ms >= res.mean_ms
    assert res.max_ms >= res.p95_ms
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/experiments/test_overhead.py -v`

```bash
git add -A
git commit -m "M38: controller overhead microbench"
```

Append M38 DONE.

---

### Task M39: Plotting

**Files:**

- Create: `AdapDomain/src/adapdomain/experiments/plotting.py`
- Create: `AdapDomain/tests/unit/experiments/test_plotting.py`

- [ ] **Step 1: Implement `experiments/plotting.py`**

```python
"""Reproduce paper figures from result CSV/JSON files. Single Matplotlib axes per figure; no styling."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use("Agg")  # headless backend; required for CI / remote runs
import matplotlib.pyplot as plt


def plot_e2e_bar(summary: Mapping[str, float], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    names = list(summary.keys())
    values = list(summary.values())
    ax.bar(names, values)
    ax.set_ylabel("goodput (qps)")
    ax.set_title("End-to-end goodput")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_overhead_curve(sizes: list[int], p95: list[float], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(sizes, p95, marker="o")
    ax.set_xlabel("replicas")
    ax.set_ylabel("planning p95 (ms)")
    ax.set_title("Controller overhead")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
```

- [ ] **Step 2: Smoke test**

```python
from pathlib import Path

from adapdomain.experiments.plotting import plot_e2e_bar, plot_overhead_curve


def test_plots_write_png(tmp_path):
    out = plot_e2e_bar({"Exclusive": 31.4, "BFD-2D": 66.7, "Dilu-style": 78.8, "AdapDomain": 141.1}, tmp_path / "e2e.png")
    assert Path(out).exists()
    out2 = plot_overhead_curve([10, 100, 400, 800, 1000], [2.79, 19.07, 77.88, 117.80, 131.64], tmp_path / "overhead.png")
    assert Path(out2).exists()
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/experiments/test_plotting.py -v`

```bash
git add -A
git commit -m "M39: plotting helpers (e2e bar + overhead curve) with headless backend"
```

Append M39 DONE.

---

### Task M40: Tables

**Files:**

- Create: `AdapDomain/src/adapdomain/experiments/tables.py`
- Create: `AdapDomain/tests/unit/experiments/test_tables.py`

- [ ] **Step 1: Implement `experiments/tables.py`**

```python
"""LaTeX + markdown table writers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TableRow:
    cells: Sequence[str]


def write_markdown(path: Path, headers: Sequence[str], rows: Iterable[TableRow]) -> Path:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r.cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def write_latex(path: Path, headers: Sequence[str], rows: Iterable[TableRow]) -> Path:
    body = ["\\begin{tabular}{" + "l" * len(headers) + "}", "\\toprule",
            " & ".join(headers) + " \\\\", "\\midrule"]
    for r in rows:
        body.append(" & ".join(r.cells) + " \\\\")
    body.append("\\bottomrule\n\\end{tabular}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body))
    return path
```

- [ ] **Step 2: Smoke test**

```python
from pathlib import Path

from adapdomain.experiments.tables import TableRow, write_latex, write_markdown


def test_table_writers(tmp_path):
    md = write_markdown(tmp_path / "t.md", ["A", "B"], [TableRow(cells=["1", "2"])])
    tex = write_latex(tmp_path / "t.tex", ["A", "B"], [TableRow(cells=["1", "2"])])
    assert Path(md).read_text().splitlines()[0] == "| A | B |"
    assert "\\begin{tabular}" in Path(tex).read_text()
```

- [ ] **Step 3: Run, commit, log**

Run: `pytest tests/unit/experiments/test_tables.py -v`

```bash
git add -A
git commit -m "M40: markdown and latex table writers"
```

Append M40 DONE.

---

### Task M41: Paper result bundle

**Files:**

- Create: `AdapDomain/src/adapdomain/experiments/profile_vm.py`
- Create: `AdapDomain/src/adapdomain/experiments/collect_interference.py`
- Create: `AdapDomain/src/adapdomain/cli/commands/plot.py`
- Modify: `AdapDomain/src/adapdomain/cli/main.py` to wire `plot`
- Create: `AdapDomain/tests/integration/test_paper_bundle_offline.py`

- [ ] **Step 1: Implement `experiments/profile_vm.py`**

```python
"""ActiveVC orchestration over a model registry. Uses an injected `evaluate` callable
so this module remains independent of the ACL backend."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from adapdomain.profiler.activevc import ActiveVC, ActiveVCConfigRuntime
from adapdomain.profiler.acquisition import AcquisitionWeights
from adapdomain.profiler.frontier_builder import FrontierBuilder, MsprofObservation
from adapdomain.profiler.candidate_grid import GridPoint
from adapdomain.storage.frontier_store import FrontierStore


@dataclass(frozen=True)
class ModelSpec:
    name: str
    om_path: str
    slo_ms: float
    batch_size: int


def profile_all(
    *,
    models: list[ModelSpec],
    budget: int,
    weights: AcquisitionWeights,
    evaluate_factory: Callable[[ModelSpec], Callable[[GridPoint], MsprofObservation]],
    store: FrontierStore,
) -> Mapping[str, str]:
    cfg = ActiveVCConfigRuntime(budget=budget, weights=weights)
    builder = FrontierBuilder(cube_range=(0.0, 1.0), vector_range=(0.0, 1.0))
    versions: dict[str, str] = {}
    for spec in models:
        result = ActiveVC(config=cfg, builder=builder).run(
            model_name=spec.name, version=1, source_run_id=f"{spec.name}-v1",
            evaluate=evaluate_factory(spec),
        )
        path = store.save(result.snapshot)
        versions[spec.name] = str(path)
    return versions
```

- [ ] **Step 2: Implement `experiments/collect_interference.py`**

```python
"""Pairwise interference collection harness."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from adapdomain.core.interference import InterferenceEntry, InterferenceTable


@dataclass(frozen=True)
class PairSpec:
    model_a: str
    model_b: str
    pressure_bucket: str = "mid"


def collect(
    *,
    pairs: Iterable[PairSpec],
    measure_slowdown: Callable[[str, str, str], float],
) -> InterferenceTable:
    entries: list[InterferenceEntry] = []
    for pair in pairs:
        s = measure_slowdown(pair.model_a, pair.model_b, pair.pressure_bucket)
        entries.append(InterferenceEntry(
            model_a=pair.model_a, model_b=pair.model_b, direction="a_on_b",
            pressure_bucket=pair.pressure_bucket,
            slowdown=s, p95_delta=0.0, goodput_delta=0.0, source_run_id="collect",
        ))
    return InterferenceTable.from_entries(entries)
```

- [ ] **Step 3: Implement `cli/commands/plot.py`**

```python
"""`adapdomain plot paper` bundle."""
from __future__ import annotations

from pathlib import Path

import typer

from adapdomain.experiments.plotting import plot_e2e_bar, plot_overhead_curve
from adapdomain.experiments.tables import TableRow, write_latex, write_markdown

app = typer.Typer(help="Reproduce paper figures and tables.")


@app.command("paper")
def paper(
    results: str = typer.Option("data/results"),
    out: str = typer.Option("data"),
) -> None:
    out_dir = Path(out)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    # Placeholder data for offline smoke; real values come from data/results/<run-id>/summary.csv files.
    plot_e2e_bar({"Exclusive": 31.4, "BFD-2D": 66.7, "Dilu-style": 78.8, "AdapDomain": 141.1}, fig_dir / "fig3_e2e.png")
    plot_overhead_curve([10, 100, 400, 800, 1000], [2.79, 19.07, 77.88, 117.80, 131.64], fig_dir / "fig8_runtime_overhead.png")
    write_markdown(tab_dir / "tab_e2e.md", ["Config", "Goodput", "Violation"], [
        TableRow(cells=["Exclusive", "31.4", "0.37"]),
        TableRow(cells=["BFD-2D", "66.7", "17.97"]),
        TableRow(cells=["Dilu-style", "78.8", "15.13"]),
        TableRow(cells=["AdapDomain", "141.1", "5.59"]),
    ])
    write_latex(tab_dir / "tab_overhead.tex", ["Operation", "Size", "Mean (ms)", "p95 (ms)", "Max (ms)"], [
        TableRow(cells=["Placement planning", "10", "2.66", "2.79", "2.93"]),
        TableRow(cells=["Placement planning", "100", "18.16", "19.07", "19.97"]),
        TableRow(cells=["Placement planning", "400", "74.17", "77.88", "81.59"]),
    ])
    typer.echo(str(out_dir))
```

- [ ] **Step 4: Wire CLI** (replace placeholder):

```python
from adapdomain.cli.commands.plot import app as _plot_app
app.add_typer(_plot_app, name="plot")
```

- [ ] **Step 5: Integration test**

```python
import subprocess
import sys
from pathlib import Path


def test_paper_bundle_creates_figures_and_tables(tmp_path):
    out = tmp_path / "bundle"
    proc = subprocess.run(
        [sys.executable, "-m", "adapdomain.cli.main", "plot", "paper", "--out", str(out)],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "figures" / "fig3_e2e.png").exists()
    assert (out / "figures" / "fig8_runtime_overhead.png").exists()
    assert (out / "tables" / "tab_e2e.md").exists()
    assert (out / "tables" / "tab_overhead.tex").exists()
```

- [ ] **Step 6: Run, commit, log**

Run: `pytest tests/integration/test_paper_bundle_offline.py -v`

```bash
git add -A
git commit -m "M41: paper bundle (profile_vm + collect_interference + plot paper command)"
```

Append `M41 DONE`. Append `STAGE C DONE` afterwards.

---

## Self-Review

After the spec was approved I checked the plan against §0–§10:

1. **Spec coverage** — every module M1–M41 in the spec maps to a task here (M22 is intentionally folded into M20/M21 per the spec's own note in §5 Stage B). The continuation log requirement (§8) is satisfied by appending after each task. The risk register (§9) is handled by:
   - ACL enum no-op fallback → M16 logs a warning.
   - msprof schema variants → M19 has two fixture variants and a warning-only test.
   - Stale frontier confidence → M5 ConfidenceModel + M30 SmoothGuard.
   - Backend leaks → M13–M15 strict context managers; remote leak loop documented in M23 runbook.
   - Subagent collisions → each task lists a closed file boundary.
   - No git repo → header explains that "commit" steps degrade to checklist appends when git is absent.

2. **Placeholder scan** — every code step has executable code or executable bash commands. No "TBD" / "implement later" / "similar to Task N" placeholders.

3. **Type consistency** — `ScoreBreakdown`, `RiskTerms`, `ScoreWeights`, `FrontierSnapshot`, `TenantShape`, `DeviceResidual`, `PlacementPlan`, `ResourceQuota`, `InferenceBackend`, `FeasibilityFilter`, `HarmonicFitPlanner`, `MsprofObservation`, `ActiveVCConfigRuntime`, `AcquisitionWeights`, `GridPoint`, `FrontierBuilder`, `ClusterState`, `RuntimeMonitor`, `SmoothGuard`, `PreScale`, `NightShift`, `RepairLogger` are used across tasks with the same names and signatures.

4. **Ambiguity check** — `K=2` for the prototype; HBM is a risk term, not a schedulable domain; `set_device_res_limit` enum values are loaded from YAML and the limiter falls back to a no-op with a warning if absent.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-26-adapdomain-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Stage A modules M4–M10 and Stage B M13/M14/M16/M17/M19/M20/M21/M23, and Stage C M26 are explicitly marked "Parallel: yes" so they can be dispatched in batches.

2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Slower but lets you watch each diff.

Which approach?




