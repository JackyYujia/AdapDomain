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
    assert cfg.acl.set_device_res_limit.cube_limit_type == 0
    assert cfg.acl.set_device_res_limit.vector_limit_type == 1
    assert cfg.acl.set_device_res_limit.hbm_limit_type is None


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
