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
