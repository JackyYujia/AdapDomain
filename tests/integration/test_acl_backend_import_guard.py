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
