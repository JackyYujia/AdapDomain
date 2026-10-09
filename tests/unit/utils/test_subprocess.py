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
    with pytest.raises(SubprocessError) as excinfo:
        run_checked([sys.executable, "-c", "import time; time.sleep(5)"], timeout=1)
    assert excinfo.value.returncode == -1
