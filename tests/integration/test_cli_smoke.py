import subprocess
import sys


def test_help_lists_subcommands():
    proc = subprocess.run(
        [sys.executable, "-m", "adapdomain.cli.main", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    for cmd in ["version", "verify-remote", "run-model", "profile", "schedule", "replay", "ablate", "plot", "llm-pd"]:
        assert cmd in proc.stdout, f"missing subcommand {cmd}: {proc.stdout}"


def test_version_prints_string():
    proc = subprocess.run(
        [sys.executable, "-m", "adapdomain.cli.main", "version"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() != ""
