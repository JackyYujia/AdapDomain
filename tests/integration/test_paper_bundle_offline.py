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
