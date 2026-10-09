from pathlib import Path

from adapdomain.experiments.plotting import plot_e2e_bar, plot_overhead_curve


def test_plots_write_png(tmp_path):
    out = plot_e2e_bar({"Exclusive": 31.4, "BFD-2D": 66.7, "Dilu-style": 78.8, "AdapDomain": 141.1}, tmp_path / "e2e.png")
    assert Path(out).exists()
    out2 = plot_overhead_curve([10, 100, 400, 800, 1000], [2.79, 19.07, 77.88, 117.80, 131.64], tmp_path / "overhead.png")
    assert Path(out2).exists()
