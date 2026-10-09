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
