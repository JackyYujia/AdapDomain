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
