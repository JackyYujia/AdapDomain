from pathlib import Path

from adapdomain.experiments.tables import TableRow, write_latex, write_markdown


def test_table_writers(tmp_path):
    md = write_markdown(tmp_path / "t.md", ["A", "B"], [TableRow(cells=["1", "2"])])
    tex = write_latex(tmp_path / "t.tex", ["A", "B"], [TableRow(cells=["1", "2"])])
    assert Path(md).read_text().splitlines()[0] == "| A | B |"
    assert "\\begin{tabular}" in Path(tex).read_text()
