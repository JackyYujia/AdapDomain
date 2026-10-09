import pytest

from adapdomain.storage.jsonl import JsonlReader, JsonlWriter


def test_writer_appends_and_reader_filters(tmp_path):
    path = tmp_path / "events.jsonl"
    w = JsonlWriter(path)
    w.write({"type": "start", "id": 1})
    w.write({"type": "tick", "id": 2})
    w.write({"type": "tick", "id": 3})
    assert list(JsonlReader(path).filter(type="tick")) == [
        {"type": "tick", "id": 2},
        {"type": "tick", "id": 3},
    ]


def test_reader_tolerates_corrupt_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"type":"ok"}\n{not json\n{"type":"ok2"}\n')
    items = list(JsonlReader(path).read_all(tolerant=True))
    assert [r["type"] for r in items] == ["ok", "ok2"]
