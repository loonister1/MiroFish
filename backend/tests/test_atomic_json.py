import json

import pytest

from app.utils.atomic_json import write_json_atomic


def test_writes_readable_json(tmp_path):
    path = tmp_path / "state.json"

    write_json_atomic(str(path), {"status": "停止", "rounds": 3})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "停止", "rounds": 3}


def test_failed_write_keeps_previous_file_intact(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"status": "running"}', encoding="utf-8")

    with pytest.raises(TypeError):
        write_json_atomic(str(path), {"status": "stopping", "bad": object()})

    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "running"}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]
