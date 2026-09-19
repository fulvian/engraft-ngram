"""Tests for scripts/ask.py against the minimal fake engine of tests/test_engine.py
(no GGUF, no tokenizer: `--tokens` and `--lens-cmd`).

uv run pytest tests/test_ask.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from engraft.lens import write_pleo
from engraft.table import ROW_LEN

HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def ask_mod():
    return _load(HERE.parent / "scripts" / "ask.py", "engraft_ask_script")


@pytest.fixture()
def fake_engine_cmd(tmp_path):
    src = _load(HERE / "test_engine.py", "engraft_test_engine")._FAKE_ENGINE_SRC
    path = tmp_path / "fake_engine_no_table.py"
    path.write_text(src)
    return f"{sys.executable} {path}"


@pytest.fixture()
def overlay(tmp_path):
    rows = np.arange(1000, 1004, dtype=np.int32)
    data = np.ones((4, ROW_LEN), dtype=np.float32)
    path = tmp_path / "overlay.pleo"
    write_pleo(path, rows, data)
    return path


def test_ask_json_reports_both_sides_and_overlay_changes_top(ask_mod, fake_engine_cmd, overlay, capsys):
    rc = ask_mod.main(["--lens-cmd", fake_engine_cmd, "--tokens", "1", "2", "3",
                       "--overlay", str(overlay), "--n", "3", "--top", "4", "--json"])
    assert rc == 0
    res = json.loads(capsys.readouterr().out.strip())
    assert res["tokens"] == [1, 2, 3]
    for side in ("base", "overlay"):
        assert len(res[side]["top"]) == 4
        assert len(res[side]["greedy"]) == 3
        p = [t["p"] for t in res[side]["top"]]
        assert p == sorted(p, reverse=True) and 0.0 < p[0] <= 1.0
    assert res["base"]["overlay_hits"] == 0
    assert res["overlay"]["overlay_hits"] == 4
    # the fake engine's logits are a linear map of the overlay rows: with rows of ones
    # the distribution differs from the base (zero embedding) one
    assert res["base"]["top"] != res["overlay"]["top"]


def test_ask_without_overlay_runs_base_only(ask_mod, fake_engine_cmd, capsys):
    rc = ask_mod.main(["--lens-cmd", fake_engine_cmd, "--tokens", "5", "--n", "0", "--top", "2", "--json"])
    assert rc == 0
    res = json.loads(capsys.readouterr().out.strip())
    assert "overlay" not in res or res["overlay"] is None
    assert "base" in res and res["base"]["greedy"] == []


def test_ask_text_output_mentions_both_sides(ask_mod, fake_engine_cmd, overlay, capsys):
    rc = ask_mod.main(["--lens-cmd", fake_engine_cmd, "--tokens", "1", "--overlay", str(overlay), "--n", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[base]" in out and "[overlay]" in out and "greedy:" in out
