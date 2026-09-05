"""Tests for engraft.config: env precedence, missing-key errors, defaults."""
from __future__ import annotations

import subprocess
import sys

import pytest

from engraft.config import Config, ConfigError


def test_missing_key_raises_with_key_and_path(tmp_path):
    cfg = Config(tmp_path / "engraft.toml")
    with pytest.raises(ConfigError) as excinfo:
        cfg.get("model.table")
    msg = str(excinfo.value)
    assert "model.table" in msg
    assert str(tmp_path / "engraft.toml") in msg


def test_reads_value_from_toml_file(tmp_path):
    toml_path = tmp_path / "engraft.toml"
    toml_path.write_text('[model]\ntable = "/x/table.gguf"\n')
    cfg = Config(toml_path)
    assert cfg.get("model.table") == "/x/table.gguf"


def test_env_var_overrides_toml_file(tmp_path, monkeypatch):
    toml_path = tmp_path / "engraft.toml"
    toml_path.write_text('[model]\ntable = "/from/file.gguf"\n')
    monkeypatch.setenv("ENGRAFT_MODEL_TABLE", "/from/env.gguf")
    cfg = Config(toml_path)
    assert cfg.get("model.table") == "/from/env.gguf"


def test_default_used_when_absent_everywhere(tmp_path):
    cfg = Config(tmp_path / "engraft.toml")
    assert cfg.get("engine.fork_commit") == "9d9f9f9ad"
    assert cfg.get("run.ram_cache_gb") == 40


def test_default_overridden_by_file(tmp_path):
    toml_path = tmp_path / "engraft.toml"
    toml_path.write_text('[engine]\nfork_commit = "deadbeef"\n')
    cfg = Config(toml_path)
    assert cfg.get("engine.fork_commit") == "deadbeef"


def test_get_list_type_checks(tmp_path):
    toml_path = tmp_path / "engraft.toml"
    toml_path.write_text('[model]\nshards = ["a.gguf", "b.gguf"]\n')
    cfg = Config(toml_path)
    assert cfg.get_list("model.shards") == ["a.gguf", "b.gguf"]


def test_get_list_rejects_non_list(tmp_path):
    toml_path = tmp_path / "engraft.toml"
    toml_path.write_text('[model]\ntable = "not-a-list"\n')
    cfg = Config(toml_path)
    with pytest.raises(ConfigError):
        cfg.get_list("model.table")


def test_import_does_not_require_engraft_toml(tmp_path, monkeypatch):
    """Collection-time safety (gate 6.1): importing the module never touches disk."""
    monkeypatch.chdir(tmp_path)
    import importlib

    import engraft.config as config_mod

    importlib.reload(config_mod)  # re-import with cwd lacking engraft.toml: must not raise


def test_cli_print_missing_key_reports_error(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "engraft.config", "--config", str(tmp_path / "engraft.toml"), "--print", "model.table"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "model.table" in result.stdout


def test_cli_print_prints_value(tmp_path):
    toml_path = tmp_path / "engraft.toml"
    toml_path.write_text('[engine]\nlens_bin = "/x/llama-ple-lens"\n')
    result = subprocess.run(
        [sys.executable, "-m", "engraft.config", "--config", str(toml_path), "--print", "engine.lens_bin"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "/x/llama-ple-lens"
