"""Configuration: reads ``engraft.toml`` plus ``ENGRAFT_*`` environment overrides.

No module in this package hardcodes a GGUF path, a tokenizer path, or the engine
binary: every one of those lives in ``engraft.toml`` (untracked, see
``engraft.toml.example``) or in an ``ENGRAFT_<SECTION>_<KEY>`` environment
variable, which takes precedence over the file. A missing key raises
``ConfigError`` naming the key and the file, so a caller never fails with a
bare ``KeyError``.

Resolution is always lazy: nothing here is read at import time, so importing
this module (or any module that imports it) never requires ``engraft.toml``
to exist. This matters for test collection, which must succeed even when no
configuration file is present.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("engraft.toml")

# Keys with a built-in default: not required in engraft.toml or the environment.
_DEFAULTS: dict[str, Any] = {
    "engine.fork_commit": "9d9f9f9ad",
    "run.cache_dir": "results/cache",
    "run.ram_cache_gb": 40,
    "run.disk_cache_gb": 40,
}


class ConfigError(RuntimeError):
    """A configuration key is missing from both the environment and the file."""


def _env_name(key: str) -> str:
    return "ENGRAFT_" + key.upper().replace(".", "_")


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _lookup(data: dict[str, Any], key: str) -> Any:
    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return node


class Config:
    """Resolves a dotted key (``"model.table"``) from env, then file, then default."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        self._data: dict[str, Any] | None = None

    def _ensure_loaded(self) -> dict[str, Any]:
        if self._data is None:
            self._data = _load_toml(self.path)
        return self._data

    def get(self, key: str, default: Any = ...) -> Any:
        env_name = _env_name(key)
        if env_name in os.environ:
            return os.environ[env_name]
        try:
            return _lookup(self._ensure_loaded(), key)
        except KeyError:
            pass
        if key in _DEFAULTS:
            return _DEFAULTS[key]
        if default is not ...:
            return default
        raise ConfigError(
            f"{key!r} not found: no {env_name} environment variable and no matching "
            f"key in {self.path} (see engraft.toml.example)."
        )

    def get_path(self, key: str, default: Any = ...) -> Path:
        return Path(self.get(key, default))

    def get_list(self, key: str, default: Any = ...) -> list:
        value = self.get(key, default)
        if not isinstance(value, list):
            raise ConfigError(f"{key!r} in {self.path}: expected a list, found {value!r}")
        return value

    def get_int(self, key: str, default: Any = ...) -> int:
        return int(self.get(key, default))


def load(path: str | Path | None = None) -> Config:
    return Config(path)


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to engraft.toml (default: ./engraft.toml)")
    parser.add_argument("--print", dest="key", required=True, help="dotted key to resolve and print")
    args = parser.parse_args(argv)

    cfg = load(args.config)
    try:
        value = cfg.get(args.key)
    except ConfigError as e:
        print(str(e), flush=True)
        return 1
    print(value)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
