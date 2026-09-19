"""CPU-only smoke test for scripts/check_readme_numbers.py: runs the whole
checklist against this repository's actual README.md and data files, and
asserts every check reads "same" (exit code 0). No GPU, no model, no
network -- just JSON files already on disk.

uv run pytest tests/test_check_readme_numbers.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_readme_numbers.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_readme_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_readme_numbers_match():
    module = _load_module()
    exit_code = module.main()
    assert exit_code == 0, "one or more README numbers failed to recompute -- see stdout above"


def test_every_check_pattern_matches_exactly_once():
    """A regression guard distinct from `main()`'s own count: if a future
    README edit removes or duplicates one of the cited numbers, this fails
    with a clear per-check name instead of a bare nonzero exit code."""
    module = _load_module()
    text = module.README.read_text()
    import re

    failures = []
    for check in module.CHECKS:
        n = len(list(re.finditer(check["readme_pattern"], text)))
        if n != 1:
            failures.append(f"{check['label']}: matched {n} times")
    assert not failures, "\n".join(failures)


def test_every_check_computes_without_raising():
    module = _load_module()
    for check in module.CHECKS:
        try:
            check["compute"]()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{check['label']}: compute() raised {exc!r}")
