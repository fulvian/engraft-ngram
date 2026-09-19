"""CPU-only smoke test for scripts/check_paper_numbers.py: runs the Table 1
checklist against this repository's actual paper/engraft.tex and data
files, and asserts the checkable entries all read "same" (exit code 0). No
GPU, no model, no network -- just JSON and LaTeX files already on disk.

uv run pytest tests/test_check_paper_numbers.py -q
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_paper_numbers.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_paper_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_checkable_paper_numbers_match():
    module = _load_module()
    exit_code = module.main()
    assert exit_code == 0, "one or more paper numbers failed to recompute -- see stdout above"


def test_every_check_pattern_matches_exactly_once():
    module = _load_module()
    text = module.PAPER.read_text()
    failures = []
    for check in module.CHECKS:
        n = len(list(re.finditer(check["paper_pattern"], text)))
        if n != 1:
            failures.append(f"{check['label']}: matched {n} times")
    assert not failures, "\n".join(failures)


def test_public_checks_compute_without_raising():
    module = _load_module()
    for check in module.CHECKS:
        if not check["public"]:
            continue
        try:
            check["compute"]()
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{check['label']}: compute() raised {exc!r}")


def test_non_public_checks_raise_not_public_data():
    module = _load_module()
    for check in module.CHECKS:
        if check["public"]:
            continue
        with pytest.raises(module.NotPublicData):
            check["compute"]()
