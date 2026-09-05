"""Fakes used by the dry-run entry points (`--fake`) and by the test suite.

None of these touch a real GGUF; they reproduce the addressing/interface
contracts of the real table, replica, and engine well enough to exercise the
orchestration code without a model.
"""
