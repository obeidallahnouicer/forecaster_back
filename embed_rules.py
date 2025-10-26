"""Backward-compatible shim to `tools.rag.embed_rules`.

This file keeps backwards compatibility for code that imports `embed_rules` from
the project root. It re-exports all public symbols from the proper package
implementation under `tools.rag`.
"""
from tools.rag.embed_rules import *  # noqa: F401,F403
