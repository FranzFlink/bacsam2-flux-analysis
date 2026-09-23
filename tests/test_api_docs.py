"""The API reference (docs/api/) and the figure pages (docs/figures/) of the website are
generated from the code by docs/build_api.py and committed. These tests fail when the
code changed and the pages were not regenerated (fix: ``pixi run api``)."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("build_api", REPO / "docs" / "build_api.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_api"] = module                      # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def test_api_pages_are_current():
    builder = _builder()
    pages = builder.build()
    stale = [p.relative_to(REPO).as_posix() for p, text in pages.items()
             if not p.exists() or p.read_text(encoding="utf-8").replace("\r\n", "\n") != text]
    extra = [p.relative_to(REPO).as_posix() for folder in builder.OUTPUT_DIRS for p in folder.glob("*.md")
             if p not in pages]
    assert not stale and not extra, f"run `pixi run api`: out of date {stale}, not generated {extra}"


def test_api_pages_are_in_the_table_of_contents():
    builder = _builder()
    assert builder.toc_missing(builder.build()) == []
