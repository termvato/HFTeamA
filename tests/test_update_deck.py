from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parent.parent / "presentation" / "update_deck.py"
_spec = importlib.util.spec_from_file_location("update_deck", PATH)
deck = importlib.util.module_from_spec(_spec)
sys.modules["update_deck"] = deck
_spec.loader.exec_module(deck)

STATUS = """# Status

## Backlog state

| # | Item | State |
|---|---|---|
| #2 | Task model | first version |
| #6 | Greedy | done: greedy + HEFT |
| #7 | CP-SAT | Done |

## Notes
"""


def test_md_table_reads_rows_under_heading():
    rows = deck.md_table(STATUS, "## Backlog state")
    assert [r["#"] for r in rows] == ["#2", "#6", "#7"]
    assert rows[1]["State"] == "done: greedy + HEFT"


def test_md_table_missing_heading_or_text():
    assert deck.md_table(STATUS, "## Nope") == []
    assert deck.md_table(None, "## Backlog state") == []


def test_count_done_counts_done_prefix_case_insensitive():
    assert deck.count_done(deck.md_table(STATUS, "## Backlog state"), "State") == "2/3"
    assert deck.count_done([], "State") is None


def test_limit_label_plain_words():
    assert deck.limit_label("resource:input_bin") == "input bin"
    assert deck.limit_label("order") == "order"


def test_inject_replaces_only_between_markers_and_escapes_script_end():
    html = f"<p>a</p>\n{deck.START}\nold\n{deck.END}\n<p>b</p>"
    out = deck.inject(html, {"x": "</script>"})
    assert out.startswith("<p>a</p>") and out.endswith(f"{deck.END}\n<p>b</p>")
    assert "old" not in out and "<\\/script>" in out
    body = out.split('id="deck-facts">\n', 1)[1].split("\n</script>", 1)[0]
    assert json.loads(body) == {"x": "</script>"}


def test_inject_requires_markers():
    with pytest.raises(ValueError):
        deck.inject("<p>no markers</p>", {})


def test_deck_has_markers():
    html = (PATH.parent / "index.html").read_text(encoding="utf-8")
    assert deck.START in html and deck.END in html
