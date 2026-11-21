"""Refresh the pitch deck's facts from the repo (presentation/index.html).

Reads status files and code from git refs with `git show` / `git grep`, so it never checks out
a branch and is safe while other people or agents work in the repo. Runs the planner on the
checked-out code for the scaling numbers. Writes one JSON block between the FACTS markers in the
deck; the deck's script fills its numbers from it. A fact that cannot be read stays null and the
slide keeps its text.

Run: python presentation/update_deck.py [--fetch] [--remote origin] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DECK = HERE / "index.html"
START, END = "<!-- FACTS:START -->", "<!-- FACTS:END -->"
TEAM_BRANCHES = ["main", "orchestrator", "hardware", "action", "action-arm-skills"]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout


def show(ref: str, path: str) -> str | None:
    try:
        return git("show", f"{ref}:{path}")
    except subprocess.CalledProcessError:
        return None


def existing_refs(remote: str) -> list[str]:
    refs = []
    for b in TEAM_BRANCHES:
        try:
            git("rev-parse", "--verify", "--quiet", f"{remote}/{b}")
            refs.append(f"{remote}/{b}")
        except subprocess.CalledProcessError:
            continue
    return refs


def md_table(text: str | None, heading: str) -> list[dict[str, str]]:
    """Rows of the first Markdown table under the heading line that starts with `heading`."""
    if not text:
        return []
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().startswith(heading))
    except StopIteration:
        return []
    table: list[list[str]] = []
    for ln in lines[start + 1 :]:
        if ln.startswith("#"):
            break
        if ln.strip().startswith("|"):
            table.append([c.strip() for c in ln.strip().strip("|").split("|")])
        elif table:
            break
    if len(table) < 2:
        return []
    head, body = table[0], table[2:]
    return [dict(zip(head, row, strict=False)) for row in body]


def count_done(rows: list[dict[str, str]], column: str) -> str | None:
    if not rows:
        return None
    done = sum(r.get(column, "").lower().startswith("done") for r in rows)
    return f"{done}/{len(rows)}"


def search(pattern: str, text: str | None) -> re.Match[str] | None:
    return re.search(pattern, text) if text else None


def test_count(refs: list[str]) -> int | None:
    """Unique test functions (file, name) across the team branches."""
    names: set[tuple[str, str]] = set()
    for ref in refs:
        try:
            out = git("grep", "-E", r"^\s*def test_\w+", ref, "--", "tests/")
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            m = re.match(r"[^:]+:([^:]+):\s*def (test_\w+)", line)
            if m:
                names.add((m.group(1), m.group(2)))
    return len(names) or None


def commit_count(refs: list[str]) -> int | None:
    try:
        return int(git("rev-list", "--count", "--no-merges", *refs).strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def limit_label(binding: str) -> str:
    if binding.startswith("resource:"):
        return binding.removeprefix("resource:").replace("_", " ")
    return binding


def scaling(task: str = "sort12", max_arms: int = 4) -> dict | None:
    """HEFT plan, lower bound and binding term for 1..max_arms arms, on the checked-out code."""
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    try:
        import scaling_benchmark as sb

        from hfteama.taskgraph import load

        rows = sb.plan_rows(task, load(sb.TASKS[task]), range(1, max_arms + 1))
    except Exception as exc:  # optional planner deps missing: the deck keeps its numbers
        print(f"warning: scaling skipped ({exc})", file=sys.stderr)
        return None
    best = max(rows, key=lambda r: (round(r.planned_speedup, 2), -r.n))
    return {
        "task": task,
        "rows": [
            {
                "n": r.n,
                "planned": round(r.planned, 1),
                "bound": round(r.bound, 1),
                "ideal": round(r.ideal, 1),
                "limit": limit_label(r.binding),
            }
            for r in rows
        ],
        "best_speedup": f"{best.planned_speedup:.1f}\u00d7",
        "best_arms": best.n,
    }


def collect(remote: str) -> dict:
    refs = existing_refs(remote)
    main = f"{remote}/main"
    ops = show(main, "OPERATIONS_STATUS.md")
    action = show(f"{remote}/action-arm-skills", "ACTION_STATUS.md")
    vla = show(f"{remote}/action", "vla/README.md")
    pairs = show(main, "configs/teleop_pairs.json")

    sim = search(r"(\d+/\d+) steps ok on (\d+) arms", action)
    dry = search(r"1 arm (\d+) s, 2 arms (\d+) s", ops)
    vlm = search(r"(\d+/\d+) objects, error mean ([\d.]+) cm", ops)
    episodes = [
        int(re.sub(r"\D", "", r.get("Episodes", "")) or 0)
        for r in md_table(vla, "## Datasets recorded")
    ]
    try:
        followers = len(
            git("ls-tree", "--name-only", f"{main}:configs/calibration/robots/so_follower").split()
        )
        n_pairs = len(json.loads(pairs)["pairs"]) if pairs else None
    except (subprocess.CalledProcessError, ValueError, KeyError):
        followers, n_pairs = None, None

    sha = git("rev-parse", "--short", main).strip() if main in refs else None
    return {
        "generated": datetime.now().strftime("%a %d %b %H:%M"),
        "main_sha": sha,
        "show": {
            "tests": test_count(refs),
            "commits": commit_count(refs),
            "ops_backlog": count_done(md_table(ops, "## Backlog state"), "State"),
            "action_work": count_done(md_table(action, "## Workstreams"), "Status"),
            "followers": followers,
            "pairs": n_pairs,
            "episodes": sum(episodes) or None,
            "sim_steps": sim.group(1) if sim else None,
            "sim_arms": sim.group(2) if sim else None,
            "dry_1": f"{dry.group(1)} s" if dry else None,
            "dry_2": f"{dry.group(2)} s" if dry else None,
            "vlm_found": vlm.group(1) if vlm else None,
            "vlm_error": f"{vlm.group(2)} cm" if vlm else None,
        },
        "scaling": scaling(),
    }


def inject(html: str, facts: dict) -> str:
    """Replace the JSON between the FACTS markers."""
    i, j = html.find(START), html.find(END)
    if i < 0 or j < i:
        raise ValueError(f"deck is missing the {START} ... {END} markers")
    body = json.dumps(facts, indent=1, ensure_ascii=False).replace("</", "<\\/")
    block = f'{START}\n<script type="application/json" id="deck-facts">\n{body}\n</script>\n'
    return html[:i] + block + html[j:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--fetch", action="store_true", help="git fetch the remote first (refs only)")
    ap.add_argument("--dry-run", action="store_true", help="print the facts, do not write")
    args = ap.parse_args()
    if args.fetch:
        git("fetch", "--quiet", args.remote)
    facts = collect(args.remote)
    print(json.dumps(facts, indent=1, ensure_ascii=False))
    missing = [k for k, v in facts["show"].items() if v is None]
    if missing:
        print(f"warning: not found, slides keep their text: {', '.join(missing)}", file=sys.stderr)
    if not args.dry_run:
        DECK.write_text(inject(DECK.read_text(encoding="utf-8"), facts), encoding="utf-8")
        print(f"updated {DECK.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
