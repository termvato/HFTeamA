"""Command-line entry point for HFTeamA.

Without a subcommand it prints the version. Hardware subcommands (calibrate, teleop,
record, ...) live in hfteama.hardware.cli; run `hfteama --help` for the list.
"""

from __future__ import annotations

import argparse
import sys

from hfteama import __version__


def parser() -> argparse.ArgumentParser:
    from hfteama.hardware.cli import add_subcommands

    p = argparse.ArgumentParser(prog="hfteama", description="HFTeamA command-line interface.")
    p.add_argument("--version", action="version", version=f"HFTeamA v{__version__}")
    p.add_argument("--config", help="Pairs config (default: configs/teleop_pairs.json)")
    p.add_argument(
        "--pair", help="Leader/follower pair, or several for teleop/record: --pair rakesh,hacklab"
    )
    p.add_argument("-n", "--dry-run", action="store_true", help="Print the command only")
    add_subcommands(p.add_subparsers(dest="command"))
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the HFTeamA command-line interface."""
    args = parser().parse_args(argv)
    if args.command is None:
        print(f"HFTeamA v{__version__}")
        return 0
    from hfteama.hardware.cli import dispatch

    return dispatch(args)


if __name__ == "__main__":
    sys.exit(main())
