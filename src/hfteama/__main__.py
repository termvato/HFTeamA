"""Command-line entry point for HFTeamA."""

from hfteama import __version__


def main() -> None:
    """Run the HFTeamA command-line interface."""
    print(f"HFTeamA v{__version__}")


if __name__ == "__main__":
    main()
