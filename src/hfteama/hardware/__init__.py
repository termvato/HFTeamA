"""Hardware (#19): real SO-101 arms, calibration, teleoperation and data recording.

- pairs.py: leader/follower pairs from configs/teleop_pairs.json, ports found by USB serial.
- cli.py: `hfteama` subcommands wrapping the lerobot-* tools with each pair's settings.
- camera_view.py: live view of a pair's cameras in Rerun.

LeRobot is only needed for the commands that talk to arms; config loading and command
building work without it, so tests and CI run on the base install.
"""
