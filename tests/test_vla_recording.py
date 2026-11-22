"""VLA datasets end to end: record -> check -> discard -> save with LeRobot's own recorder.

Runs `lerobot.scripts.lerobot_record.record()` (the code behind lerobot-record) with a fake
two-arm robot shaped like bi_so_follower: 12 joints, `left_wrist`, `right_wrist` and a top-level
`front` camera. Nothing touches USB or the Hub. Skipped where lerobot is not installed (CI).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytest.importorskip("lerobot", reason="end-to-end recording needs lerobot installed")

import numpy as np  # noqa: E402
from lerobot.robots.config import RobotConfig  # noqa: E402
from lerobot.robots.robot import Robot  # noqa: E402
from lerobot.teleoperators.config import TeleoperatorConfig  # noqa: E402
from lerobot.teleoperators.teleoperator import Teleoperator  # noqa: E402

from hfteama.hardware import vla_data  # noqa: E402
from hfteama.hardware.vla_data import VlaTask, record_command  # noqa: E402
from tests.test_vla_data import FRONT, MOTORS, isolated, task  # noqa: E402, F401

H, W = 96, 128  # small frames keep video encoding fast
JOINTS = [f"{side}_{m}.pos" for side in ("left", "right") for m in MOTORS]
CAMERAS = ["left_wrist", "right_wrist", "front"]
EVENTS: dict[str, bool] = {}
CALLS = {"teleop": 0}
RERECORD_AT: list[int] = []  # teleop call numbers at which "<-" (re-record) is pressed


@RobotConfig.register_subclass("hfteam_fake_bi_follower")
@dataclass(kw_only=True)
class FakeBiFollowerConfig(RobotConfig):
    cameras: dict = field(default_factory=dict)


class FakeBiFollower(Robot):
    config_class = FakeBiFollowerConfig
    name = "bi_so_follower"

    def __init__(self, config):
        super().__init__(config)
        self._connected = False
        self.cameras = dict.fromkeys(CAMERAS)  # lerobot-record counts them
        self._q = np.zeros(len(JOINTS), dtype=np.float32)
        self._rng = np.random.default_rng(0)

    @property
    def observation_features(self) -> dict:
        return {**{j: float for j in JOINTS}, **{c: (H, W, 3) for c in CAMERAS}}

    @property
    def action_features(self) -> dict:
        return {j: float for j in JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict:
        obs = {j: float(q) for j, q in zip(JOINTS, self._q, strict=True)}
        for c in CAMERAS:
            obs[c] = self._rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict) -> dict:
        self._q = np.array([action[j] for j in JOINTS], dtype=np.float32)
        return action

    def disconnect(self) -> None:
        self._connected = False


@TeleoperatorConfig.register_subclass("hfteam_fake_bi_leader")
@dataclass(kw_only=True)
class FakeBiLeaderConfig(TeleoperatorConfig):
    pass


class FakeBiLeader(Teleoperator):
    config_class = FakeBiLeaderConfig
    name = "bi_so_leader"

    def __init__(self, config):
        super().__init__(config)
        self._connected = False

    @property
    def action_features(self) -> dict:
        return {j: float for j in JOINTS}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict:
        CALLS["teleop"] += 1
        if CALLS["teleop"] in RERECORD_AT:  # the operator presses "<-" mid-episode
            EVENTS["rerecord_episode"] = True
            EVENTS["exit_early"] = True
        t = CALLS["teleop"] / 30
        return {j: float(np.sin(t + i)) * 20 for i, j in enumerate(JOINTS)}

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False


def record_session(monkeypatch, t: VlaTask, episodes: int, rerecord_at=()) -> None:
    """One `lerobot-record` session, exactly as `hfteama vla <task> record` would configure it."""
    import lerobot.scripts.lerobot_record as lr
    from lerobot.configs.dataset import DatasetRecordConfig

    EVENTS.clear()
    EVENTS.update(exit_early=False, rerecord_episode=False, stop_recording=False)
    CALLS["teleop"] = 0
    RERECORD_AT[:] = list(rerecord_at)
    monkeypatch.setattr(lr, "init_keyboard_listener", lambda: (None, EVENTS))
    root = vla_data.dataset_root(t)
    cfg = lr.RecordConfig(
        robot=FakeBiFollowerConfig(id="hfteam_bi"),
        teleop=FakeBiLeaderConfig(id="hfteam_bi"),
        dataset=DatasetRecordConfig(
            repo_id=t.repo_id,
            root=root,
            single_task=t.task,
            fps=t.fps,
            episode_time_s=1.0,
            reset_time_s=0,
            num_episodes=episodes,
            push_to_hub=False,
            no_stamp=True,
        ),
        display_data=False,
        play_sounds=False,
        resume=(root / "meta" / "info.json").is_file(),
    )
    lr.record(cfg)


def test_record_check_discard_save(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    t = task("box_middle")
    root = vla_data.dataset_root(t)

    # Night: 2 episodes; the first is re-recorded with "<-" halfway, so 2 are kept.
    record_session(monkeypatch, t, episodes=2, rerecord_at=[15])
    assert CALLS["teleop"] >= 15 + 2 * 29  # the aborted attempt + 2 kept episodes
    s = vla_data.status(t)
    assert s["episodes"] == 2 and s["action_dim"] == 12
    assert s["cameras"] == sorted(f"observation.images.{c}" for c in CAMERAS)
    assert s["tasks"] == [t.task]
    assert all(0.9 <= n <= 1.1 for n in s["lengths_s"]), s["lengths_s"]
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["robot_type"] == "bi_so_follower"

    # Day: the same dataset (fixed name, resumed), 1 more episode.
    record_session(monkeypatch, t, episodes=1)
    assert vla_data.status(t)["episodes"] == 3
    assert vla_data.verify(t) == []

    # Discard the last episode; the version before is kept as <name>_old.
    old = vla_data.discard(t)
    assert old == root.with_name(root.name + "_old") and old.is_dir()
    assert vla_data.status(t)["episodes"] == 2
    assert vla_data.status(t, root=old)["episodes"] == 3
    assert vla_data.verify(t) == []

    # Discard a given episode, then the dataset still records on top (resume after an edit).
    vla_data.discard(t, [0])
    assert vla_data.status(t)["episodes"] == 1
    record_session(monkeypatch, t, episodes=1)
    assert vla_data.status(t)["episodes"] == 2
    assert vla_data.verify(t) == []

    # Save: checks, then pushes (stubbed) privately.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pushed = []
    monkeypatch.setattr(LeRobotDataset, "push_to_hub", lambda self, **kw: pushed.append(kw))
    assert vla_data.save(t) == []
    assert pushed and pushed[0]["private"] is True

    # The check refuses a dataset whose task text differs from the config's.
    wrong = VlaTask(**{**t.__dict__, "task": "Something else"})
    assert any("task text" in p for p in vla_data.verify(wrong))
    pushed.clear()
    assert vla_data.save(wrong) and not pushed

    # Discarding everything moves the dataset aside instead of deleting it.
    aside = vla_data.discard(t, [0, 1])
    assert aside.is_dir() and "_discarded_" in aside.name and not root.exists()


def test_record_command_parses_as_lerobot_record_config(tmp_path: Path) -> None:
    """The real command (bi_so_follower, front camera, fixed name) is valid lerobot-record input."""
    import draccus
    from lerobot.scripts.lerobot_record import RecordConfig

    t = task("box_right")
    cmd = record_command(t, episodes=5, cameras={"front": FRONT})
    cfg = draccus.parse(RecordConfig, args=cmd[1:])
    assert cfg.robot.type == "bi_so_follower" and cfg.teleop.type == "bi_so_leader"
    assert set(cfg.robot.cameras) == {"front"}
    assert set(cfg.robot.left_arm_config.cameras) == {"wrist"}
    assert set(cfg.robot.right_arm_config.cameras) == {"wrist"}
    assert cfg.robot.cameras["front"].width == 640 and cfg.robot.cameras["front"].fps == 30
    assert cfg.dataset.repo_id == t.repo_id and cfg.dataset.no_stamp is True
    assert cfg.dataset.single_task == t.task and cfg.dataset.num_episodes == 5
    assert cfg.dataset.push_to_hub is False and cfg.resume is False
