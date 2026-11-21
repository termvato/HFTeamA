"""The sorter arms behind main's ArmDriver primitives (no hardware: a fake low-level arm)."""

import threading

import pytest

np = pytest.importorskip("numpy")

from hfteama.arms.kinematics import ArmFrame  # noqa: E402
from hfteama.arms.primitives import ArmDriver, MotionError  # noqa: E402
from hfteama.arms.so101_ik import fk, solve_ik  # noqa: E402
from hfteama.sorter.bridge import MainFrame, SorterArmDriver  # noqa: E402
from hfteama.sorter.geometry import load_config  # noqa: E402
from hfteama.sorter.robot import Stopped  # noqa: E402
from hfteama.sorter.run import arm_models  # noqa: E402

TAGS = {1: (0.0, 0.0), 2: (0.734, 0.0), 3: (0.0, 0.335), 0: (0.734, 0.335)}


def test_main_frame_is_the_same_tags_with_another_origin() -> None:
    mf = MainFrame(TAGS)  # main: origin tag 0, x towards tag 3, y towards tag 2
    assert mf.to_sorter([0, 0, 0.1]) == pytest.approx([0.734, 0.335, 0.1])
    assert mf.to_sorter([0.734, 0.0]) == pytest.approx([0.0, 0.335])  # tag 3
    assert mf.to_sorter([0.0, 0.335]) == pytest.approx([0.734, 0.0])  # tag 2
    p = np.array([0.3, 0.1, 0.02])
    assert mf.to_main(mf.to_sorter(p)) == pytest.approx(p)
    assert mf.yaw_to_sorter(0.0) == pytest.approx(np.pi)  # main's +x is the sorter's -x


class FakeLowLevel:
    """Stands in for hfteama.sorter.robot.ArmDriver: records commands, moves instantly."""

    def __init__(self, q0):
        self.frame = ArmFrame(offsets_deg={**ArmFrame().offsets_deg, "wrist_roll": 90.0})
        self._q = np.asarray(q0, float)
        self.gripper_pos = 50.0
        self.stop_event = threading.Event()
        self.calls = []
        self.torque = True

    def read(self):
        return {**self.frame.to_lerobot(self._q, clip=False), "gripper.pos": self.gripper_pos}

    def q(self):
        return self._q

    def move_q(self, q, speed=None):
        if self.stop_event.is_set():
            raise Stopped("stopped")
        self.calls.append(("move_q", np.asarray(q)))
        self._q = np.asarray(q, float)

    def move_line(self, q_from, target_to, yaw, steps=8, speed=None):
        self.calls.append(("move_line", np.asarray(target_to)))
        self._q = solve_ik(target_to, yaw=yaw).q
        return self._q

    def gripper(self, value, wait=None):
        self.calls.append(("gripper", value))
        self.gripper_pos = max(value, 30.0) if value < 10 else value  # a box in the jaws
        return self.gripper_pos

    def go_home(self):
        self.calls.append(("home",))

    def torque_off(self):
        self.torque = False


def make():
    cfg = load_config()
    model = arm_models(cfg, ["rakesh"])["rakesh"]
    low = FakeLowLevel(solve_ik([0.2, 0.0, 0.08]).q)
    d = SorterArmDriver("rakesh_follower", "rakesh", low, model, MainFrame.from_config(cfg), cfg)
    return d, low, cfg


def test_it_is_an_arm_driver_and_picks_with_the_aim_correction() -> None:
    d, low, cfg = make()
    assert isinstance(d, ArmDriver)
    frame = MainFrame.from_config(cfg)
    box_sorter = np.array([0.66, 0.18, 0.015])  # a box on rakesh's side
    box_main = frame.to_main(box_sorter)
    d.open_gripper()
    d.move_to(box_main + [0, 0, 0.08])  # approach: a joint move
    d.move_to(box_main)  # down: same spot, a straight line
    assert [c[0] for c in low.calls] == ["gripper", "move_q", "move_line"]
    assert d.close_gripper().holding
    # the tool went where the sorter's aim correction says, not to the raw camera point
    from hfteama.vision.detect import Piece

    w = cfg["pieces"]["object_width"]
    ax, ay, *_ = arm_models(cfg, ["rakesh"])["rakesh"].grasp_aim(
        Piece(*box_sorter[:2], w, w, 0.0, 0), 0.08
    )
    tip = d._arm.placement.to_table(fk(low.q())[:3, 3])
    assert tip[:2] == pytest.approx([ax, ay], abs=0.01)


def test_carrying_to_the_bowl_uses_the_sorters_drop_and_no_tilted_straight_line() -> None:
    """main's pick_place: the bowl approach (rim + 11 cm) can't keep the gripper down; with a
    box in the jaws a new spot is solved as the sorter's drop, and a tilted pose is left by a
    joint move (move_line keeps the gripper down and would fail on the way)."""
    from hfteama.arms import generic_skills
    from hfteama.arms.poses import Poses
    from hfteama.sorter.plan import drop_solve, drop_target
    from hfteama.taskgraph import Task
    from hfteama.vision.detect import Box

    d, low, cfg = make()
    frame, bc = MainFrame.from_config(cfg), cfg["box"]
    bowl = Box(0.367, 0.252, 0.09)
    x, y, z, _ = drop_target(d._arm, bowl, bc["height"], 0.03, np.radians(60))
    assert 0.0 < np.hypot(x - bowl.x, y - bowl.y) <= bowl.radius - 0.03 + 1e-9
    poses = Poses(
        per_arm={"rakesh_follower": {"bowl": list(frame.to_main([x, y, z]))}},
        objects={"b": list(frame.to_main([0.66, 0.18, 0.015]))},
    )
    task = Task(id="t", skill="pick_place", args={"object": "b", "to": "bowl"})
    generic_skills.pick_place(d, task, poses)
    moves = [c for c in low.calls if c[0] != "gripper"]
    # approach, down (line), lift (line), bowl approach, place, back up: joint moves over the bowl
    assert [c[0] for c in moves] == ["move_q", "move_line", "move_line"] + ["move_q"] * 3
    place = drop_solve(d._arm, x, y, z, True, np.radians(cfg["motion"]["max_tilt_deg"]))
    assert place.ok and moves[4][1] == pytest.approx(place.q)
    assert not d._carrying  # the gripper opened over the bowl


def test_unreachable_and_below_table_are_refused_before_moving() -> None:
    d, low, cfg = make()
    far = MainFrame.from_config(cfg).to_main([0.0, 0.3, 0.05])  # 65 cm from rakesh
    assert not d.reachable(far)
    with pytest.raises(MotionError, match="unreachable"):
        d.move_to(far)
    with pytest.raises(MotionError, match="below table"):
        d.move_to([0.2, 0.2, -0.01])
    assert low.calls == []


def test_stop_latches_until_resume_and_park_refuses_motion() -> None:
    d, low, cfg = make()
    p = MainFrame.from_config(cfg).to_main([0.62, 0.2, 0.08])
    d.stop()
    with pytest.raises(MotionError, match="stopped"):
        d.move_to(p)
    d.resume()
    d.move_to(p)
    d.park()
    assert ("home",) in low.calls and not low.torque
    with pytest.raises(MotionError, match="torque off"):
        d.move_to(p)


def test_export_writes_main_poses_and_a_task_graph(monkeypatch, tmp_path) -> None:
    """Camera (faked) -> poses_real.json in main's frame + one pick_place per box."""
    import json

    import hfteama.sorter.run as run
    from hfteama.__main__ import main
    from hfteama.arms.poses import Poses
    from hfteama.taskgraph import load
    from hfteama.vision.detect import Box, Piece, Scene

    class FakeEye:
        def __init__(self, cfg, image=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def look(self, arm_chains=()):
            pieces = [
                Piece(0.66, 0.18, 0.058, 0.058, 0.0, 33.6, tag_id=3),  # rakesh's half
                Piece(0.12, 0.18, 0.058, 0.058, 0.0, 33.6, tag_id=5),  # hacklab's half
            ]
            return None, None, Scene(box=Box(0.37, 0.22, 0.09), pieces=pieces)

    monkeypatch.setattr(run, "Eye", FakeEye)
    assert main(["sort", "export", "--out", str(tmp_path)]) == 0
    poses = Poses.load(tmp_path / "poses_real.json")
    graph = load(tmp_path / "boxes.json")
    mf = MainFrame.from_config(load_config())
    assert poses.point("box_tag3", "x")[:2] == pytest.approx(mf.to_main([0.66, 0.18])[:2])
    assert poses.point("bowl", "x")[:2] == pytest.approx(mf.to_main([0.37, 0.22])[:2])
    for arm in ("rakesh_follower", "vato_follower"):  # each arm's drop point, as the sorter's
        at = mf.to_sorter(poses.point("bowl", arm))
        assert 0.0 < np.hypot(at[0] - 0.37, at[1] - 0.22) <= 0.06 + 1e-9
        assert at[2] == pytest.approx(load_config()["box"]["height"] + 0.03)
    arms = {t.args["object"]: t.allowed_arms for t in graph.tasks.values()}
    assert arms == {"box_tag3": ("rakesh_follower",), "box_tag5": ("vato_follower",)}
    assert all(t.skill == "pick_place" and t.resources == ("bowl",) for t in graph.tasks.values())
    json.loads((tmp_path / "boxes.json").read_text())


def test_heartbeat_stays_fresh_during_a_long_move() -> None:
    """A move holds the bus for seconds; its loop talks to the servos every tick. The executor
    drops an arm whose heartbeat is 1.5 s old, so alive() must count those ticks."""
    import time

    d, low, cfg = make()

    def slow_move_q(q, speed=None):
        for _ in range(20):  # 2 s, a bus write every 0.1 s
            low.last_io = time.time()
            time.sleep(0.1)
        low._q = np.asarray(q, float)

    low.move_q = slow_move_q
    p = MainFrame.from_config(cfg).to_main([0.62, 0.2, 0.08])
    th = threading.Thread(target=d.move_to, args=(p,))
    th.start()
    ages = []
    while th.is_alive():
        ages.append(time.time() - d.alive())
        time.sleep(0.1)
    th.join()
    assert len(ages) > 10 and max(ages) < 0.5
