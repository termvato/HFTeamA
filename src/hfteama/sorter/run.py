"""`hfteama sort ...`: preview, touch (fit an arm's placement), goto, run.

The arms run one at a time (alternating) so they never meet over the box. Between picks every
arm is back at home (its starting, folded pose) and the camera looks again.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from hfteama.arms.so101_ik import chain_points, fk, solve_ik
from hfteama.sorter.geometry import (
    PROJECT_ROOT,
    Placement,
    fit_placement,
    load_config,
    resolve,
    save_placement,
)
from hfteama.sorter.plan import ArmModel, Plan, make_plan

OUT_DIR = PROJECT_ROOT / "outputs" / "sorter"
# overlay colour per arm, from the arm's "colour" in the config (BGR)
PALETTE = {"black": (255, 160, 0), "green": (0, 200, 0)}
COLOURS: dict[str, tuple[int, int, int]] = {}


# --- vision -----------------------------------------------------------------------------------


class Eye:
    """Camera + intrinsics + the latest camera pose from the tags."""

    def __init__(self, cfg: dict, image: str | None = None):
        from hfteama.vision.table import Intrinsics

        self.cfg = cfg
        self.intr = Intrinsics.load(resolve(cfg["camera"]["intrinsics"]))
        self.tags = {int(k): tuple(v) for k, v in cfg["table"]["tags"].items()}
        self.image = image
        self.cam = None
        self.pose = None

    def __enter__(self):
        if self.image is None:
            from hfteama.vision.camera import OverheadCamera

            self.cam = OverheadCamera(self.cfg["camera"]).open()
        return self

    def __exit__(self, *exc):
        if self.cam:
            self.cam.close()

    def look(self, arm_chains=()):
        """(frame, pose, scene). Keeps the last good pose if tags are hidden this time.
        arm_chains: the arms' polylines in the table frame, masked out of the pieces."""
        import cv2

        from hfteama.vision.detect import detect_scene
        from hfteama.vision.table import centre_pixel, detect_tags, fit_pose

        frame = cv2.imread(self.image) if self.image else self.cam.frame()
        if frame is None:
            raise SystemExit(f"Cannot read {self.image}")
        found = detect_tags(frame, self.cfg["table"].get("dictionary", "DICT_4X4_50"))
        pose = fit_pose({i: centre_pixel(c) for i, c in found.items()}, self.tags, self.intr)
        limit = self.cfg["table"].get("max_pose_rms_px", 3.0)
        if pose is not None and pose.rms_px <= limit:
            self.pose = pose
        elif self.pose is None:
            seen = sorted(i for i in found if i in self.tags)
            raise SystemExit(f"Need 3+ reference tags (seen {seen}) to place the camera.")
        self.found = found
        scene = detect_scene(frame, self.pose, self.intr, self.cfg, arm_chains)
        return frame, self.pose, scene


def arm_models(cfg: dict, sides) -> dict[str, ArmModel]:
    for side in sides:
        COLOURS[side] = PALETTE.get(cfg["arms"][side].get("colour"), (200, 200, 200))
    import json

    from hfteama.arms.kinematics import ArmFrame
    from hfteama.arms.so101_ik import calibrated_limits, roll_window
    from hfteama.hardware.pairs import FOLLOWER_CALIBRATION_DIR, read_config

    pairs = read_config()["pairs"]
    out = {}
    for side in sides:
        a = cfg["arms"][side]
        limits = None
        cal = FOLLOWER_CALIBRATION_DIR / f"{pairs[a['pair']]['follower']['id']}.json"
        if cal.exists():
            frame = ArmFrame(offsets_deg={**ArmFrame().offsets_deg, **a.get("offsets_deg", {})})
            limits = roll_window(
                calibrated_limits(json.loads(cal.read_text()), frame),
                frame,
                a.get("wrist_roll_center_deg", 0.0),
                cfg["motion"].get("wrist_roll_half_range_deg", 90.0),
            )
        out[side] = ArmModel(
            side,
            Placement.from_config(a),
            limits,
            tuple(a.get("aim_offset", (0.0, 0.0))),
            tuple(a.get("jaw_offset", (0.0, 0.0))),
            tuple(tuple(r) for r in sorted(a.get("aim_table", []))),
            tuple(a.get("bearing_range_deg", (-180.0, 180.0))),
        )
    return out


def draw(frame, pose, intr, scene, plan: Plan | None, cfg: dict, arms: dict):
    """The camera frame with tags, box, pieces (coloured by arm), jaw directions, reach."""
    import cv2

    from hfteama.vision.table import project

    img = frame.copy()
    k = img.shape[1] / 1280

    def poly(points3, colour, closed=True, th=2):
        px = np.round(project(points3, pose, intr)).astype(np.int32)
        cv2.polylines(img, [px], closed, colour, max(1, int(th * k)))

    for tx, ty in (v for v in cfg["table"]["tags"].values()):
        poly([[tx - 0.01, ty, 0], [tx + 0.01, ty, 0]], (0, 255, 255), False)
        poly([[tx, ty - 0.01, 0], [tx, ty + 0.01, 0]], (0, 255, 255), False)
    a = np.linspace(0, 2 * np.pi, 60)
    for side, arm in arms.items():
        p = arm.placement
        circ = [[p.x + 0.30 * np.cos(t), p.y + 0.30 * np.sin(t), 0] for t in a]
        poly(circ, COLOURS.get(side, (200, 200, 200)), th=1)
        poly([[p.x, p.y, 0], [*p.to_table([0.08, 0.0])[:2], 0]], COLOURS.get(side), False, 3)
    if scene.box is not None:
        b, h = scene.box, cfg["box"]["height"]
        poly(
            [[b.x + b.radius * np.cos(t), b.y + b.radius * np.sin(t), h] for t in a], (255, 0, 255)
        )
    side_of = {}
    if plan:
        side_of = {id(pk.piece): pk.side for pk in plan.picks}

    def label(xyz, text, colour=(255, 255, 255)):
        u, v = np.round(project([xyz], pose, intr)[0]).astype(int)
        for c, t in (((0, 0, 0), 4), (colour, 2)):
            cv2.putText(
                img, text, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * k, c, int(t * k)
            )

    # the table frame: each tag's id and coordinates, the axes from the origin
    tags = {int(i): v for i, v in cfg["table"]["tags"].items()}
    for i, (tx, ty) in tags.items():
        label([tx, ty, 0], f"tag {i} ({tx:.3f}, {ty:.3f})", (0, 255, 255))
    o = next((v for v in tags.values() if v[0] == 0 and v[1] == 0), None)
    if o is not None:
        for (dx, dy), name in (((0.1, 0), "x"), ((0, 0.1), "y")):
            a3 = np.round(project([[0, 0, 0], [dx, dy, 0]], pose, intr)).astype(int)
            cv2.arrowedLine(img, tuple(a3[0]), tuple(a3[1]), (0, 255, 255), int(3 * k))
            cv2.putText(
                img,
                name,
                tuple(a3[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * k,
                (0, 255, 255),
                int(2 * k),
            )
    for side, arm in arms.items():
        p = arm.placement
        u, v = np.round(project([[p.x, p.y, 0]], pose, intr)[0]).astype(int)
        cv2.drawMarker(img, (u, v), COLOURS.get(side), cv2.MARKER_CROSS, int(30 * k), int(3 * k))
        label([p.x, p.y, 0], f"{side} ({p.x:.3f}, {p.y:.3f})", COLOURS.get(side))
        # how far the arm centre is from the nearest tag, along x and y (what is measured)
        tx, ty = min(tags.values(), key=lambda t: np.hypot(t[0] - p.x, t[1] - p.y))
        corner = [p.x, ty, 0]
        poly([[tx, ty, 0], corner], COLOURS.get(side), False, 2)
        poly([corner, [p.x, p.y, 0]], COLOURS.get(side), False, 2)
        label([(tx + p.x) / 2, ty, 0], f"{abs(p.x - tx) * 100:.1f} cm", COLOURS.get(side))
        label([p.x, (ty + p.y) / 2 - 0.02, 0], f"{abs(p.y - ty) * 100:.1f} cm", COLOURS.get(side))
    if scene.box is not None:
        b = scene.box
        label([b.x, b.y, cfg["box"]["height"]], f"box ({b.x:.3f}, {b.y:.3f})", (255, 0, 255))
    for i, pc in enumerate(scene.pieces, 1):
        label([pc.x, pc.y, 0], f"   ({pc.x:.3f}, {pc.y:.3f}) {piece_angle(pc):.0f}deg")
        # the piece number, big, just beside the piece: what `sort jog` takes
        u, v = np.round(project([[pc.x, pc.y, 0]], pose, intr)[0]).astype(int)
        org = (u - int(40 * k), v - int(20 * k))
        for c, t in (((0, 0, 0), 6), ((255, 255, 0), 3)):
            cv2.putText(img, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 1.3 * k, c, int(t * k))
    for pc in scene.pieces:
        # coloured by the arm that takes it; red = nobody can; cyan when there is no plan
        unplanned = (0, 0, 255) if plan else (255, 255, 0)
        colour = COLOURS.get(side_of.get(id(pc)), unplanned)
        poly(np.c_[pc.outline, np.zeros(len(pc.outline))], colour)
        d = 0.5 * pc.width + 0.01
        dx, dy = d * np.cos(pc.yaw), d * np.sin(pc.yaw)
        z = cfg["pieces"].get("grasp_z", 0.015)
        poly([[pc.x - dx, pc.y - dy, z], [pc.x + dx, pc.y + dy, z]], (255, 255, 255), False)
    lines = [f"pose rms {pose.rms_px:.2f}px tags {list(pose.tags_used)}"]
    if plan:
        lines += [
            "picks: " + ", ".join(f"{s} {len(plan.for_arm(s))}" for s in arms),
            f"unreachable {len(plan.unreachable)}  " + " ".join(plan.notes),
        ]
    for i, t in enumerate(lines):
        y = int((30 + 30 * i) * k)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (0, 0, 0), int(4 * k))
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * k, (255, 255, 255), int(2 * k))
    return img


def save_image(img, name: str) -> Path:
    import cv2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{name}.jpg"
    cv2.imwrite(str(path), img)
    return path


def piece_angle(p) -> float:
    """Orientation of a piece's long side, deg in [0, 180), from table +x towards +y."""
    return float(np.degrees(p.yaw + np.pi / 2) % 180.0)


def report(scene, plan: Plan, arms) -> None:
    """Arm centres, box and every piece (pose in the table frame, metres / degrees)."""
    print(
        "Table frame (seen from behind the arms): origin tag 1 (back-left), x -> right (tag 2),"
        " y -> forward to the camera (tags 3, 0), metres."
    )
    print("\nArm centres (base_link origin):")
    for side, arm in arms.items():
        p = arm.placement
        drop = plan.drops.get(side)
        reach = (
            f"reaches box, drop tilt {np.degrees(drop.tilt):.0f} deg"
            if drop
            else ("CAN'T reach the box")
        )
        print(
            f"  {side:5}  x {p.x:.3f}  y {p.y:.3f}  heading {np.degrees(p.yaw):.1f} deg  ({reach})"
        )
    b = scene.box
    print("\nBox:", "not found" if b is None else f"centre x {b.x:.3f}  y {b.y:.3f}")
    side_of = {id(pk.piece): pk for pk in plan.picks}
    print(f"\nPieces: {len(scene.pieces)}")
    print("   #      x      y   angle   length x width   area  arm    dist")
    for i, p in enumerate(scene.pieces, 1):
        pk = side_of.get(id(p))
        who = pk.side if pk else "-"
        dist = f"{np.hypot(*pk.target_arm[:2]) * 100:.0f} cm" if pk else "unreachable"
        print(
            f"  {i:2}  {p.x:.3f}  {p.y:.3f}  {piece_angle(p):5.0f}   "
            f"{p.length * 100:5.1f} x {p.width * 100:4.1f} cm  {p.area_cm2:5.1f}  {who:5}  {dist}"
        )
    for n in plan.notes:
        print("note:", n)


# --- commands ---------------------------------------------------------------------------------


def cmd_preview(args) -> int:
    """Camera only: detect, plan, print, and save the annotated frame; with --live, stream it
    to Rerun (the venv's OpenCV is headless: no windows)."""
    cfg = load_config(args.sorter_config)
    sides = _sides(args, cfg)
    arms = arm_models(cfg, sides)
    with Eye(cfg, args.image) as eye:
        if not args.live:
            frame, pose, scene = eye.look()
            plan = make_plan(scene, arms, cfg)
            report(scene, plan, arms)
            print(
                "saved", save_image(draw(frame, pose, eye.intr, scene, plan, cfg, arms), "preview")
            )
            return 0
        return _preview_live(eye, arms, cfg)


def _preview_live(eye, arms, cfg) -> int:
    import contextlib
    import io

    import rerun as rr

    rr.init("hfteama_sort_preview", spawn=True)
    print("Streaming to Rerun (camera, top-down view, piece mask, poses). Ctrl+C to stop.")
    text = ""
    try:
        while True:
            frame, pose, scene = eye.look()
            plan = make_plan(scene, arms, cfg)
            img = draw(frame, pose, eye.intr, scene, plan, cfg, arms)
            rr.log("camera", rr.Image(img[..., ::-1]).compress(jpeg_quality=80))
            if scene.top_down is not None:
                rr.log("top_down", rr.Image(scene.top_down[..., ::-1]).compress(jpeg_quality=80))
                rr.log("piece_mask", rr.Image(scene.mask))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                report(scene, plan, arms)
            text = buf.getvalue()
            rr.log("poses", rr.TextDocument(f"```\n{text}```", media_type=rr.MediaType.MARKDOWN))
            if eye.image:  # a saved frame does not change
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    print(text)
    return 0


def _sides(args, cfg) -> list[str]:
    sides = [s.strip() for s in (args.arms or ",".join(cfg["arms"])).split(",") if s.strip()]
    bad = [s for s in sides if s not in cfg["arms"]]
    if bad:
        raise SystemExit(f"Unknown arm {bad}; config has {list(cfg['arms'])}")
    return sides


def cmd_touch(args) -> int:
    """Fit an arm's placement: touch tag centres with the tool tip, torque off."""
    from hfteama.sorter.robot import ArmDriver
    from hfteama.vision.table import centre_pixel, pixel_to_table

    cfg = load_config(args.sorter_config)
    side = args.arm
    print(
        f"Touch calibration of the {side} arm.\n"
        "Torque will be switched OFF: hold the arm, it will go limp.\n"
        "For each point: type the id of a tag the arm can reach (a reference tag 0-3, or any\n"
        "other DICT_4X4_50 tag lying flat on the table), with the arm away from it, then\n"
        "touch the tag centre with the tip of the closed gripper (between the finger tips)\n"
        "and press Enter. 3+ points spread out; empty id to finish."
    )
    input("Press Enter when ready (Ctrl-C to abort)... ")
    drv = ArmDriver(side, cfg["arms"][side], cfg["motion"], args.config).connect()
    arm_pts, table_pts = [], []
    try:
        drv.torque_off()
        with Eye(cfg) as eye:
            while True:
                tid = input("tag id (empty to finish): ").strip()
                if not tid:
                    break
                frame, pose, _ = eye.look()
                tid = int(tid)
                if tid in eye.tags:
                    xy = np.array(eye.tags[tid])
                elif tid in eye.found:
                    xy = pixel_to_table([centre_pixel(eye.found[tid])], pose, eye.intr, 0.0)[0, :2]
                else:
                    print(f"  tag {tid} not visible; move the arm out of the way")
                    continue
                input(f"  tag {tid} at ({xy[0]:.3f}, {xy[1]:.3f}). Touch its centre, Enter... ")
                tip = drv.tcp()
                print(f"  tip in arm frame ({tip[0]:.3f}, {tip[1]:.3f}, z {tip[2]:.3f})")
                arm_pts.append(tip[:2])
                table_pts.append(xy)
        if not arm_pts:
            return 1
        guess = Placement.from_config(cfg["arms"][side])
        fit, rms = fit_placement(arm_pts, table_pts, yaw_guess=guess.yaw)
        print(
            f"fit: base ({fit.x:.4f}, {fit.y:.4f}) m, yaw {np.degrees(fit.yaw):.1f} deg,"
            f" rms {rms * 1000:.1f} mm over {len(arm_pts)} points"
            f" (was ({guess.x:.3f}, {guess.y:.3f}), {np.degrees(guess.yaw):.1f} deg)"
        )
        if input("Save it? [y/N] ").strip().lower() == "y":
            print("saved", save_placement(side, fit, rms, len(arm_pts)))
        input("Put the arm back at rest (folded) and press Enter to release it... ")
    finally:
        drv.disconnect(go_home=False)
    return 0


def cmd_goto(args) -> int:
    """Hover the tool tip over a table point (a tag centre, x y, or the box seen by the
    camera), wait for Enter, then go home."""
    from hfteama.sorter.plan import plan_drop
    from hfteama.sorter.robot import ArmDriver

    cfg = load_config(args.sorter_config)
    side = args.arm
    if side not in cfg["arms"]:
        raise SystemExit(f"Unknown arm {side!r}; config has {list(cfg['arms'])}")
    arm = arm_models(cfg, [side])[side]
    if args.box:
        bc, mc = cfg["box"], cfg["motion"]
        with Eye(cfg, args.image) as eye:
            _, _, scene = eye.look()
        if scene.box is None:
            raise SystemExit("Box not found by the camera.")
        b = scene.box
        above = bc.get("drop_above", 0.04) if args.z is None else args.z
        print(
            f"box centre ({b.x:.3f}, {b.y:.3f}); rim at {bc['height']:.3f} m,"
            f" going {above:.3f} above it"
        )
        sol = plan_drop(arm, b, bc["height"], above, np.radians(mc.get("drop_max_tilt_deg", 60.0)))
        if sol is None:
            d = np.hypot(b.x - arm.placement.x, b.y - arm.placement.y)
            print(
                f"{side} arm can't reach over the box ({d * 100:.0f} cm from its base)."
                " Move the box closer."
            )
            return 1
        xy = arm.placement.to_table(fk(sol.q)[:3, 3])
        z = xy[2]
    else:
        z = 0.03 if args.z is None else args.z
        if args.tag is not None:
            xy = cfg["table"]["tags"].get(str(args.tag))
            if xy is None:
                raise SystemExit(f"Tag {args.tag} is not a reference tag")
        elif args.xy:
            xy = args.xy
        else:
            raise SystemExit("Give --box, --tag ID or --xy X Y")
        sol = solve_ik(arm.placement.to_arm([xy[0], xy[1], z]), limits=arm.limits)
    target = arm.placement.to_arm([xy[0], xy[1], z])
    print(
        f"target table ({xy[0]:.3f}, {xy[1]:.3f}, {z:.3f}) = arm "
        f"({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}); IK ok={sol.ok} "
        f"err {sol.position_error * 1000:.1f} mm tilt {np.degrees(sol.tilt):.0f} deg"
    )
    print(
        "joints (LeRobot deg):", {k: round(v, 1) for k, v in _lerobot_deg(cfg, side, sol.q).items()}
    )
    if not sol.ok and not args.box:
        return 1
    if args.dry_run:
        return 0
    drv = ArmDriver(side, cfg["arms"][side], cfg["motion"], args.config).connect()
    try:
        drv.move_q(sol.q)
        tip = arm.placement.to_table(drv.tcp())
        print(f"tip now at table ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) by the joint readings")
        input("Check where the tip is, then Enter to go home... ")
    finally:
        drv.disconnect()
    return 0


def _lerobot_deg(cfg, side, q) -> dict:
    from hfteama.arms.kinematics import ArmFrame

    a = cfg["arms"][side]
    frame = ArmFrame(offsets_deg={**ArmFrame().offsets_deg, **a.get("offsets_deg", {})})
    return frame.to_lerobot(q, clip=False)


JOG_HELP = """Commands (directions as seen from BEHIND the arm, looking where it looks):
  N       go over piece N (numbers as in the Rerun view)
  f / b   forward (away from the arm) / back      l / r   left / right
  u / d   up / down                                add a number for several steps: f3
  s N     step size in mm (now {step:.0f})          y N     turn the jaws by N deg
  o / c   open / close the gripper                 g       grasp here: down to grasp_z, close, up
  g N     grasp going down exactly N cm             z N     the tip is really N cm above the table
  box     over the box and open (drop)             scan    home, look again, renumber
  log     the last grasp DID hold the piece (if `g` said empty): log it for fit-grasps
  ok      the tip is right over the piece's centre: record camera-vs-arm offset
  h       help                                     q       finish (fit, save) and go home"""


class Jog:
    """Interactive session: the arm hovers over pieces the camera found; the operator nudges
    it, grasps, drops in the box. A background thread streams the camera to Rerun with the
    pieces numbered, the aim point (red) and the tool tip by the encoders (yellow)."""

    def __init__(self, cfg, side, arm, drv, eye, z, step):
        self.cfg, self.side, self.arm, self.drv, self.eye = cfg, side, arm, drv, eye
        self.pc, self.mc = cfg["pieces"], cfg["motion"]
        self.z, self.step = z, step
        self.scene = None
        self.piece = None  # index of the current piece
        self.target = None  # table xyz the arm aims at
        self.yaw = 0.0
        self.q = drv.q()
        self.pairs_arm, self.pairs_table = [], []
        self.z_changed = False
        self.last_grasp = None  # (piece, aim, q) of the last `g` not logged yet, for `log`
        self.dz = 0.0  # height correction of the current piece (ArmModel.aim_table)
        self.planned = None  # table xy the current piece was aimed at, before nudges

    # --- vision -------------------------------------------------------------------------------

    def scan(self) -> None:
        """Back home, look, renumber the pieces."""
        if self.target is not None:
            self.drv.go_home()
            self.q = self.drv.q()
            self.target = None
        chain = np.array([self.arm.placement.to_table(p) for p in chain_points(self.q)])
        with self.lock:
            _, _, self.scene = self.eye.look([chain])
        self.piece = None
        b = self.scene.box
        print("box:", "not found" if b is None else f"({b.x:.3f}, {b.y:.3f})")
        for i, p in enumerate(self.scene.pieces, 1):
            d = np.hypot(*self.arm.placement.to_arm([p.x, p.y])[:2])
            print(
                f"  {i}: ({p.x:.3f}, {p.y:.3f}) {p.length * 100:.1f} x {p.width * 100:.1f} cm,"
                f" {d * 100:.0f} cm from the base"
            )
        if not self.scene.pieces:
            print("  no pieces seen")

    def stream(self, stop) -> None:
        """Camera -> Rerun, ~5 fps, until stop is set."""
        import cv2
        import rerun as rr

        while not stop.is_set():
            try:
                with self.lock:
                    frame = self.eye.cam.frame()
                if self.eye.pose is None or self.scene is None:
                    continue
                img = draw(
                    frame,
                    self.eye.pose,
                    self.eye.intr,
                    self.scene,
                    None,
                    self.cfg,
                    {self.side: self.arm},
                )
                self._marks(cv2, img)
                rr.log("camera", rr.Image(img[..., ::-1]).compress(jpeg_quality=75))
            except Exception as e:  # never kill the session for a dropped frame
                print(f"  [viewer] {e}")
            stop.wait(0.2)

    def _marks(self, cv2, img) -> None:
        from hfteama.vision.table import project

        pose, intr = self.eye.pose, self.eye.intr
        k = img.shape[1] / 1280
        if self.target is not None:
            g = np.round(project([[*self.target[:2], 0.0]], pose, intr)[0]).astype(int)
            cv2.circle(img, tuple(g), int(10 * k), (0, 0, 255), int(2 * k))
        tip = self.arm.placement.to_table(fk(self.q)[:3, 3])
        t = np.round(project([[tip[0], tip[1], 0.0]], pose, intr)[0]).astype(int)
        cv2.drawMarker(img, tuple(t), (0, 255, 255), cv2.MARKER_CROSS, int(24 * k), int(2 * k))
        txt = f"tip ({tip[0]:.3f}, {tip[1]:.3f}, z {tip[2]:.3f})"
        if self.piece is not None:
            txt = f"piece {self.piece + 1}  " + txt
        cv2.putText(
            img,
            txt,
            (10, img.shape[0] - int(20 * k)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * k,
            (0, 255, 255),
            int(2 * k),
        )

    # --- motion -------------------------------------------------------------------------------

    def move(self, target, yaw) -> bool:
        sol = solve_ik(
            self.arm.placement.to_arm(target), yaw=yaw, seed=self.q, limits=self.drv.limits
        )
        if not sol.ok:
            d = np.hypot(*self.arm.placement.to_arm(target)[:2]) * 100
            print(
                f"  can't reach ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}):"
                f" {d:.0f} cm from the base (the gripper reaches about 30 cm pointing down)."
                " Move the piece closer to the arm."
            )
            return False
        self.drv.move_q(sol.q)
        self.q, self.target, self.yaw = sol.q, np.asarray(target, float), yaw
        return True

    def goto_piece(self, i: int) -> None:
        if not self.scene or not 0 <= i < len(self.scene.pieces):
            print("  no such piece (scan to renumber)")
            return
        p = self.scene.pieces[i]
        if not self.arm.takes(p.x, p.y):
            lo, hi = self.arm.bearing_range
            side = "left" if self.arm.bearing(p.x, p.y) > hi else "right"
            print(
                f"  piece {i + 1} is {abs(self.arm.bearing(p.x, p.y)):.0f} deg to the arm's {side}:"
                f" {self.side} only takes pieces between {lo:+.0f} and {hi:+.0f} deg"
                " (config bearing_range_deg)"
            )
            return
        ax, ay, yaw, _, dz = self.arm.grasp_aim(p, self.z)
        if self.move([ax, ay, self.z + dz], yaw):
            self.dz = dz
            self.planned = np.array([ax, ay])  # the aim before any nudge, for `ok`
            self.piece = i
            print(
                f"  over piece {i + 1}: camera says ({p.x:.3f}, {p.y:.3f}),"
                f" {self.z * 100:.0f} cm up"
            )

    def nudge(self, key: str, n: int) -> None:
        if self.target is None:
            print("  pick a piece first (its number)")
            return
        fwd = (
            self.arm.placement.to_table([1.0, 0.0])[:2]
            - self.arm.placement.to_table([0.0, 0.0])[:2]
        )
        left = np.array([-fwd[1], fwd[0]])
        d = {"f": fwd, "b": -fwd, "l": left, "r": -left}.get(key, np.zeros(2))
        new = self.target.copy()
        new[:2] += d * self.step * n
        new[2] += {"u": 1, "d": -1}.get(key, 0) * self.step * n
        if new[2] < self.pc.get("grasp_z", 0.015) - 1e-9:
            print(f"  not lower than grasp_z ({self.pc.get('grasp_z', 0.015) * 100:.1f} cm)")
            return
        if self.move(new, self.yaw):
            self._where()

    def _where(self) -> None:
        t = self.target
        msg = f"  aim ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})"
        if self.piece is not None:
            p = self.scene.pieces[self.piece]
            dx, dy = (t[0] - p.x) * 100, (t[1] - p.y) * 100
            msg += f", {dx:+.1f} / {dy:+.1f} cm from the camera's point (x / y)"
        print(msg)

    def grasp(self, depth: float | None = None) -> None:
        """Open, straight down (to grasp_z, or `depth` m below the aim), close, back up."""
        if self.target is None:
            print("  pick a piece first")
            return
        dz = self.dz if self.piece is not None else 0.0
        z = self.pc.get("grasp_z", 0.005) + dz if depth is None else self.target[2] - depth
        if z < dz - 0.002:
            print(f"  that would go {-z * 100:.1f} cm into the table")
            return
        self.drv.gripper(self.mc.get("gripper_open", 60.0), wait=0.3)
        up = self.arm.placement.to_arm(self.target)
        low = self.arm.placement.to_arm([*self.target[:2], z])
        try:
            self.q = self.drv.move_line(self.q, low, self.yaw)
            closed = self.drv.gripper(self.mc.get("gripper_closed", 2.0))
            held = closed > self.mc.get("gripper_closed", 2.0) + self.mc.get("grasp_min_open", 4.0)
            print(f"  gripper closed to {closed:.1f}: {'HOLDING something' if held else 'empty'}")
            self.last_grasp = (
                None if self.piece is None else (self.piece, self.target.copy(), self.q.copy())
            )
            if held and self.last_grasp:
                self._log_grasp(*self.last_grasp)
                self.last_grasp = None
            elif self.last_grasp:
                print("  (if it did hold the piece, type `log` to log this grasp)")
            self.q = self.drv.move_line(self.drv.q(), up, self.yaw, steps=4)
        except RuntimeError as e:
            print(f"  {e}")

    def drop(self) -> None:
        from hfteama.sorter.plan import plan_drop

        b = self.scene.box if self.scene else None
        if b is None:
            print("  box not found (scan)")
            return
        bc = self.cfg["box"]
        sol = plan_drop(
            self.arm,
            b,
            bc["height"],
            bc.get("drop_above", 0.04),
            np.radians(self.mc.get("drop_max_tilt_deg", 60.0)),
        )
        if sol is None:
            print("  can't reach over the box")
            return
        self.drv.move_q(sol.q)
        self.drv.gripper(self.mc.get("gripper_open", 60.0))
        tip = self.arm.placement.to_table(fk(sol.q)[:3, 3])
        print(f"  dropped from {(tip[2] - bc['height']) * 100:.0f} cm over the rim")
        self.drv.go_upright()  # straight up, clear of the box, before going anywhere else
        self.q = self.drv.q()
        self.target = None
        self.piece = None
        print("  arm upright; `scan` to look again, or a piece number")

    def set_height(self, real: float) -> None:
        """The operator measured the tip at `real` m above the table: shift the arm's z."""
        if self.target is None:
            print("  pick a piece first")
            return
        pl = self.arm.placement
        pl.z += real - self.target[2]
        self.target = self.target.copy()
        self.target[2] = real
        self.z_changed = True
        print(f"  z_offset now {pl.z * 100:+.1f} cm (saved with q); aim height {real * 100:.1f} cm")

    def _log_grasp(self, piece: int, target, q) -> None:
        """Save where the tool really had to be for this piece (data for fit-grasps)."""
        import time as _time

        from hfteama.arms.so101_ik import jaw_yaw
        from hfteama.sorter.geometry import log_grasp

        p, pl = self.scene.pieces[piece], self.arm.placement
        heading = jaw_yaw(fk(q)) + pl.yaw
        n = log_grasp(
            self.side,
            {
                "time": _time.strftime("%Y-%m-%d %H:%M:%S"),
                "camera": [round(p.x, 4), round(p.y, 4)],
                "piece_yaw_deg": round(float(np.degrees(p.yaw)), 1),
                "aim": [round(float(target[0]), 4), round(float(target[1]), 4)],
                "heading_deg": round(float(np.degrees(heading)), 1),
                "placement": [pl.x, pl.y, round(float(np.degrees(pl.yaw)), 2), pl.z],
            },
        )
        print(f"  grasp logged ({n} for {self.side}); `hfteama sort fit-grasps {self.side}`")

    def record(self) -> None:
        if self.piece is None:
            print("  go over a piece first")
            return
        # where the tip should have been: the camera's point plus the arm's aim offsets, as
        # planned (so the fit corrects the placement only, not the offsets a second time)
        want = self.planned
        tip = self.drv.tcp()
        off = (self.arm.placement.to_table(tip)[:2] - want) * 100
        self.pairs_arm.append(tip[:2])
        self.pairs_table.append(want)
        print(
            f"  recorded: real = planned aim + ({off[0]:+.1f}, {off[1]:+.1f}) cm"
            f" ({len(self.pairs_arm)} points)"
        )

    # --- loop ---------------------------------------------------------------------------------

    def handle(self, cmd: str) -> bool:
        """One command; False to quit."""
        key, arg = cmd[:1], cmd[1:].strip()
        if cmd.isdigit():
            self.goto_piece(int(cmd) - 1)
        elif cmd == "ok":
            self.record()
        elif cmd == "box":
            self.drop()
        elif cmd == "log":
            if self.last_grasp is None:
                print("  nothing to log: `g` over a piece first")
            else:
                self._log_grasp(*self.last_grasp)
                self.last_grasp = None
        elif cmd == "scan":
            self.scan()
        elif key == "q":
            return False
        elif key == "s" and arg:
            self.step = float(arg) / 1000
        elif key == "y" and arg:
            if self.target is not None and self.move(
                self.target, self.yaw + np.radians(float(arg))
            ):
                self._where()
        elif cmd == "o":
            self.drv.gripper(self.mc.get("gripper_open", 60.0))
        elif cmd == "c":
            print(f"  gripper at {self.drv.gripper(self.mc.get('gripper_closed', 2.0)):.1f}")
        elif cmd == "g":
            self.grasp()
        elif key == "g" and _is_number(arg):
            self.grasp(float(arg) / 100)
        elif key == "z" and _is_number(arg):
            self.set_height(float(arg) / 100)
        elif key in ("f", "b", "l", "r", "u", "d") and (not arg or arg.isdigit()):
            self.nudge(key, int(arg) if arg else 1)
        else:
            print(JOG_HELP.format(step=self.step * 1000))
        return True


def _is_number(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def cmd_jog(args) -> int:
    """Hover over detected pieces, correct by hand, grasp, drop; fit the arm placement from
    the corrections."""
    import threading

    from hfteama.sorter.robot import ArmDriver

    cfg = load_config(args.sorter_config)
    side = args.arm
    if side not in cfg["arms"]:
        raise SystemExit(f"Unknown arm {side!r}; config has {list(cfg['arms'])}")
    arm = arm_models(cfg, [side])[side]
    stop = threading.Event()
    with Eye(cfg) as eye:
        drv = ArmDriver(side, cfg["arms"][side], cfg["motion"], args.config).connect()
        jog = Jog(cfg, side, arm, drv, eye, args.z, args.step / 1000)
        jog.lock = threading.Lock()
        viewer = None
        try:
            jog.scan()
            if not args.no_viewer:
                import rerun as rr

                rr.init("hfteama_sort_jog", spawn=True)
                viewer = threading.Thread(target=jog.stream, args=(stop,), daemon=True)
                viewer.start()
            print(JOG_HELP.format(step=jog.step * 1000))
            while True:
                cmd = input(f"[{side}] > ").strip().lower()
                if cmd and not jog.handle(cmd):
                    break
        finally:
            stop.set()
            if viewer:
                viewer.join(timeout=2)
            drv.disconnect()
    old = Placement.from_config(cfg["arms"][side])
    if jog.pairs_arm or jog.z_changed:
        if jog.pairs_arm:
            fit, rms = fit_placement(jog.pairs_arm, jog.pairs_table, yaw_guess=arm.placement.yaw)
        else:
            fit, rms = Placement(arm.placement.x, arm.placement.y, arm.placement.yaw), 0.0
        fit.z = arm.placement.z
        print(
            f"placement from {len(jog.pairs_arm)} points: base ({fit.x:.4f}, {fit.y:.4f}), heading"
            f" {np.degrees(fit.yaw):.1f} deg, z_offset {fit.z * 100:+.1f} cm,"
            f" rms {rms * 1000:.1f} mm (was ({old.x:.3f}, {old.y:.3f}),"
            f" {np.degrees(old.yaw):.1f} deg, {old.z * 100:+.1f} cm)"
        )
        if input("Save it as this arm's placement? [y/N] ").strip().lower() == "y":
            print("saved", save_placement(side, fit, rms, len(jog.pairs_arm)))
    return 0


def cmd_fit_grasps(args) -> int:
    """Fit the arm's grasp aim offsets from the grasps logged by `sort jog`."""
    import json

    from hfteama.sorter.geometry import GRASPS_FILE, PLACEMENTS_FILE, fit_aim

    cfg = load_config(args.sorter_config)
    side = args.arm
    entries = json.loads(GRASPS_FILE.read_text()).get(side, []) if GRASPS_FILE.exists() else []
    skipped = [e for e in entries if e.get("excluded")]
    entries = [e for e in entries if not e.get("excluded")]
    if skipped:
        print(f"({len(skipped)} grasps marked excluded in {GRASPS_FILE.name})")
    if not entries:
        print(f"No grasps logged for {side}: grasp pieces with `hfteama sort jog {side}` (g).")
        return 1
    pl = Placement.from_config(cfg["arms"][side])
    # a grasp's aim is a table point under the placement in use then: re-express it with
    # today's placement (same arm-frame point), so a corrected base doesn't skew the fit
    for e in entries:
        if "placement" in e:
            x, y, yaw, z = e["placement"]
            then = Placement(x, y, np.radians(yaw), z)
            e["aim"] = list(pl.to_table(then.to_arm(e["aim"]))[:2])
    aim, jaw, rms = fit_aim(entries, pl.yaw)
    print(
        f"{len(entries)} grasps, jaw headings: "
        + ", ".join(f"{e['heading_deg']:.0f}" for e in entries)
    )
    print(
        f"aim_offset (forward, left) = ({aim[0] * 100:+.1f}, {aim[1] * 100:+.1f}) cm;"
        f" jaw_offset (along, across the jaws) = ({jaw[0] * 100:+.1f}, {jaw[1] * 100:+.1f}) cm;"
        f" rms {rms * 100:.1f} cm"
    )
    if not jaw.any():
        print("(jaw headings too similar to separate the turning part: grasp rotated pieces)")
    if args.save or input("Save? [y/N] ").strip().lower() == "y":
        data = json.loads(PLACEMENTS_FILE.read_text()) if PLACEMENTS_FILE.exists() else {}
        d = data.setdefault(side, {})
        d["aim_offset"] = [round(float(aim[0]), 4), round(float(aim[1]), 4)]
        d["jaw_offset"] = [round(float(jaw[0]), 4), round(float(jaw[1]), 4)]
        PLACEMENTS_FILE.write_text(json.dumps(data, indent=2) + "\n")
        print("saved", PLACEMENTS_FILE)
    return 0


def cmd_run(args) -> int:
    """Sort: look, plan, one pick per turn, alternating arms; until the table is clear."""
    from hfteama.sorter.robot import ArmDriver

    if args.image and not args.dry_run:
        raise SystemExit("--image only works with --dry-run (-n)")
    cfg = load_config(args.sorter_config)
    sides = _sides(args, cfg)
    arms = arm_models(cfg, sides)
    mc, pc = cfg["motion"], cfg["pieces"]
    drivers: dict[str, ArmDriver] = {}
    failed: list[tuple[float, float]] = []  # where grasps missed; skipped after 2 misses
    picked = 0
    try:
        with Eye(cfg, args.image) as eye:
            if not args.dry_run:
                for s in sides:
                    drivers[s] = ArmDriver(s, cfg["arms"][s], mc, args.config).connect()
            turn = 0
            while args.max_picks is None or picked < args.max_picks:
                chains = [
                    np.array([arms[s].placement.to_table(p) for p in chain_points(d.q())])
                    for s, d in drivers.items()
                ]
                frame, pose, scene = eye.look(chains)
                plan = make_plan(scene, arms, cfg)
                save_image(draw(frame, pose, eye.intr, scene, plan, cfg, arms), "plan")
                todo = {
                    s: [
                        p
                        for p in plan.for_arm(s)
                        if s in plan.drops and _misses(failed, p.piece) < 2
                    ]
                    for s in sides
                }
                if not any(todo.values()):
                    report(scene, plan, arms)
                    print("Nothing (more) to pick.")
                    return 0
                order = sides[turn % len(sides) :] + sides[: turn % len(sides)]
                side = next(s for s in order if todo[s])
                turn = sides.index(side) + 1
                pick = todo[side][0]
                p = pick.piece
                print(f"[{side}] piece at ({p.x:.3f}, {p.y:.3f}) {p.length * 100:.0f} cm long")
                if args.dry_run:
                    report(scene, plan, arms)
                    return 0
                ok = pick_and_drop(drivers[side], pick, plan.drops[side], mc, pc)
                if ok:
                    picked += 1
                else:
                    failed.append((p.x, p.y))
    except KeyboardInterrupt:
        print("\nStopping: arms going home.")
    finally:
        for d in drivers.values():
            d.disconnect()
    print(f"Picked {picked}.")
    return 0


def _misses(failed, piece, radius=0.02) -> int:
    return sum(np.hypot(x - piece.x, y - piece.y) < radius for x, y in failed)


def pick_and_drop(drv, pick, drop, mc: dict, pc: dict) -> bool:
    """One pick: open, above, down, close, up, over the box, open, home. True if something
    was in the gripper when it closed."""
    yaw = float(np.arctan2(fk(pick.grasp.q)[1, 0], fk(pick.grasp.q)[0, 0]))
    try:
        drv.gripper(mc.get("gripper_open", 60.0), wait=0.0)
        drv.move_q(pick.above.q)
        drv.move_line(pick.above.q, pick.target_arm, yaw)
        closed = drv.gripper(mc.get("gripper_closed", 2.0))
        holding = closed > mc.get("gripper_closed", 2.0) + mc.get("grasp_min_open", 4.0)
        up = fk(pick.above.q)[:3, 3]
        drv.move_line(drv.q(), up, yaw, steps=4)
        if not holding:
            print(f"[{drv.side}] missed (gripper closed to {closed:.1f})")
            drv.gripper(mc.get("gripper_open", 60.0), wait=0.0)
            drv.go_home()
            return False
        drv.move_q(drop.q)
        drv.gripper(mc.get("gripper_open", 60.0))
        drv.go_upright()  # straight up, clear of the box, then home
        drv.go_home()
        return True
    except (RuntimeError, ValueError) as e:
        print(f"[{drv.side}] {e}; going home")
        drv.go_home()
        return False


def add_parser(sub) -> None:
    s = sub.add_parser("sort", help="Vision-guided IK sorter: pieces on the table into the box")
    s.add_argument("--sorter-config", help="default: configs/sorter.json")
    ss = s.add_subparsers(dest="sort_command", required=True)

    p = ss.add_parser("preview", help="Camera only: detect pieces and box, show the plan")
    p.add_argument("--arms", help="Arms to plan for, e.g. rakesh,hacklab (default: all)")
    p.add_argument("--image", help="Use a saved frame instead of the camera")
    p.add_argument("--live", action="store_true", help="Stream to Rerun instead of one saved image")

    t = ss.add_parser("touch", help="Fit an arm's placement by touching tags (torque off)")
    t.add_argument("arm", help="Arm name from the config: rakesh or hacklab")

    g = ss.add_parser("goto", help="Hover the tool tip over a table point, then go home")
    g.add_argument("arm", help="Arm name from the config: rakesh or hacklab")
    g.add_argument("--tag", type=int, help="Reference tag id")
    g.add_argument("--xy", type=float, nargs=2, metavar=("X", "Y"), help="Table point (m)")
    g.add_argument("--box", action="store_true", help="Over the box, found by the camera")
    g.add_argument(
        "--z",
        type=float,
        help="Height (m): above the table (default 0.03), or above the rim with --box",
    )
    g.add_argument("--image", help="With -n and --box: find the box in a saved frame")

    j = ss.add_parser("jog", help="Hover over a piece, correct by hand, fit the arm's placement")
    j.add_argument("arm", help="Arm name from the config: rakesh or hacklab")
    j.add_argument("--z", type=float, default=0.05, help="Hover height above the table (m)")
    j.add_argument("--step", type=float, default=5.0, help="Jog step (mm)")
    j.add_argument("--no-viewer", action="store_true", help="Don't open the Rerun viewer")

    fg = ss.add_parser("fit-grasps", help="Fit grasp aim offsets from the grasps jog logged")
    fg.add_argument("arm", help="Arm name from the config: rakesh or hacklab")
    fg.add_argument("--save", action="store_true", help="Save without asking")

    r = ss.add_parser("run", help="Sort: pick every piece into the box")
    r.add_argument("--arms", help="Arms to use, e.g. rakesh or rakesh,hacklab (default: all)")
    r.add_argument("--max-picks", type=int, help="Stop after this many picks")
    r.add_argument("--image", help="With --dry-run: plan from a saved frame")


def dispatch(args) -> int:
    return {
        "preview": cmd_preview,
        "touch": cmd_touch,
        "goto": cmd_goto,
        "jog": cmd_jog,
        "fit-grasps": cmd_fit_grasps,
        "run": cmd_run,
    }[args.sort_command](args)
