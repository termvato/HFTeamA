"""Mirror a real SO-101 in the MuJoCo model: move the arm by hand, watch the model follow.

Checks the sign and zero of every joint at once: if a joint of the model turns the other way
than the real one, its sign is flipped; if it follows but sits off, it needs an offset.

    cd /data/diego/Projects/HFTeam-action
    source ../HFTeam/.venv/bin/activate
    export PYTHONPATH=$PWD/src
    python scripts/real_mirror.py --arm rakesh_follower      # or vato_follower
    python scripts/real_mirror.py --demo                     # no arm: the model sweeps its joints

Torque goes OFF after connecting (so you can move the arm): SUPPORT THE ARM.
Close the window or press Ctrl-C in the terminal to stop. Readings are logged to runs/stage1/.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from hfteama.arms.kinematics import JOINTS
from hfteama.arms.model import MODEL_XML

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "configs" / "real_arms.json"
SHORT = dict(zip(JOINTS, ("pan", "lift", "elbow", "w_flex", "w_roll", "grip"), strict=True))


class Demo:
    """Stands in for the arm: sweeps one joint after the other."""

    def __init__(self):
        self.t0 = time.time()

    def read(self) -> np.ndarray:
        t = time.time() - self.t0
        q = np.zeros(6)
        q[int(t / 3) % 6] = 0.8 * np.sin(2 * np.pi * t / 3)
        return q

    def close(self) -> None:
        pass


class Arm:
    def __init__(self, arm: str):
        from hfteama.arms.real import from_config

        input("SUPPORT THE ARM (connecting turns torque on), then Enter ")
        self.d = from_config(arm, REAL)
        input("Still supporting it? Torque goes OFF now and a raised arm drops. Enter ")
        with self.d._bus_lock:
            self.d._robot.bus.disable_torque()

    def read(self) -> np.ndarray:
        return self.d.read_joints()

    def close(self) -> None:
        input("SUPPORT THE ARM, then Enter to disconnect ")
        try:
            self.d._robot.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f"disconnect: {exc}")


def fmt(q: np.ndarray) -> str:
    return "  ".join(f"{SHORT[j]} {np.degrees(v):7.1f}" for j, v in zip(JOINTS, q, strict=True))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", default="rakesh_follower", help="rakesh_follower or vato_follower")
    ap.add_argument("--demo", action="store_true", help="no arm: sweep the model's joints")
    a = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    adr = [model.jnt_qposadr[model.joint(j).id] for j in JOINTS]

    name = "demo" if a.demo else a.arm
    log = ROOT / "runs" / "stage1" / f"mirror_{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not a.demo:
        offsets = json.loads(REAL.read_text())["arms"][a.arm].get("offsets_deg", {})
        print(f"config offsets_deg for {a.arm}: {offsets or 'none (defaults: elbow 6.9)'}")
    src = Demo() if a.demo else Arm(a.arm)
    print(f"log: {log}\nmove one joint at a time and compare the model with the real arm")

    last_print = 0.0
    try:
        with log.open("a", buffering=1) as f, mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                q = src.read()
                data.qpos[adr] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                if time.time() - last_print > 0.5:
                    line = fmt(q)
                    print(line)
                    f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
                    last_print = time.time()
                time.sleep(1 / 30)
    except KeyboardInterrupt:
        print()
    finally:
        src.close()


if __name__ == "__main__":
    main()
