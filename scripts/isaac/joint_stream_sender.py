"""Stream joint positions to so101_scene.py over UDP (the "live bridge" test).

Plain Python, no Isaac Sim needed: this is what an executor, a MuJoCo run or a reader of a real
arm's present positions would do to mirror arms in the RTX scene. It sends only to localhost and
never touches a robot or serial port.

Packet: one JSON object per datagram, {"arm": <index>, "q": [6 joint angles in radians]}, in
joint order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (the MJCF /
LeRobot order).

Run (two terminals):
  env -u PYTHONPATH <isaacsim>/python.sh scripts/isaac/so101_scene.py \
      --mjcf <...>/so101.xml --listen 5605
  python3 scripts/isaac/joint_stream_sender.py --port 5605 --arm 1 --seconds 10
"""

import argparse
import json
import math
import socket
import time

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5605)
parser.add_argument("--arm", type=int, default=1)
parser.add_argument("--rate", type=float, default=50.0, help="packets per second")
parser.add_argument("--seconds", type=float, default=10.0)
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t0, sent = time.time(), 0
while time.time() - t0 < args.seconds:
    t = time.time() - t0
    # A slow "reach down and come back" motion, clearly different from the scene's own sweep.
    q = [
        0.8 * math.sin(0.5 * t),
        -0.6 + 0.5 * math.sin(t),
        0.9 - 0.5 * math.sin(t),
        0.6,
        0.0,
        0.8 * (1 + math.sin(2 * t)) / 2,
    ]
    sock.sendto(json.dumps({"arm": args.arm, "q": q}).encode(), ("127.0.0.1", args.port))
    sent += 1
    time.sleep(1.0 / args.rate)
print(f"sent {sent} packets to udp://127.0.0.1:{args.port} for arm {args.arm}")
