"""Joint streamer for the 3D rehearsal scene (#33): one datagram per arm, bad readers skipped."""

import json
import socket

from hfteama.view.joint_stream import JointStreamer


def test_streams_each_arm_and_skips_a_broken_reader() -> None:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)

    def broken() -> list[float]:
        raise OSError("bus")

    s = JointStreamer(
        {"left": lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "dead": broken},
        port=rx.getsockname()[1],
    )
    s.add("right", lambda: [0.0] * 6)
    s.send_once()
    got = [json.loads(rx.recv(1024)) for _ in range(2)]
    s.stop()
    rx.close()
    assert {m["id"]: m["arm"] for m in got} == {"left": 0, "right": 2}
    assert got[0]["q"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] and s.sent == 2
