"""Raw Feetech STS3215 access for the two things LeRobot does not do safely for us.

1. `probe`: a passive look at an arm (ping, read position / torque / voltage / temperature). Reads
   only.
2. `hold_present_pose`: while torque is still OFF, write every servo's goal position = its present
   position. LeRobot's `SOFollower.connect()` enables torque without doing this, and a servo drives
   toward whatever goal it holds when torque comes on (a stale one from an earlier session, or the
   default), so an arm can lunge the moment it is connected. This is the only write we make, and it
   makes the arm hold still.

Packet: 0xFF 0xFF, id, length (= params + 2), instruction, params..., checksum (~sum of id..params).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

PING, READ, WRITE = 0x01, 0x02, 0x03
REG_TORQUE, REG_GOAL, REG_POSITION, REG_VOLTAGE, REG_TEMPERATURE = 40, 42, 56, 62, 63


class ArmNotReady(RuntimeError):
    """A servo did not answer, so nothing was written."""


def packet(servo_id: int, instruction: int, params: list[int]) -> bytes:
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([0xFF, 0xFF, *body, ~sum(body) & 0xFF])


def read_register(port: Any, servo_id: int, address: int, length: int) -> bytes | None:
    """One register read; None if the servo does not answer or the reply is damaged."""
    port.reset_input_buffer()
    port.write(packet(servo_id, READ, [address, length]))
    reply = port.read(6 + length)
    if len(reply) != 6 + length or reply[:2] != b"\xff\xff" or reply[2] != servo_id:
        return None
    if (~sum(reply[2:-1]) & 0xFF) != reply[-1]:
        return None
    return bytes(reply[5:-1])


def probe(port: Any, ids: Iterable[int] = range(1, 7)) -> dict[int, dict[str, Any]]:
    """Per answering servo: raw position (0-4095), torque on?, volts, temperature. Reads only."""
    found: dict[int, dict[str, Any]] = {}
    for servo_id in ids:
        pos = read_register(port, servo_id, REG_POSITION, 2)
        if pos is None:
            continue
        torque = read_register(port, servo_id, REG_TORQUE, 1)
        volts = read_register(port, servo_id, REG_VOLTAGE, 1)
        temp = read_register(port, servo_id, REG_TEMPERATURE, 1)
        found[servo_id] = {
            "position": pos[0] | pos[1] << 8,
            "torque": bool(torque[0]) if torque else None,
            "volts": volts[0] / 10.0 if volts else None,
            "temp_c": temp[0] if temp else None,
        }
    return found


def hold_present_pose(port: Any, ids: Iterable[int] = range(1, 7)) -> dict[int, int]:
    """Write goal = present for every servo, so enabling torque holds the pose instead of moving.

    Returns the positions written. Raises `ArmNotReady` (writing nothing) unless every servo in
    `ids` answers: a half-connected arm is never partially written.
    """
    ids = list(ids)
    positions: dict[int, int] = {}
    for servo_id in ids:
        pos = read_register(port, servo_id, REG_POSITION, 2)
        if pos is None:
            raise ArmNotReady(f"servo {servo_id} did not answer: nothing was written")
        positions[servo_id] = pos[0] | pos[1] << 8
    for servo_id, p in positions.items():
        port.reset_input_buffer()
        port.write(packet(servo_id, WRITE, [REG_GOAL, p & 0xFF, p >> 8]))
        reply = port.read(6)
        if len(reply) != 6 or reply[4] != 0:
            raise ArmNotReady(f"servo {servo_id} did not accept the goal write")
    return positions
