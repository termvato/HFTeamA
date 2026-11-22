"""Mirror a running MuJoCo `SimTable` into Isaac Sim (scripts/isaac/mirror.py renders it).

MuJoCo keeps running the task (IK, timing, grasps, locks); Isaac Sim only shows it, with RTX.

1. On start, every visible shape of the compiled MuJoCo scene (geom groups 0-2: the arms' visual
   meshes, the objects, the location discs) is written once to `<cache>/scene_<id>.npz`: per body,
   its geoms with their local pose, size, colour and, for meshes, vertices and faces.
2. Then, `hz` times a second, the world pose of every body that has shapes is sent over UDP:
     b"MJM1" | scene id (u32) | sim time (f64) | n (u32) | n x [x y z qw qx qy qz] (f32)
   and once a second the scene file:  b"MJS1" | scene id (u32) | path (utf-8).

With `record=<file>` every packet is also appended to a file (u32 length, then the packet), the
scene packet first: `mirror.py --replay <file>` renders it afterwards frame by frame, for a video
at full frame rate whatever the render speed.

The receiver can start before or after the sim and picks up whichever scene arrives. Both must
run on the same machine (the scene goes by file path). Isaac Sim draws nothing that MuJoCo does
not have, so what is shown is what was simulated.

    with SimTable(poses) as table, IsaacMirror(table, port=9871):
        ...
"""

from __future__ import annotations

import hashlib
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

STATE_MAGIC = b"MJM1"
SCENE_MAGIC = b"MJS1"
DEFAULT_PORT = 9871
VISIBLE_GROUPS = (0, 1, 2)  # the SO-101 model: 2 = visual meshes, 3-4 = collision
DEFAULT_CACHE = Path(
    os.environ.get("HFTEAMA_ISAAC_CACHE", Path.home() / ".cache" / "hfteama_isaac")
)


def export_scene(model: Any, path: Path) -> np.ndarray:
    """Write the visible geoms of `model` to `path` (.npz); returns the body ids in packet order."""
    m = model
    geoms = [
        g
        for g in range(m.ngeom)
        if m.geom_group[g] in VISIBLE_GROUPS
        and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE  # the floor: the receiver draws a table
        and _rgba(m, g)[3] > 0
    ]
    bodies = sorted({int(m.geom_bodyid[g]) for g in geoms})
    index = {b: i for i, b in enumerate(bodies)}
    meshes = sorted(
        {int(m.geom_dataid[g]) for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    )
    mesh_index = {k: i for i, k in enumerate(meshes)}
    verts, faces, vadr, fadr, vnum, fnum = [], [], [], [], [], []
    nv = nf = 0
    for k in meshes:
        a, n = m.mesh_vertadr[k], m.mesh_vertnum[k]
        fa, fn = m.mesh_faceadr[k], m.mesh_facenum[k]
        verts.append(m.mesh_vert[a : a + n])
        faces.append(m.mesh_face[fa : fa + fn])  # indices local to this mesh
        vadr.append(nv), vnum.append(n), fadr.append(nf), fnum.append(fn)
        nv, nf = nv + n, nf + fn
    np.savez_compressed(
        path,
        body_names=np.array([m.body(b).name or "world" for b in bodies]),
        body_static=np.array([b == 0 for b in bodies]),
        g_body=np.array([index[int(m.geom_bodyid[g])] for g in geoms], dtype=np.int32),
        g_name=np.array([m.geom(g).name for g in geoms]),
        g_type=np.array([int(m.geom_type[g]) for g in geoms], dtype=np.int32),
        g_size=np.array([m.geom_size[g] for g in geoms], dtype=np.float32).reshape(-1, 3),
        g_pos=np.array([m.geom_pos[g] for g in geoms], dtype=np.float32).reshape(-1, 3),
        g_quat=np.array([m.geom_quat[g] for g in geoms], dtype=np.float32).reshape(-1, 4),
        g_rgba=np.array([_rgba(m, g) for g in geoms], dtype=np.float32).reshape(-1, 4),
        g_mesh=np.array(
            [
                mesh_index.get(int(m.geom_dataid[g]), -1)
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                else -1
                for g in geoms
            ],
            dtype=np.int32,
        ),
        mesh_vert=np.concatenate(verts).astype(np.float32)
        if verts
        else np.zeros((0, 3), np.float32),
        mesh_face=np.concatenate(faces).astype(np.int32) if faces else np.zeros((0, 3), np.int32),
        mesh_vadr=np.array(vadr, dtype=np.int32),
        mesh_vnum=np.array(vnum, dtype=np.int32),
        mesh_fadr=np.array(fadr, dtype=np.int32),
        mesh_fnum=np.array(fnum, dtype=np.int32),
    )
    return np.array(bodies, dtype=np.int32)


def _rgba(m: Any, g: int) -> np.ndarray:
    mat = m.geom_matid[g]
    return m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]


class IsaacMirror:
    """Streams a running `SimTable` to an Isaac Sim receiver (scripts/isaac/mirror.py)."""

    def __init__(
        self,
        table: Any,
        port: int = DEFAULT_PORT,
        host: str = "127.0.0.1",
        hz: float = 30.0,
        cache: Path = DEFAULT_CACHE,
        record: Path | None = None,
    ) -> None:
        self.table = table
        self.addr = (host, port)
        self.hz = hz
        cache.mkdir(parents=True, exist_ok=True)
        # The scene id changes when the scene does (another task, layout or arm count)
        model = table._model  # noqa: SLF001  (SimTable owns the model; read-only use)
        digest = hashlib.sha1(
            np.concatenate(
                [model.geom_size.ravel(), model.geom_pos.ravel(), model.body_pos.ravel()]
            ).tobytes()
        ).digest()
        self.scene_id = struct.unpack("<I", digest[:4])[0]
        self.scene_path = cache / f"scene_{self.scene_id:08x}.npz"
        self.bodies = export_scene(model, self.scene_path)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent = 0
        self._record = None
        if record is not None:
            Path(record).parent.mkdir(parents=True, exist_ok=True)
            self._record = open(record, "wb")  # noqa: SIM115  (closed in __exit__)
            self._write(self.scene_packet())

    def __enter__(self) -> IsaacMirror:
        self._thread = threading.Thread(target=self._run, name="isaac-mirror", daemon=True)
        self._thread.start()
        print(
            f"[isaac] mirroring {len(self.bodies)} bodies to udp://{self.addr[0]}:{self.addr[1]} "
            f"(scene {self.scene_path})",
            flush=True,
        )
        return self

    def __exit__(self, *exc: object) -> None:
        self._quit.set()
        if self._thread is not None:
            self._thread.join(1.0)
        self.send_state()  # the final pose (after the thread: one writer to the recording)
        self._sock.close()
        if self._record is not None:
            self._record.close()

    def _write(self, packet: bytes) -> None:
        if self._record is not None:
            self._record.write(struct.pack("<I", len(packet)) + packet)

    def scene_packet(self) -> bytes:
        return SCENE_MAGIC + struct.pack("<I", self.scene_id) + str(self.scene_path).encode()

    def state_packet(self) -> bytes:
        t = self.table
        with t._lock:  # noqa: SLF001  (a consistent snapshot between physics steps)
            pos = t._data.xpos[self.bodies].astype(np.float32)  # noqa: SLF001
            quat = t._data.xquat[self.bodies].astype(np.float32)  # noqa: SLF001
            sim_t = float(t._data.time)  # noqa: SLF001
        head = STATE_MAGIC + struct.pack("<IdI", self.scene_id, sim_t, len(self.bodies))
        return head + np.hstack([pos, quat]).tobytes()

    def send_state(self) -> None:
        packet = self.state_packet()
        self._write(packet)
        try:
            self._sock.sendto(packet, self.addr)
            self.sent += 1
        except OSError:
            pass  # no receiver yet, or it went away: keep simulating

    def _run(self) -> None:
        last_scene = 0.0
        period = 1.0 / self.hz
        while not self._quit.is_set():
            now = time.monotonic()
            if now - last_scene >= 1.0:
                try:
                    self._sock.sendto(self.scene_packet(), self.addr)
                except OSError:
                    pass
                last_scene = now
            self.send_state()
            self._quit.wait(max(0.0, period - (time.monotonic() - now)))


def parse_state(data: bytes) -> tuple[int, float, np.ndarray] | None:
    """(scene id, sim time, n x 7 poses) from a state packet, or None if it is not one."""
    if not data.startswith(STATE_MAGIC) or len(data) < 20:
        return None
    scene_id, sim_t, n = struct.unpack_from("<IdI", data, 4)
    poses = np.frombuffer(data, dtype=np.float32, offset=20)
    if poses.size != n * 7:
        return None
    return scene_id, sim_t, poses.reshape(n, 7)


def parse_scene(data: bytes) -> tuple[int, str] | None:
    """(scene id, scene file path) from a scene packet, or None if it is not one."""
    if not data.startswith(SCENE_MAGIC) or len(data) < 9:
        return None
    (scene_id,) = struct.unpack_from("<I", data, 4)
    return scene_id, data[8:].decode()


def read_recording(path: Path):
    """The packets of a `record=` file, in order."""
    with open(path, "rb") as fh:
        while head := fh.read(4):
            (n,) = struct.unpack("<I", head)
            yield fh.read(n)
