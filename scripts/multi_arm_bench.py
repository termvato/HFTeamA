import sys
import time

import mujoco

src = (
    sys.argv[1] if len(sys.argv) > 1 else "mujoco_menagerie/robotstudio_so101/so101.xml"
)  # path to the Menagerie SO-101 model
for N in (1, 4, 8, 16):
    world = mujoco.MjSpec()
    world.worldbody.add_light(pos=[0, 0, 3])
    world.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.1])
    for i in range(N):
        arm = mujoco.MjSpec.from_file(src)
        f = world.worldbody.add_frame(pos=[0.45 * (i % 4), 0.5 * (i // 4), 0])
        f.attach_body(arm.worldbody.first_body(), f"arm{i}_", "")
    m = world.compile()
    d = mujoco.MjData(m)
    t = time.perf_counter()
    steps = 5000
    for _ in range(steps):
        mujoco.mj_step(m, d)
    dt = time.perf_counter() - t
    print(
        f"N={N}: nq={m.nq} nu={m.nu} timestep={m.opt.timestep} -> {steps / dt:.0f} steps/s, "
        f"realtime x{steps * m.opt.timestep / dt:.1f}"
    )
