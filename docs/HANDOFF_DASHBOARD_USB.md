# Handoff: robots over USB, live on the dashboard

For whoever sets up the demo laptop (Windows 11). Branch `frontEndDash`. Written Sat 21 Nov 2026,
20:15. Background: [`ORCHESTRA_DASHBOARD.md`](ORCHESTRA_DASHBOARD.md) (section "How a robot
connects to the stack").

## Read this first

There are two screens. Only the first one is live today.

| You want | Works today? | Go to |
|---|---|---|
| Plug a follower in by USB and see it **join** the run live; pull the cable and see it **leave** (the arm does not move) | **Yes**, on the Dash screen, http://127.0.0.1:8050 | Steps 1 to 3 |
| The real arms **do the work**, joining and leaving by USB | **Not yet.** Needs Action's code merged, measured config files, and the staged first-motion checks. No real arm has moved under the orchestrator yet | Steps 4 and 5 |
| The same, on the new **Orchestra** page (30 / 70 layout) | **Not yet.** The page is built, but the small server that feeds it live data is not. Until then it shows demo data only | Step 6 |

Rules that apply to every step:

- Run every command **from the `HFTeamA` folder**: the config paths are relative to it.
- Only **followers** count (`rakesh_follower`, `vato_follower`). Leaders are for teleoperation
  and are ignored by the orchestrator.
- `frontEndDash` is not on GitHub yet. On another machine, push it first:
  `git push -u origin frontEndDash`.

---

## What you need

### Hardware

| Item | Note |
|---|---|
| SO-101 follower arms, calibrated | Calibrations are committed in `configs/calibration/`. **Do not calibrate again** |
| One USB data cable per follower | Some cables only charge; they do not show up as a COM port |
| 12 V supply per follower, switched on | Powers the servos. USB carries only data |
| The demo laptop | Windows 11; checked with Python 3.13 |

### Software

| Dependency | Version | Install | Needed for |
|---|---|---|---|
| Python | 3.12 or 3.13 (LeRobot needs 3.12+) | python.org or the Microsoft Store | everything |
| git | any | git-scm.com | getting the code |
| `hfteama` with extras `view`, `arms`, `sched` | from `pyproject.toml`: `dash>=2.17`, `plotly>=5.22`, `pyserial>=3.5`, `ortools>=9.10` | `pip install -e ".[view,arms,sched]"` | Steps 1 to 3 |
| Extra `sim` | `mujoco>=4.0.1` (after the merge in Step 4) | `pip install -e ".[sim]"` | Steps 4 and 5: the IK runs in MuJoCo, also for real arms |
| LeRobot | `lerobot[feetech,core-scripts]>=0.7` | `pip install "lerobot[feetech,core-scripts]>=0.7"` | Steps 4 and 5 only |
| `fastapi`, `uvicorn` | `fastapi>=0.115`, `uvicorn>=0.30` (not in `pyproject.toml` yet) | `pip install fastapi uvicorn` | Step 6 only |
| A browser | Chrome or Edge | | both screens |
| USB-serial driver for the follower's servo board | usually built into Windows | only if no COM port appears (see Troubleshooting) | detection |

---

## Step 1. Install (once per laptop)

PowerShell, in the `HFTeamA` folder:

```powershell
git switch frontEndDash
py -3.13 -m venv $HOME\venvs\hfteama
& $HOME\venvs\hfteama\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[view,arms,sched]"
```

The environment lives in your user folder (`$HOME\venvs\hfteama`), not inside `HFTeamA`, because
`HFTeamA` may sit in a OneDrive folder and OneDrive would try to sync the thousands of files
an environment holds (gigabytes once LeRobot is in). If PowerShell says running scripts is disabled,
run `Set-ExecutionPolicy -Scope Process Bypass` and activate again. This only affects the current
window.

Every new PowerShell window: `& $HOME\venvs\hfteama\Scripts\Activate.ps1` first.

## Step 2. Plug in the followers and check that the laptop sees them

1. Switch on each follower's 12 V supply.
2. Plug each follower's USB cable into the laptop.
3. Run:

```powershell
python -c "from hfteama.discovery import connected_serials as c; print(c())"
hfteama pairs
```

**Expected:** the first line prints a set that contains `5B79014975` (`rakesh_follower`) and
`5A7A015861` (`vato_follower`); it may also contain the leaders' serials. `hfteama pairs` shows each
follower as `connected at COM<n>`. The COM number can change every time you plug in; that is fine,
because the arms are found by serial number, never by port.

If a follower is missing, stop here and use the Troubleshooting table. No screen can show an arm
that this step does not see.

## Step 3. Watch the arms join and leave live (no motion)

```powershell
python -m hfteama.demo --backend fake --watch --task tasks/sort/hacklab6.json --table configs/hacklab_table.json
```

Open http://127.0.0.1:8050.

| Do | You should see within about 1 second |
|---|---|
| Plug in a follower | Feed: "Arm rakesh_follower joined"; the estimate updates |
| Press **Go** | The countdown starts; the plan splits the six objects over the joined arms |
| Plug in the second follower mid-run | It joins; feed "Re-split: finish now …"; the countdown drops |
| Pull a follower's USB cable | Feed: "Arm … lost (unplugged), moving its work"; its steps go to the other arm |

What `--backend fake` means: each arm is **detected for real** by its USB serial, but its work is
simulated (the step just waits). The laptop never opens the arm's port, so **the arm does not move
and its torque is never switched on**. This is the safe way to show plug-and-produce on the real
cables.
`--watch` turns on USB detection (it is on by default only for `--backend real`).

Stop with `Ctrl+C`. Each run writes its event log to `runs/<date>-<time>/events.jsonl`.

## Step 4. Prepare real arms (once, Hardware and Action together)

Do these in order. Each one has a check.

| # | Do | Check |
|---|---|---|
| 4a | Bring in Action's arm code: `git merge origin/action-arm-skills` | Merges without conflicts (tested 21 Nov at `75278b2`). Afterwards `src/hfteama/arms/worker.py` and `real.py` exist |
| 4b | `pip install -e ".[view,arms,sched,sim]"` | `python -c "import mujoco; print(mujoco.__version__)"` prints 4.0.1 or newer |
| 4c | `pip install "lerobot[feetech,core-scripts]>=0.7"` | `python -c "from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig"` prints nothing. If this import fails, Action's handoff says that line in `arms/real.py::from_config` is the only thing to fix |
| 4d | Fill in `configs/real_arms.json`: one entry per follower, keyed `rakesh_follower` and `vato_follower`. Template: `configs/real_arms.example.json`; field guide: `docs/handoff/action-to-hardware.md` | Both arm ids are present. **Today the file has only `vato_follower`, as a one-arm bench test, with the base in the arm's own frame**, so it cannot be used for the table run yet |
| 4e | Make the arm bases agree. The planner reads bases from `configs/hacklab_table.json`; the driver's IK reads them from `configs/real_arms.json`. Both must be the **same measured pose in the table frame** (ArUco marker 0, metres; bases measured by touching two markers with each gripper, Blockers #2) | Same numbers in both files. **If they differ, an arm is sent to the wrong place** |
| 4f | Create `tasks/sort/poses_real.json` from `tasks/sort/poses_real.example.json`, with measured points keyed by the follower ids. The example's numbers are from the simulator and are wrong for the table | `python -m hfteama.arm_check check-config --poses tasks/sort/poses_real.json` prints no problems. (With the example file today it lists reach problems, as expected) |
| 4g | Per arm, the supervised first-motion checks, in order: `probe`, `read`, `joint`, `point`, `descend`, `step`, `park` (`python -m hfteama.arm_check --help`; details in `docs/handoff/action-to-hardware.md`) | Each stage passes with a person at the rig. Nothing in this list has been run on a real arm yet |

## Step 5. Real run, with live USB detection

Only after Step 4 is complete. A person at the rig, a hand on the 12 V supply.

```powershell
python -m hfteama.demo --backend real --task tasks/sort/hacklab6.json --table configs/hacklab_table.json --poses tasks/sort/poses_real.json
```

Open http://127.0.0.1:8050. USB detection is on by default with `--backend real`.

**Safety: plugging a follower in now connects to it at once and turns its torque on**, with no
prompt. The arm holds whatever pose it is in, so support a raised arm before plugging in. Then:

- **Remove** in the screen's arm controls is a slow stop: finish the step, park, torque off.
- Pulling the USB cable drops the arm from the run immediately, but does not park it: the arm
  stays where it stopped. Use Remove to take an arm out on purpose.
- The on-screen controls are software stops. **The emergency stop is the follower's 12 V supply.**
- On Windows only one program can open a COM port: close teleop, recording and `arm_check` first.

The runner starts even without LeRobot or a complete config (checked 21 Nov). The problem shows
only when an arm is plugged in: the console logs `arm <id> could not start: …` and the arm does not
join. Read that line; the Troubleshooting table lists the usual causes.

## Step 6. Live on the Orchestra page (still to build)

The Orchestra page is in `src/hfteama/orchestra/static/`. Opened today, it says "Demo data" and
never looks at USB. To make it live, someone builds milestone M0/M1 of
[`ORCHESTRA_DASHBOARD.md`](ORCHESTRA_DASHBOARD.md) (sections 5 and 8):

1. Add to `pyproject.toml`: `orchestra = ["fastapi>=0.115", "uvicorn>=0.30"]`, then
   `pip install -e ".[orchestra]"`.
2. Build `src/hfteama/orchestra/` (API, state hub, commands) and a `--orchestra` flag on
   `python -m hfteama.demo` that starts it on port 8060 next to the Dash screen. It serves the
   page from `static/`.
3. Add the small executor and watcher additions it reads: G3 (richer `snapshot()`), G4
   (`ArmWatcher.present()`), G5 (`Rig.plug(arm_id)`), G6 (the flags).

**Done when:**
`python -m hfteama.demo --backend fake --watch --task tasks/sort/hacklab6.json --table configs/hacklab_table.json --orchestra`
gives http://127.0.0.1:8060 showing **Live** (not "Demo"), and plugging in a follower moves the
counter from `1 / 2` to `2 / 2` and adds its card within about 1 second.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Step 2 prints `set()` | No arm seen: cable out, charge-only cable, or no driver | Device Manager → "Ports (COM & LPT)": the board must appear there when plugged in. Try another (data) cable or USB port. If it appears with a warning sign or not at all, install the USB-serial driver from the board maker |
| A serial is printed, but the arm never joins | It is a leader, or a follower whose serial is not in `configs/teleop_pairs.json` | Use the follower. A new follower needs its serial added there (Hardware) |
| `hfteama pairs` says `not connected` | Same as above, or the 12 V supply is off | Check cable and supply |
| `ModuleNotFoundError: No module named 'serial'` | Extra `arms` missing | `pip install -e ".[arms]"` |
| `ModuleNotFoundError: No module named 'dash'` | Extra `view` missing | `pip install -e ".[view]"` |
| `hfteama` is not recognised | The virtual environment is not active, or Step 1 was skipped | `& $HOME\venvs\hfteama\Scripts\Activate.ps1`, then `pip install -e ".[view,arms,sched]"` |
| `--backend real needs Action's arm code …` and the runner exits | Step 4a not done | `git merge origin/action-arm-skills` |
| Console: `arm rakesh_follower could not start: 'rakesh_follower'` (the arm id in quotes) | No entry for that arm in `configs/real_arms.json` | Step 4d |
| Console: `arm <id> could not start: No module named 'lerobot'` | LeRobot not installed | Step 4c |
| Console: `arm <id> could not start` mentioning access denied or the port being busy | Another program has the COM port open | Close teleop, recording or `arm_check`, then unplug and replug |
| An arm joins and is dropped again about 1.5 s later | Heartbeat lost: the arm stopped answering on the bus | Check the 12 V supply and the cable; replug |
| Page at :8050 does not load | Runner not running, or port 8050 taken | Look at the console; or add `--port 8051` and open that |
| The Orchestra page says "Demo data" | Expected until Step 6 is built | Use the Dash screen at :8050 |

---

## What was checked, and what was not (21 Nov, 20:10, this laptop)

Windows 11, Python 3.13.14, fresh virtual environment, `frontEndDash` with `origin/action-arm-skills`
(`75278b2`) merged in a throwaway copy:

| Check | Result |
|---|---|
| `pip install -e ".[view,arms,sched,sim,dev]"` | OK: dash 4.5.1, plotly 7.1.0, pyserial 3.5, ortools 9.15.6755, mujoco 4.0.1, numpy 2.5.5 |
| Merge `origin/action-arm-skills` | No conflicts |
| `pytest tests/test_discovery.py tests/test_demo.py` (plug in, unplug, debounce, runner) | 8 passed |
| Step 3 command | Starts, `USB watch: True`, page answers on :8050 |
| Step 5 command, without LeRobot and without arms | Starts, `USB watch: True`, page answers |
| `hfteama pairs`, `python -m hfteama.arm_check check-config` | Run; check-config reports the expected reach problems with the example poses |
| `connected_serials()` | `set()`: **no robot was plugged into this laptop**, so detection of the real followers on Windows is **not yet confirmed**. Step 2 is that test |
| LeRobot on Windows, real-arm motion, Orchestra API | **Not checked**: not installed, not run, not built |
