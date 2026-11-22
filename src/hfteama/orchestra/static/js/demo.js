/* Orchestra demo source: an in-browser stand-in for the orchestrator, used when the Orchestra API
 * (docs/ORCHESTRA_DASHBOARD.md, section 5) is not running.
 *
 * It is NOT the executor. It replays the shape of a real run so the screen can be built, shown and
 * rehearsed without hardware: the hacklab6 batch (tasks/sort/hacklab6.json), the arm positions and
 * reach from configs/hacklab_table.json and registry.SO101_REACH, a nearest-reachable-arm dispatch,
 * the bowl as a shared area, retries, slow stop and lost arms. Joint angles are a smooth animation
 * toward each waypoint, not IK. Everything it produces is labelled "demo data" on screen.
 */
(function () {
  'use strict';
  const O = (window.Orchestra = window.Orchestra || {});
  const D2R = Math.PI / 180;

  // Menagerie SO-101 joint limits [rad], same as src/hfteama/arms/kinematics.py LIMITS_RAD.
  const LIMITS = [
    [-1.91986, 1.91986], // shoulder_pan
    [-1.7453293, 1.7453293], // shoulder_lift
    [-1.69, 1.69], // elbow_flex
    [-1.658063, 1.658063], // wrist_flex
    [-2.7438473, 2.8412063], // wrist_roll
    [-0.174533, 1.7453292], // gripper
  ];
  O.LIMITS = LIMITS;
  const REACH = { r_min: 0.14, r_max: 0.42, pan_offset: 0.0388, pan_limit: 110 * D2R };
  const PICK_S = 5.0;
  const PLACE_S = 4.0;
  const MAX_RETRIES = 2;
  const BOWL = { id: 'bowl', label: 'bowl', xyz: [0.317, 0.126, 0.1] };

  // configs/teleop_pairs.json + configs/hacklab_table.json. The third arm is not set up yet
  // (Blockers #6 in OPERATIONS_STATUS.md): it exists here only so the 3-arm story can be shown.
  const CATALOG = [
    { arm_id: 'rakesh_follower', pair: 'rakesh', serial: '5B79014975', label: 'Black arm · left', color: '#30343a', base: [0.089, 0.42, 0], yaw: -90 },
    { arm_id: 'vato_follower', pair: 'hacklab', serial: '5A7A015861', label: 'Green arm · right', color: '#2e8b57', base: [0.488, 0.44, 0], yaw: -90 },
    { arm_id: 'third_follower', pair: '—', serial: 'to come', label: 'Third arm · front (demo only)', color: '#3b6fb6', base: [0.317, -0.16, 0], yaw: 90 },
  ];

  // tasks/sort/hacklab6.json
  const OBJECTS = [
    { id: 'lip_balm', label: 'lip balm', xyz: [0.1, 0.1, 0.02] },
    { id: 'tube', label: 'tube', xyz: [0.2, 0.02, 0.02] },
    { id: 'tape', label: 'tape', xyz: [0.26, 0.16, 0.02] },
    { id: 'snack_bar', label: 'snack bar', xyz: [0.42, 0.04, 0.02] },
    { id: 'marker', label: 'marker', xyz: [0.52, 0.14, 0.02] },
    { id: 'clip', label: 'clip', xyz: [0.6, 0.06, 0.02] },
  ];

  const HOME_Q = [0, -1.55, 1.5, 1.05, 0, 0.9];
  const PARK_Q = [0, -1.7, 1.62, 1.2, 0, 0.0];

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const add = (p, dz) => [p[0], p[1], p[2] + dz];
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

  function panAxis(c) {
    const y = c.yaw * D2R;
    return [c.base[0] + REACH.pan_offset * Math.cos(y), c.base[1] + REACH.pan_offset * Math.sin(y)];
  }

  function reaches(c, p) {
    const a = panAxis(c);
    const dx = p[0] - a[0];
    const dy = p[1] - a[1];
    const r = Math.hypot(dx, dy);
    let ang = Math.atan2(dy, dx) - c.yaw * D2R;
    ang = Math.atan2(Math.sin(ang), Math.cos(ang));
    return r >= REACH.r_min && r <= REACH.r_max && Math.abs(ang) <= REACH.pan_limit;
  }

  // Smooth, plausible joint angles toward a table point (display only, not the IK solver).
  function poseToward(c, p, grip) {
    const a = panAxis(c);
    const dx = p[0] - a[0];
    const dy = p[1] - a[1];
    let pan = Math.atan2(dy, dx) - c.yaw * D2R;
    pan = Math.atan2(Math.sin(pan), Math.cos(pan));
    const r = Math.hypot(dx, dy);
    const L1 = 0.116;
    const L2 = 0.135;
    const wx = Math.max(0.03, r - 0.03);
    const wz = p[2] + 0.1 - 0.12;
    const d = Math.min(Math.hypot(wx, wz), L1 + L2 - 1e-3);
    const elbow = Math.acos(clamp((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1));
    const lift = Math.atan2(wz, wx) + Math.atan2(L2 * Math.sin(elbow), L1 + L2 * Math.cos(elbow));
    const q1 = Math.PI / 2 - lift - 0.15;
    const q2 = elbow - Math.PI / 2;
    const q3 = 0.3 - (q1 + q2) * 0.6; // keeps the tool roughly down, well inside the wrist limits
    const q = [pan, q1, q2, q3, pan * 0.15, grip];
    return q.map((v, i) => clamp(v, LIMITS[i][0] + 0.02, LIMITS[i][1] - 0.02));
  }

  function gripRad(pct) {
    return LIMITS[5][0] + (LIMITS[5][1] - LIMITS[5][0]) * pct / 100;
  }

  function hashErr(s) {
    let h = 7;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 1000;
    return 0.2 + (h % 70) / 100; // 0.2 .. 0.9 mm
  }

  function waypoints(c, step) {
    const pts =
      step.skill === 'pick'
        ? [['approach', add(step.xyz, 0.08)], ['grasp', step.xyz], ['lift', add(step.xyz, 0.1)]]
        : [['approach', add(BOWL.xyz, 0.08)], ['release', BOWL.xyz]];
    return pts.map(([name, xyz]) => {
      const ok = reaches(c, xyz);
      return { name, xyz, ok, error_mm: ok ? hashErr(c.arm_id + step.id + name) : null };
    });
  }

  function DemoSource(opts) {
    opts = opts || {};
    const speed = clamp(Number(opts.speed) || 1, 0.25, 20);
    const autoplay = opts.autoplay || ''; // '' | 'once' | 'loop'
    let S;
    let listener = null;
    let lastWall = 0;
    let timer = null;

    function reset() {
      const tasks = {};
      OBJECTS.forEach((o) => {
        tasks['pick_' + o.id] = { id: 'pick_' + o.id, skill: 'pick', object: o.id, label: o.label, xyz: o.xyz, est: PICK_S, state: 'pending', arm: null, retries: 0 };
        tasks['place_' + o.id] = { id: 'place_' + o.id, skill: 'place', object: o.id, label: o.label, xyz: BOWL.xyz, est: PLACE_S, state: 'pending', arm: null, retries: 0, partner: 'pick_' + o.id };
      });
      const arms = {};
      CATALOG.forEach((c, i) => {
        const pa = panAxis(c);
        arms[c.arm_id] = {
          c, order: i, node_state: 'known', connected: false, joined: false, parking: false,
          run: null, held: null, joined_at: null, left_at: null, ok: 0, failed: 0, reason: null,
          history: [], failNext: false, startingUntil: 0, parkUntil: 0, unblocked: 0,
          q: PARK_Q.slice(), tcp: [pa[0] + 0.15 * Math.cos(c.yaw * D2R), pa[1] + 0.15 * Math.sin(c.yaw * D2R), 0.16],
          hb: 0.05, ik: null,
        };
      });
      S = { phase: 'estimate', paused: false, now: 0, story: 0, storyStep: 0, finishedAt: null, finishedStory: null,
        tasks, arms, bowlBy: null, feed: [], replan: null };
      // One arm is plugged in when the screen opens (as in the demo plan, step 1).
      joinNow('rakesh_follower', true);
      feed('Demo data: an in-browser stand-in for the orchestrator', 'info');
    }

    function feed(text, level) {
      S.feed.unshift({ t: S.now, text, level: level || 'info', running: S.phase === 'running' });
      S.feed = S.feed.slice(0, 40);
    }

    const joinedIds = () => Object.values(S.arms).filter((a) => a.joined).sort((x, y) => x.order - y.order).map((a) => a.c.arm_id);

    // ---------------------------------------------------------------- planning (estimate only)

    function plan(armIds, fresh) {
      const arms = armIds.map((id) => {
        const a = S.arms[id];
        let free = fresh ? 0 : S.now;
        let held = null;
        if (!fresh && a.run) free = Math.max(S.now, a.run.start + a.run.dur);
        if (!fresh && a.run && a.run.step.skill === 'pick') held = 'place_' + a.run.step.object;
        if (!fresh && a.held && !a.run) held = a.held;
        return { id, c: a.c, free, held };
      });
      let bowlFree = fresh ? 0 : S.now;
      if (!fresh && S.bowlBy) {
        const r = S.arms[S.tasks[S.bowlBy].arm].run;
        if (r) bowlFree = r.start + r.dur;
      }
      const taken = new Set();
      arms.forEach((a) => a.held && taken.add(S.tasks[a.held].object));
      const picks = OBJECTS.filter((o) => {
        if (fresh) return true;
        const p = S.tasks['pick_' + o.id];
        return p.state === 'pending' && !taken.has(o.id);
      });
      const blocked = picks.filter((o) => !arms.some((a) => reaches(a.c, o.xyz))).length;
      let open = picks.filter((o) => arms.some((a) => reaches(a.c, o.xyz)));
      let end = fresh ? 0 : S.now;
      const live = arms.slice();
      let bowlWork = 0;
      while (live.length) {
        live.sort((x, y) => x.free - y.free);
        const a = live[0];
        if (a.held) {
          const start = Math.max(a.free, bowlFree);
          a.free = bowlFree = start + PLACE_S;
          bowlWork += PLACE_S;
          a.held = null;
          end = Math.max(end, a.free);
          continue;
        }
        const here = [a.c.base[0], a.c.base[1], 0];
        const mine = open.filter((o) => reaches(a.c, o.xyz)).sort((x, y) => dist(here, x.xyz) - dist(here, y.xyz));
        if (!mine.length) {
          live.shift();
          continue;
        }
        open = open.filter((o) => o !== mine[0]);
        a.free += PICK_S;
        a.held = 'place_' + mine[0].id;
        end = Math.max(end, a.free);
      }
      return { finish: Math.max(end, bowlFree), blocked, bowlWork };
    }

    function limitText(p, nArms) {
      if (nArms === 0) return 'Plug in an arm to start.';
      if (p.blocked > 0) {
        return `${p.blocked} object${p.blocked === 1 ? ' is' : 's are'} out of reach of the arms here: plug in an arm that reaches ${p.blocked === 1 ? 'it' : 'them'}.`;
      }
      const left = p.finish - (S.phase === 'running' ? S.now : 0);
      if (p.bowlWork > 0 && left <= p.bowlWork * 1.15 + PICK_S + 0.5) {
        return "The bowl is the bottleneck: more arms won't help, a second bowl would.";
      }
      return 'More arms would make this faster.';
    }

    // ---------------------------------------------------------------- arms join and leave

    function joinNow(id, quiet) {
      const a = S.arms[id];
      const before = S.phase === 'running' ? plan(joinedIds(), false) : null;
      const others = joinedIds().map((x) => S.arms[x].c);
      a.unblocked = 0;
      if (others.length) {
        OBJECTS.forEach((o) => {
          const p = S.tasks['pick_' + o.id];
          if (p.state === 'pending' && !others.some((c) => reaches(c, o.xyz)) && reaches(a.c, o.xyz)) a.unblocked += 2;
        });
      }
      Object.assign(a, { node_state: 'ready', connected: true, joined: true, parking: false, joined_at: S.now, left_at: null, reason: null });
      a.hb = 0.05;
      if (!quiet) feed(`Arm ${id} joined${a.unblocked ? `: ${a.unblocked} more steps within reach` : ''}`, 'info');
      if (before) replanned(`Arm ${id} joined`, before, a.unblocked);
    }

    function replanned(what, before, unblocked) {
      const after = plan(joinedIds(), false);
      const fmt = O.clock;
      const newlyOut = after.blocked - before.blocked;
      S.replan = {
        head: newlyOut > 0
          ? `${what}: ${newlyOut} object${newlyOut === 1 ? '' : 's'} now out of reach, ${fmt(after.finish - S.now)} for the rest`
          : `${what}: ${fmt(before.finish - S.now)} left → ${fmt(after.finish - S.now)} left`,
        detail: `${unblocked ? `${unblocked} more steps now within reach. ` : ''}Plans compared: nearest free arm, critical work first.`,
        until: S.now + 4,
        startedAt: S.now,
      };
      feed(`Re-split: finish now ${fmt(after.finish)}${after.blocked ? ` for what can be reached (${after.blocked} out of reach)` : ''}`, after.blocked ? 'warn' : 'info');
    }

    function plugNext(id) {
      const target = id
        ? S.arms[id]
        : Object.values(S.arms).sort((x, y) => x.order - y.order).find((a) => !a.connected && !a.joined);
      if (!target) return { accepted: false, message: 'No arm left to plug in' };
      if (target.joined || target.node_state === 'starting') return { accepted: false, message: `${target.c.arm_id} is already in` };
      target.connected = true;
      target.node_state = 'starting';
      target.startingUntil = S.now + 1.4;
      feed(`Arm ${target.c.arm_id} plugged in: starting (USB serial ${target.c.serial})`, 'info');
      return { accepted: true, message: `Plugging in ${target.c.arm_id}` };
    }

    function backToPool(a, why) {
      if (a.run) {
        const step = a.run.step;
        step.state = 'pending';
        step.arm = null;
        if (step.skill === 'place') {
          S.tasks[step.partner].state = 'pending';
          S.tasks[step.partner].arm = null;
        }
        if (S.bowlBy === step.id) S.bowlBy = null;
        a.history.unshift({ task: step.id, outcome: 'moved' });
        a.run = null;
      }
      if (a.held) {
        const place = S.tasks[a.held];
        S.tasks[place.partner].state = 'pending';
        S.tasks[place.partner].arm = null;
        a.held = null;
        feed(`Part dropped by ${a.c.arm_id}: its pick is redone (${why})`, 'warn');
      }
    }

    function lose(id, why) {
      const a = S.arms[id];
      if (!a.joined) return { accepted: false, message: `${id} is not in the run` };
      const before = S.phase === 'running' ? plan(joinedIds(), false) : null;
      backToPool(a, why);
      Object.assign(a, { node_state: 'lost', joined: false, connected: false, parking: false, reason: why, left_at: S.now });
      feed(`Arm ${id} lost (${why}), moving its work`, 'bad');
      if (before) replanned(`Arm ${id} lost`, before, 0);
      return { accepted: true, message: `${id}: cable pulled` };
    }

    function remove(id, abort) {
      const a = S.arms[id];
      if (!a.joined) return { accepted: false, message: `${id} is not in the run` };
      if (a.parking) return { accepted: false, message: `${id} is already stopping` };
      a.parking = true;
      if (abort && a.run) {
        backToPool(a, 'aborted: arm removed');
        feed(`${id}: step aborted, parking`, 'warn');
      } else {
        feed(`${id}: slow stop requested (finishes its step${a.held || (a.run && a.run.step.skill === 'pick') ? ' and places the part' : ''})`, 'info');
      }
      return { accepted: true, message: `${id}: slow stop` };
    }

    function finishPark(a) {
      const before = S.phase === 'running' ? plan(joinedIds(), false) : null;
      Object.assign(a, { node_state: 'removed', joined: false, parking: false, reason: 'parked and removed', left_at: S.now });
      feed(`Arm ${a.c.arm_id} parked and removed`, 'info');
      if (before && joinedIds().length) replanned(`Arm ${a.c.arm_id} removed`, before, 0);
    }

    // ---------------------------------------------------------------- dispatch and steps

    function start(a, step) {
      step.state = 'running';
      step.arm = a.c.arm_id;
      if (step.skill === 'place') S.bowlBy = step.id;
      const dur = step.est * (0.9 + Math.random() * 0.2);
      a.run = { step, start: S.now, dur };
      a.node_state = 'busy';
      a.ik = { task: step.id, waypoints: waypoints(a.c, step) };
    }

    function dispatch() {
      const ready = Object.values(S.arms).filter((x) => x.joined && !x.run && x.node_state !== 'starting').sort((x, y) => x.order - y.order);
      for (const a of ready) {
        if (a.held) {
          if (!S.bowlBy) start(a, S.tasks[a.held]);
          else a.node_state = 'ready';
          continue;
        }
        if (a.parking) {
          if (!a.parkUntil) {
            a.parkUntil = S.now + 1.5;
            a.node_state = 'parking';
          } else if (S.now >= a.parkUntil) {
            a.parkUntil = 0;
            finishPark(a);
          }
          continue;
        }
        if (S.phase !== 'running' || S.paused) {
          a.node_state = 'ready';
          continue;
        }
        const here = [a.c.base[0], a.c.base[1], 0];
        const next = Object.values(S.tasks)
          .filter((t) => t.skill === 'pick' && t.state === 'pending' && reaches(a.c, t.xyz))
          .sort((x, y) => dist(here, x.xyz) - dist(here, y.xyz))[0];
        if (next) start(a, next);
        else a.node_state = 'ready';
      }
    }

    function finishStep(a) {
      const step = a.run.step;
      a.run = null;
      if (step.skill === 'place' && S.bowlBy === step.id) S.bowlBy = null;
      const fail = step.skill === 'pick' && a.failNext;
      if (fail) {
        a.failNext = false;
        a.failed += 1;
        step.retries += 1;
        a.history.unshift({ task: step.id, outcome: 'failed' });
        if (step.retries > MAX_RETRIES) {
          step.state = 'failed';
          S.tasks['place_' + step.object].state = 'failed';
          feed(`${step.id} failed on ${a.c.arm_id} (grasp): given up`, 'bad');
        } else {
          step.state = 'pending';
          step.arm = null;
          feed(`${step.id} failed on ${a.c.arm_id} (grasp), retrying`, 'bad');
        }
      } else {
        step.state = 'done';
        a.ok += 1;
        a.history.unshift({ task: step.id, outcome: 'ok' });
        if (step.skill === 'pick') a.held = 'place_' + step.object;
        else {
          a.held = null;
          feed(`${step.label} is in the bowl (${a.c.arm_id})`, 'ok');
        }
      }
      a.history = a.history.slice(0, 10);
      a.node_state = 'ready';
      if (Object.values(S.tasks).every((t) => t.state === 'done' || t.state === 'failed')) {
        S.phase = 'finished';
        S.finishedAt = S.now;
        S.finishedStory = S.story;
        feed(`Finished in ${O.clock(S.now)}`, 'ok');
      }
    }

    // ---------------------------------------------------------------- motion (display only)

    function animate(a, dt) {
      let qt = HOME_Q;
      let pt = null;
      if (a.node_state === 'lost') return;
      if (a.node_state === 'removed' || a.node_state === 'known' || a.parking || a.node_state === 'parking') {
        qt = PARK_Q;
      } else if (a.run) {
        const f = clamp((S.now - a.run.start) / a.run.dur, 0, 1);
        const s = a.run.step;
        if (s.skill === 'pick') {
          const grip = gripRad(f < 0.5 ? 70 : 22);
          pt = f < 0.35 ? add(s.xyz, 0.08) : f < 0.6 ? add(s.xyz, 0.0) : add(s.xyz, 0.1);
          qt = poseToward(a.c, pt, grip);
        } else {
          const grip = gripRad(f < 0.75 ? 22 : 70);
          pt = f < 0.5 ? add(BOWL.xyz, 0.08) : f < 0.8 ? BOWL.xyz : add(BOWL.xyz, 0.08);
          qt = poseToward(a.c, pt, grip);
        }
      } else if (a.held) {
        pt = add(S.tasks[S.tasks[a.held].partner].xyz, 0.1);
        qt = poseToward(a.c, pt, gripRad(22));
      }
      const k = Math.min(1, dt * 5);
      a.q = a.q.map((v, i) => v + (qt[i] - v) * k);
      if (pt) a.tcp = a.tcp.map((v, i) => v + (pt[i] - v) * k);
      a.target = pt;
      a.hb = clamp(a.hb + (Math.random() - 0.5) * 0.03, 0.03, 0.16);
    }

    // ---------------------------------------------------------------- autoplay story

    const STORY = [
      [1.5, () => command('go')],
      [8, () => plugNext('vato_follower')],
      [21, () => plugNext('third_follower')],
      [27, () => { S.arms.third_follower.failNext = true; }],
    ];

    function story() {
      if (!autoplay) return;
      while (S.storyStep < STORY.length && S.story >= STORY[S.storyStep][0]) {
        STORY[S.storyStep][1]();
        S.storyStep += 1;
      }
      if (autoplay === 'loop' && S.phase === 'finished' && S.story - S.finishedStory > 10) reset();
    }

    // ---------------------------------------------------------------- tick and model

    function tick() {
      const wall = performance.now();
      const dt = Math.min(0.5, (wall - lastWall) / 1000) * speed;
      lastWall = wall;
      S.story += dt;
      if (S.phase === 'running' && !S.paused) S.now += dt;
      else if (S.phase === 'running' && S.paused) S.now += dt; // the clock runs on while paused
      story();
      Object.values(S.arms).forEach((a) => {
        if (a.node_state === 'starting' && S.now >= a.startingUntil) joinNow(a.c.arm_id);
        if (a.run && S.now >= a.run.start + a.run.dur) finishStep(a);
      });
      // Before Go nothing advances S.now: let starting arms finish on story time.
      if (S.phase !== 'running') {
        Object.values(S.arms).forEach((a) => {
          if (a.node_state === 'starting') {
            a.startingUntil -= dt;
            if (a.startingUntil <= S.now) joinNow(a.c.arm_id);
          }
          if (a.parkUntil) a.parkUntil -= dt;
        });
      }
      dispatch();
      Object.values(S.arms).forEach((a) => animate(a, dt));
      if (S.replan && S.now > S.replan.until) S.replan = null;
      if (listener) listener(model());
    }

    function model() {
      const joined = joinedIds();
      const p = plan(joined, S.phase !== 'running');
      const whatif = {};
      for (let k = 1; k <= CATALOG.length; k++) {
        const w = plan(CATALOG.slice(0, k).map((c) => c.arm_id), true);
        whatif[k] = { finish: w.finish, blocked: w.blocked };
      }
      const tasks = Object.values(S.tasks);
      const done = tasks.filter((t) => t.state === 'done').length;
      let remaining = null;
      let big = null;
      if (S.phase === 'running') remaining = Math.max(0, p.finish - S.now);
      else if (S.phase === 'finished') big = S.finishedAt;
      else big = joined.length ? p.finish : null;
      const objects = OBJECTS.map((o) => {
        const pick = S.tasks['pick_' + o.id];
        const place = S.tasks['place_' + o.id];
        let state = 'waiting';
        let arm = null;
        if (place.state === 'done') { state = 'done'; arm = place.arm; }
        else if (place.state === 'failed' || pick.state === 'failed') state = 'failed';
        else if (place.state === 'running') { state = 'placing'; arm = place.arm; }
        else if (pick.state === 'running') { state = 'picking'; arm = pick.arm; }
        else if (pick.state === 'done') { state = 'held'; arm = pick.arm; }
        else if (!joined.some((id) => reaches(S.arms[id].c, o.xyz))) state = 'out_of_reach';
        return { id: o.id, label: o.label, state, arm };
      });
      const nodes = Object.values(S.arms).sort((x, y) => x.order - y.order).map((a) => {
        const run = a.run;
        let task = null;
        if (run) {
          task = { id: run.step.id, skill: run.step.skill, label: run.step.label, target: run.step.skill === 'place' ? 'bowl' : null, elapsed: S.now - run.start, estimate: run.step.est };
        }
        const live = a.joined || a.node_state === 'starting';
        return {
          arm_id: a.c.arm_id, label: a.c.label, color: a.c.color, pair: a.c.pair, serial: a.c.serial,
          node_state: a.joined && a.parking ? 'parking' : a.node_state, backend: 'demo', connected: a.connected, joined: a.joined, parking: a.parking,
          task, held: a.held ? { task: a.held, label: S.tasks[a.held].label } : null,
          waiting_for: a.held && !run && S.bowlBy ? 'bowl' : null,
          heartbeat_age_s: live ? a.hb : null, joined_at: a.joined_at, left_at: a.left_at,
          steps_ok: a.ok, steps_failed: a.failed, last_reason: a.reason, unblocked: a.unblocked,
          history: a.history.slice(), fail_armed: a.failNext,
          joints: a.node_state === 'known' ? null : { q: a.q.slice(), stale: a.node_state === 'lost' || a.node_state === 'removed' },
          tcp: a.tcp.slice(), target: a.target || null,
          ik: run ? a.ik : null,
          reach: { r_min: REACH.r_min, r_max: REACH.r_max },
        };
      });
      const counts = {
        known: nodes.length,
        connected: nodes.filter((n) => n.connected).length,
        joined: joined.length,
      };
      return {
        source: 'demo', connected: true, updated: Date.now(), backend: 'demo',
        task: {
          name: 'hacklab6', title: 'Sorting objects on the desk',
          subtitle: `${OBJECTS.length} objects into the shared bowl`,
          objects, done, total: tasks.length,
          blocked: objects.filter((o) => o.state === 'out_of_reach').length,
        },
        run: {
          phase: S.phase, paused: S.paused, now: S.now, remaining, big,
          limit: S.phase === 'finished' ? '' : limitText(p, joined.length), whatif,
        },
        replanning: S.replan ? { head: S.phase === 'running' && S.now - S.replan.startedAt < 1.2 ? S.replan.head.split(':')[0] + ': re-planning…' : S.replan.head, detail: S.replan.detail } : null,
        counts, nodes, feed: S.feed.slice(0, 12),
        allowed: { go: S.phase === 'estimate' && joined.length > 0, pause: S.phase === 'running', stop_all: S.phase !== 'finished' && joined.length > 0, plug: nodes.some((n) => !n.connected && !n.joined), simulate_unplug: true, inject_failure: true },
      };
    }

    function command(name, args) {
      args = args || {};
      let out = { accepted: true, message: '' };
      if (name === 'go') {
        if (S.phase !== 'estimate' || !joinedIds().length) out = { accepted: false, message: 'Go needs a ready run and one arm' };
        else { S.phase = 'running'; S.now = 0; feed('Go', 'info'); out.message = 'Started'; }
      } else if (name === 'pause') {
        if (S.phase !== 'running') out = { accepted: false, message: 'Not running' };
        else { S.paused = !S.paused; feed(S.paused ? 'Paused: no new steps start; running steps finish' : 'Resumed', 'info'); out.message = S.paused ? 'Paused' : 'Resumed'; }
      } else if (name === 'stop_all') {
        joinedIds().forEach((id) => remove(id, !!args.abort));
        out.message = 'Slow stop for every arm';
      } else if (name === 'plug') out = plugNext(args.arm);
      else if (name === 'remove') out = remove(args.arm, !!args.abort);
      else if (name === 'simulate_unplug') out = lose(args.arm, 'unplugged');
      else if (name === 'inject_failure') {
        const a = S.arms[args.arm];
        if (!a || !a.joined) out = { accepted: false, message: 'Arm not in the run' };
        else { a.failNext = true; out.message = `${args.arm}: next pick will fail (grasp)`; }
      } else if (name === 'reset') { reset(); out.message = 'Demo reset'; }
      else out = { accepted: false, message: `Unknown command ${name}` };
      if (listener) listener(model());
      return Promise.resolve(out);
    }

    reset();
    return {
      kind: 'demo',
      start(onModel) {
        listener = onModel;
        lastWall = performance.now();
        timer = setInterval(tick, 100);
        tick();
      },
      stop() { clearInterval(timer); },
      command,
    };
  }

  O.DemoSource = DemoSource;
})();
