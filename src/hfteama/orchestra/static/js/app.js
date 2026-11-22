/* Orchestra screen: 30 % robots and status, 70 % the task on top and one card per robot that has
 * joined it. Renders one screen model (from js/live.js or js/demo.js); never talks to arms itself.
 *
 * URL options:  ?demo=1  demo data even if the API answers   ?autoplay=once|loop  scripted story
 *               ?speed=2 demo clock speed-up                  ?api=http://host:8050  API base URL
 *               ?token=... operator token for commands (spec section 5.3)
 */
(function () {
  'use strict';
  const O = (window.Orchestra = window.Orchestra || {});
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);

  // ------------------------------------------------------------------ helpers

  O.clock = function (s) {
    if (s == null || !isFinite(s)) return '--:--';
    s = Math.max(0, Math.round(s));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  };
  const clock = O.clock;
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const deg = (r) => (r * 180) / Math.PI;
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
  const nm = (id) => O.armName(id); // display name: Arm Node A / B (ids stay as configured)
  const rn = (s) => O.rename(s);

  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) {
      /* private window or blocked storage: the screen works without it */
    }
    return null;
  }

  O.token = function () {
    const t = params.get('token');
    if (t) store('orchestra-token', t);
    return t || store('orchestra-token') || '';
  };

  // node_state -> word, tone. Tones: idle, neutral, active, warn, bad, ok (see CSS).
  const STATES = {
    known: ['Not plugged in', 'idle'],
    connected: ['Connected', 'neutral'],
    starting: ['Starting', 'active'],
    ready: ['Ready', 'neutral'],
    busy: ['Working', 'active'],
    parking: ['Stopping', 'warn'],
    removed: ['Removed', 'idle'],
    lost: ['Lost', 'bad'],
    error: ['Error', 'bad'],
  };
  const stateOf = (n) => STATES[n.node_state] || [n.node_state, 'neutral'];
  const VERB = { pick: 'Picking', place: 'Placing', pick_place: 'Moving', inspect: 'Inspecting', park: 'Parking' };
  const JOINTS = ['Pan', 'Lift', 'Elbow', 'Wrist flex', 'Wrist roll', 'Gripper'];
  const HIST_S = 20; // seconds of joint history in each graph

  // ------------------------------------------------------------------ state

  let source = null;
  let M = null;
  let selected = store('orchestra-selected') || null;
  const rows = new Map();
  const cards = new Map();
  const joinOrder = new Map();
  const hist = new Map(); // arm id -> [{t, q}]
  let joinSeq = 0;
  let lastJoined = null;
  let busyCmd = false;

  function setHTML(el, html) {
    if (el._html !== html) {
      el.innerHTML = html;
      el._html = html;
    }
  }
  function setText(el, text) {
    if (el.textContent !== text) el.textContent = text;
  }

  // ------------------------------------------------------------------ left pane

  function renderLeft(m) {
    const joined = $('c-joined');
    if (lastJoined !== null && m.counts.joined !== lastJoined) {
      joined.classList.remove('bump');
      void joined.offsetWidth;
      joined.classList.add('bump');
    }
    lastJoined = m.counts.joined;
    setText(joined, String(m.counts.joined));
    setText($('c-known'), String(m.counts.known));
    setText($('c-label'), m.counts.known === 1 ? 'arm node joined' : 'arm nodes joined');
    setHTML($('fleet-bar'), m.nodes.map((n) => `<i data-tone="${stateOf(n)[1]}" title="${esc(nm(n.arm_id))}: ${esc(stateOf(n)[0])}"></i>`).join(''));

    const list = $('robot-list');
    m.nodes.forEach((n) => {
      let li = rows.get(n.arm_id);
      if (!li) {
        li = document.createElement('li');
        li.innerHTML = '<button class="robot" type="button"><span class="r-glyph" aria-hidden="true"></span><span class="r-id"></span><span class="st"></span><span class="r-line"></span></button>';
        const view = O.ArmView({ mini: true });
        li.querySelector('.r-glyph').appendChild(view.el);
        li._arm = view;
        li.firstElementChild.addEventListener('click', () => select(n.arm_id));
        rows.set(n.arm_id, li);
        list.appendChild(li);
      }
      const b = li.firstElementChild;
      const [word, tone] = stateOf(n);
      b.dataset.tone = tone;
      b.classList.toggle('sel', selected === n.arm_id);
      b.querySelector('.r-glyph').style.setProperty('--sw', n.color);
      li._arm.paint(n.arm_id, n.color);
      li._arm.update(n.joints && n.joints.q);
      setHTML(b.querySelector('.r-id'), nm(n.arm_id) === n.arm_id ? esc(n.arm_id) : `${esc(nm(n.arm_id))}<small>${esc(n.arm_id)}</small>`);
      const st = b.querySelector('.st');
      st.dataset.tone = tone;
      setText(st, word);
      setText(b.querySelector('.r-line'), rowLine(n));
      b.title = `${O.armFinish(n.arm_id)}${n.label ? ` · ${n.label}` : ''}`;
    });
  }

  function rowLine(n) {
    if (n.node_state === 'known') return 'Waiting for its USB cable';
    if (n.node_state === 'starting') return 'Checks and homing';
    if (n.node_state === 'lost') return n.last_reason || 'No answer';
    if (n.node_state === 'removed') return 'Parked';
    if (n.node_state === 'parking') return 'Finishing, then parking';
    if (n.task) return `${VERB[n.task.skill] || n.task.skill} ${n.task.label}`;
    if (n.held) return `Holding ${n.held.label}`;
    if (M && M.run.phase === 'estimate') return 'Ready for Go';
    return 'Idle';
  }

  // ------------------------------------------------------------------ task bar

  function renderTask(m) {
    setText($('task-title'), m.task.title);
    setText($('task-sub'), m.task.subtitle);
    document.title = `${m.task.title} · Orchestra`;

    const r = m.run;
    const n = m.counts.joined;
    let phase = ['', 'idle'];
    if (r.phase === 'estimate') phase = n ? ['Ready', 'neutral'] : ['No arms', 'idle'];
    if (r.phase === 'running') phase = r.paused ? ['Paused', 'warn'] : ['Running', 'active'];
    if (r.phase === 'finished') phase = ['Done', 'ok'];
    setText($('phase'), phase[0]);
    $('phase').dataset.tone = phase[1];

    if (r.phase === 'running') {
      setText($('big'), clock(r.remaining));
      setText($('big-label'), `left · ${plural(n, 'arm')}`);
    } else if (r.phase === 'finished') {
      setText($('big'), clock(r.big));
      setText($('big-label'), 'total time');
    } else {
      setText($('big'), clock(r.big));
      setText($('big-label'), n ? `estimate · ${plural(n, 'arm')}` : 'plug in an arm');
    }

    const objs = m.task.objects || [];
    const pct = m.task.total ? (100 * m.task.done) / m.task.total : 0;
    $('prog-bar').style.width = `${pct}%`;
    $('prog-bar').classList.toggle('done', r.phase === 'finished');
    let prog = `${m.task.done} of ${m.task.total} steps`;
    if (objs.length) {
      const done = objs.filter((o) => o.state === 'done').length;
      const out = objs.filter((o) => o.state === 'out_of_reach').length;
      prog = `${done} of ${plural(objs.length, 'object')} sorted${out ? ` · ${out} out of reach` : ''}`;
    }
    setText($('prog-text'), prog);

    const a = m.allowed || {};
    const off = stale(m) || busyCmd;
    $('btn-go').disabled = off || !a.go;
    $('btn-go').hidden = r.phase !== 'estimate';
    $('btn-pause').disabled = off || !a.pause;
    $('btn-pause').hidden = r.phase !== 'running';
    setText($('btn-pause'), r.paused ? 'Resume' : 'Pause');
    $('btn-stop').disabled = off || !a.stop_all;
    $('btn-stop').hidden = r.phase === 'finished';

    const rp = $('replan');
    rp.hidden = !m.replanning;
    if (m.replanning) {
      setText($('replan-head'), rn(m.replanning.head));
      setText($('replan-detail'), rn(m.replanning.detail || ''));
    }
    const limit = [r.limit, r.unblocked_note].filter(Boolean).join(' ');
    setText($('limit'), r.phase === 'finished' ? '' : rn(limit));
    $('real-banner').hidden = m.backend !== 'real';
  }

  // ------------------------------------------------------------------ figures strip

  function renderKpis(m) {
    const t = m.task;
    const pct = t.total ? Math.round((100 * t.done) / t.total) : 0;
    setHTML($('k-steps'), `${t.done}<small>/${t.total}</small>`);
    setText($('k-steps-sub'), `${pct}% of the batch`);
    $('k-steps-bar').style.width = `${pct}%`;

    const objs = t.objects || [];
    if (objs.length) {
      const done = objs.filter((o) => o.state === 'done').length;
      const out = objs.filter((o) => o.state === 'out_of_reach').length;
      const moving = objs.filter((o) => o.state === 'picking' || o.state === 'placing' || o.state === 'held').length;
      setText($('k-obj-h'), 'Objects sorted');
      setHTML($('k-obj'), `${done}<small>/${objs.length}</small>`);
      setText($('k-obj-sub'), out ? `${out} out of reach` : moving ? `${moving} in hand` : 'all within reach');
      $('k-obj').parentElement.dataset.tone = out ? 'warn' : done === objs.length ? 'ok' : '';
    } else {
      setText($('k-obj-h'), 'Steps left');
      setText($('k-obj'), String(Math.max(0, t.total - t.done)));
      setText($('k-obj-sub'), t.blocked ? `${t.blocked} out of reach` : '');
      $('k-obj').parentElement.dataset.tone = t.blocked ? 'warn' : '';
    }

    const failed = m.nodes.reduce((s, n) => s + (n.steps_failed || 0), 0);
    setText($('k-fail'), String(failed));
    const worst = m.nodes.filter((n) => n.steps_failed).sort((a, b) => b.steps_failed - a.steps_failed)[0];
    setText($('k-fail-sub'), failed ? `most on ${nm(worst.arm_id)} · retried` : 'none so far');
    $('k-fail-box').dataset.tone = failed ? 'bad' : '';

    const live = m.nodes.filter((n) => n.joined && n.heartbeat_age_s != null);
    const slow = live.sort((a, b) => b.heartbeat_age_s - a.heartbeat_age_s)[0];
    setHTML($('k-hb'), slow ? `${Math.round(slow.heartbeat_age_s * 1000)}<small>ms</small>` : '—');
    setText($('k-hb-sub'), slow ? `${nm(slow.arm_id)} · dropped after 1.5 s` : 'no arm node answering');
    $('k-hb-box').dataset.tone = slow && slow.heartbeat_age_s > 0.75 ? 'bad' : '';

    const on = m.nodes.filter((n) => n.joined).length;
    setText($('nodes-sub'), `${on} on this task · ${m.nodes.length} known`);
  }

  // ------------------------------------------------------------------ robot cards

  function cardSkeleton() {
    const el = document.createElement('article');
    el.className = 'card enter';
    el.innerHTML = `
      <header class="c-head"><span class="sw" aria-hidden="true"></span><h3></h3><span class="st"></span><div class="c-conn"></div></header>
      <div class="c-stage">
        <div class="stage-tl"><span class="pan-slot"></span><span><small>Pan</small><b class="pan-v">—</b></span></div>
        <div class="stage-tr"><span class="tag"></span></div>
        <div class="stage-bl"><span><small>Gripper</small><b class="grip-v">—</b></span></div>
        <div class="stage-br"></div>
        <div class="stage-note" hidden></div>
      </div>
      <div class="c-now"></div>
      <div class="c-ik live-only"></div>
      <div class="c-joints live-only">
        ${JOINTS.map((j, i) => `<div class="jg" data-i="${i}"><div class="jl"><span>${j}</span><b>—</b></div><canvas aria-label="${j}, last ${HIST_S} seconds"></canvas></div>`).join('')}
        <div class="no-joints" hidden></div>
      </div>
      <footer class="c-foot"><span class="stats"></span><span class="actions"></span></footer>`;
    const stage = el.querySelector('.c-stage');
    el._arm = O.ArmView({ label: 'Live side view of the SO-101' });
    stage.insertBefore(el._arm.el, stage.firstChild);
    el._pan = O.panDial();
    el.querySelector('.pan-slot').appendChild(el._pan.el);
    el.addEventListener('animationend', () => el.classList.remove('enter', 'flash'));
    return el;
  }

  // The arm drawn from its latest joint angles; parked (rest pose, dimmed) when there are none.
  function renderStage(c, n, m, gone) {
    const stage = c.querySelector('.c-stage');
    const j = n.joints;
    const q = j && j.q && j.q.length >= 5 ? j.q : null;
    c._arm.paint(n.arm_id, n.color);
    c._arm.update(q);
    c._pan.update(q ? q[0] : 0);
    stage.classList.toggle('stale', gone || !!(j && j.stale) || n.node_state === 'lost');
    stage.classList.toggle('nodata', !q);
    const note = stage.querySelector('.stage-note');
    note.hidden = !!q;
    if (!q) setText(note, n.backend === 'fake' ? 'Fake arm: no joints, rest pose shown' : 'No joint data yet: rest pose shown');
    setText(stage.querySelector('.pan-v'), q ? `${deg(q[0]).toFixed(0)}°` : '—');
    const [lo, hi] = O.LIMITS[5];
    setText(stage.querySelector('.grip-v'), q && q.length > 5 ? `${Math.round((100 * (q[5] - lo)) / (hi - lo))}% open` : '—');
    const tag = stage.querySelector('.tag');
    const be = m.source === 'demo' ? 'demo' : n.backend || '';
    setText(tag, be === 'real' || be === 'sorter' ? 'Real SO-101' : be ? `${be} · SO-101` : 'SO-101');
    tag.classList.toggle('real', be === 'real' || be === 'sorter');
    const hb = n.heartbeat_age_s != null && !gone
      ? `<span title="Time since the arm last answered; dropped after 1.5 s">♥ <b class="${n.heartbeat_age_s > 0.75 ? 'bad' : ''}">${Math.round(n.heartbeat_age_s * 1000)} ms</b></span>`
      : '';
    setHTML(stage.querySelector('.stage-br'), hb);
  }

  // How the arm is attached, in one line (see docs/ORCHESTRA_DASHBOARD.md, "How a robot connects").
  function connLine(n, m) {
    const usb = n.serial ? `USB ${n.serial}` : 'USB';
    if (!n.connected && !n.joined) return `${usb} · not seen`;
    const where = n.port ? ` on ${n.port}` : '';
    const how = m.source === 'demo'
      ? 'simulated'
      : n.backend === 'real'
        ? 'Feetech bus via LeRobot'
        : n.backend === 'fake'
          ? 'fake: USB cable only, servos and 12 V not checked'
          : n.backend;
    return `${usb}${where} · ${how}`;
  }

  function nowBlock(n, m) {
    const t = n.task;
    if (n.node_state === 'lost') {
      return `<div class="v bad">Lost: ${esc(n.last_reason || 'no answer')}</div><div class="meta">${n.left_at != null ? `At ${clock(n.left_at)}. ` : ''}Its work moved to the other arms.</div>`;
    }
    if (n.node_state === 'removed') return `<div class="v">Parked and removed</div><div class="meta">${n.left_at != null ? `At ${clock(n.left_at)}.` : ''}</div>`;
    if (n.node_state === 'starting') return '<div class="v">Joining…</div><div class="meta">Safety checks and homing</div>';
    if (n.node_state === 'parking') return `<div class="v">Stopping</div><div class="meta">${t ? `Finishing ${esc(t.label)}, then parking` : 'Moving to the park pose, then torque off'}</div>`;
    if (t) {
      const f = t.elapsed != null && t.estimate ? Math.min(1, t.elapsed / t.estimate) : 0;
      const slow = t.elapsed != null && t.estimate && t.elapsed > t.estimate * 1.5;
      return `<div class="v">${esc(VERB[t.skill] || t.skill)} ${esc(t.label)}${t.target ? ` <span class="tgt">→ ${esc(t.target)}</span>` : ''}</div>
        <div class="step-bar"><i class="${slow ? 'slow' : ''}" style="width:${(f * 100).toFixed(1)}%"></i></div>`;
    }
    if (n.held) return `<div class="v">Holding ${esc(n.held.label)}</div><div class="meta">${n.waiting_for ? `Waiting for the ${esc(n.waiting_for)}` : 'Places it next'}</div>`;
    if (m.run.phase === 'estimate') return '<div class="v">Ready</div><div class="meta">Waiting for Go</div>';
    if (m.run.paused) return '<div class="v">Paused</div>';
    if (m.run.phase === 'finished') return '<div class="v">Done</div>';
    return '<div class="v">Idle</div><div class="meta">No step it can reach right now</div>';
  }

  function ikLine(n) {
    const w = n.ik && n.ik.waypoints;
    if (!w || !w.length) return '';
    const bad = w.filter((p) => !p.ok);
    const last = w[w.length - 1];
    const target = n.target || last.xyz;
    let away = '';
    if (n.tcp && target) {
      const d = Math.hypot(n.tcp[0] - target[0], n.tcp[1] - target[1], n.tcp[2] - target[2]);
      away = ` · <b>${(d * 100).toFixed(1)} cm</b> to target`;
    }
    if (bad.length) return `IK <span class="bad">✕ ${esc(bad.map((p) => p.name).join(', '))} unreachable</span>${away}`;
    const maxErr = Math.max(...w.map((p) => Number(p.error_mm) || 0));
    return `IK <span class="ok">✓</span> ${w.length}/${w.length} waypoints · <b>${maxErr.toFixed(1)} mm</b>${away}`;
  }

  function renderCards(m) {
    const box = $('cards');
    const present = new Set();
    m.nodes.forEach((n) => {
      const inTask = n.joined || n.node_state === 'starting' || n.node_state === 'parking';
      if (inTask && !joinOrder.has(n.arm_id)) joinOrder.set(n.arm_id, ++joinSeq);
      if (!joinOrder.has(n.arm_id)) return; // never joined: listed on the left only
      present.add(n.arm_id);
      let c = cards.get(n.arm_id);
      if (!c) {
        c = cardSkeleton();
        cards.set(n.arm_id, c);
      }
      const gone = !inTask;
      if (c.dataset.gone === '1' && !gone) {
        void c.offsetWidth;
        c.classList.add('enter');
      }
      c.dataset.gone = gone ? '1' : '0';
      c.classList.toggle('gone', gone);
      const [word, tone] = stateOf(n);
      c.dataset.tone = tone;
      c.querySelector('.sw').style.background = n.color;
      setHTML(c.querySelector('h3'), nm(n.arm_id) === n.arm_id ? esc(n.arm_id) : `${esc(nm(n.arm_id))}<small>${esc(n.arm_id)}</small>`);
      const st = c.querySelector('.st');
      st.dataset.tone = tone;
      setText(st, word);
      setText(c.querySelector('.c-conn'), `${O.armFinish(n.arm_id)} · ${connLine(n, m)}`);
      c.querySelector('.c-conn').title = `${n.label} · pair ${n.pair || '—'}`;
      renderStage(c, n, m, gone);
      setHTML(c.querySelector('.c-now'), nowBlock(n, m));
      const ik = ikLine(n);
      c.querySelector('.c-ik').hidden = !ik;
      setHTML(c.querySelector('.c-ik'), ik);
      recordJoints(n);
      renderJoints(c, n, m);
      setHTML(c.querySelector('.stats'), `${n.steps_ok} done${n.steps_failed ? ` · <span class="bad">${n.steps_failed} failed</span>` : ''}${n.label ? ` · ${esc(n.label)}` : ''}`);
      renderActions(c, n, m, gone);
    });

    const order = [...present].sort((x, y) => {
      const gx = cards.get(x).dataset.gone === '1';
      const gy = cards.get(y).dataset.gone === '1';
      if (gx !== gy) return gx ? 1 : -1;
      return joinOrder.get(x) - joinOrder.get(y);
    });
    order.forEach((id, i) => {
      const c = cards.get(id);
      if (box.children[i] !== c) box.insertBefore(c, box.children[i] || null);
    });

    let slot = box.querySelector('.slot');
    const waiting = m.nodes.filter((n) => !n.connected && !n.joined);
    if (waiting.length && m.run.phase !== 'finished') {
      if (!slot) {
        slot = document.createElement('div');
        slot.className = 'slot';
        slot.innerHTML = '<span></span><button class="btn small" type="button">Plug in next arm node</button>';
        slot.insertBefore(O.ArmView({ mini: true }).el, slot.firstChild);
        slot.querySelector('button').addEventListener('click', plugNext);
      }
      setText(
        slot.querySelector('span'),
        m.watching
          ? `${plural(waiting.length, 'more arm node')} can join: plug in ${waiting.length === 1 ? 'its' : 'their'} USB cable (${waiting.map((n) => nm(n.arm_id)).join(', ')})`
          : `${plural(waiting.length, 'more arm node')} can join this task (${waiting.map((n) => nm(n.arm_id)).join(', ')})`,
      );
      slot.querySelector('button').hidden = !!m.watching;
      slot.querySelector('button').disabled = stale(m) || busyCmd || !(m.allowed || {}).plug;
      box.appendChild(slot);
    } else if (slot) slot.remove();
  }

  // ------------------------------------------------------------------ live joint graphs

  function recordJoints(n) {
    if (!n.joints || n.joints.stale) return;
    const t = performance.now() / 1000;
    let h = hist.get(n.arm_id);
    if (!h) hist.set(n.arm_id, (h = []));
    if (h.length && t - h[h.length - 1].t < 0.05) return;
    h.push({ t, q: n.joints.q.slice() });
    while (h.length && t - h[0].t > HIST_S + 1) h.shift();
  }

  let colors = null;
  function readColors() {
    const s = getComputedStyle(document.documentElement);
    const v = (k) => s.getPropertyValue(k).trim();
    colors = { line: v('--active'), warn: v('--warn'), faint: v('--faint'), grid: v('--line'), stale: v('--faint') };
  }

  function renderJoints(c, n, m) {
    const box = c.querySelector('.c-joints');
    const none = box.querySelector('.no-joints');
    const h = hist.get(n.arm_id) || [];
    const has = h.length > 0;
    none.hidden = has;
    if (!has) {
      setText(none, n.backend === 'fake' ? 'Fake arms have no joints. Run with --backend sim to see them.' : 'No joint data yet.');
    }
    const stale = !n.joints || n.joints.stale;
    box.querySelectorAll('.jg').forEach((g, i) => {
      g.hidden = !has;
      if (!has) return;
      const [lo, hi] = O.LIMITS[i];
      const q = h[h.length - 1].q[i];
      const near = i < 5 && deg(Math.min(q - lo, hi - q)) < 5;
      g.classList.toggle('near', near && !stale);
      setText(g.querySelector('b'), i === 5 ? `${Math.round((100 * (q - lo)) / (hi - lo))}%` : `${deg(q).toFixed(0)}°`);
      drawGraph(g.querySelector('canvas'), h, i, lo, hi, stale ? colors.stale : near ? colors.warn : colors.line);
    });
  }

  // One joint over the last HIST_S seconds. y is scaled to what the joint did in the window (at
  // least 20° tall so a still joint draws flat), clipped to the joint's limits; the gripper uses
  // its full range. The dashed line is the joint's zero.
  function drawGraph(cv, h, i, lo, hi, color) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth;
    const ht = cv.clientHeight;
    if (!w || !ht) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(ht * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(ht * dpr);
    }
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, ht);
    const now = performance.now() / 1000;
    let ymin = lo;
    let ymax = hi;
    if (i < 5) {
      let a = Infinity;
      let b = -Infinity;
      h.forEach((p) => { a = Math.min(a, p.q[i]); b = Math.max(b, p.q[i]); });
      const span = Math.max(b - a, 0.35);
      const mid = (a + b) / 2;
      ymin = Math.max(lo, mid - span * 0.6);
      ymax = Math.min(hi, mid + span * 0.6);
    }
    const X = (t) => w * (1 - (now - t) / HIST_S);
    const Y = (v) => 2 + (ht - 4) * (1 - (v - ymin) / (ymax - ymin || 1));
    if (i < 5 && ymin < 0 && ymax > 0) {
      ctx.strokeStyle = colors.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.moveTo(0, Math.round(Y(0)) + 0.5);
      ctx.lineTo(w, Math.round(Y(0)) + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.beginPath();
    h.forEach((p, k) => {
      const x = X(p.t);
      const y = Y(p.q[i]);
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const last = h[h.length - 1];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(Math.min(X(last.t), w - 2.5), Y(last.q[i]), 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  function renderActions(c, n, m, gone) {
    const a = m.allowed || {};
    const key = [gone, n.parking, n.node_state, a.simulate_unplug, a.inject_failure, stale(m), busyCmd].join('|');
    const box = c.querySelector('.actions');
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    const dis = stale(m) || busyCmd ? ' disabled' : '';
    if (gone) {
      box.innerHTML = `<button class="btn small" data-cmd="plug"${dis}>Plug in again</button>`;
    } else {
      const canStop = n.joined && !n.parking;
      box.innerHTML = `<button class="btn small" data-cmd="remove"${canStop ? dis : ' disabled'}>Remove</button>
        <details class="menu"><summary class="btn small" aria-label="More actions for ${esc(n.arm_id)}">⋯</summary><div class="pop">
          <button data-cmd="remove-abort"${canStop ? dis : ' disabled'}>Remove now (abort its step)</button>
          <button data-cmd="simulate_unplug"${a.simulate_unplug && n.joined ? dis : ' disabled'}>Simulate unplug</button>
          <button data-cmd="inject_failure"${a.inject_failure && n.joined ? dis : ' disabled'}>Fail its next pick</button>
        </div></details>`;
    }
    box.querySelectorAll('[data-cmd]').forEach((b) => b.addEventListener('click', () => {
      const d = b.closest('details');
      if (d) d.open = false;
      armCommand(b.dataset.cmd, n.arm_id);
    }));
  }

  // ------------------------------------------------------------------ feed, connection

  function renderFeed(m) {
    setHTML($('feed'), (m.feed || []).slice(0, 8).map((f) => `<li data-level="${esc(f.level)}"><span class="t">${f.t != null && f.running !== false ? clock(f.t) : ''}</span><span class="x">${esc(rn(f.text))}</span></li>`).join(''));
  }

  function stale(m) {
    return m.source === 'live' && (!m.connected || Date.now() - m.updated > 3000);
  }

  function renderConn(m) {
    const s = stale(m);
    const c = $('conn');
    c.className = 'conn ' + (s ? 'down' : m.source === 'live' ? 'live' : '');
    setText(c, m.source === 'demo' ? 'Demo' : s ? 'Offline' : 'Live');
    $('app').classList.toggle('stale', s);
    const lb = $('lost-banner');
    lb.hidden = !s;
    if (s) setText(lb, `Connection to the orchestrator lost, reconnecting… Last update ${m.updated ? Math.round((Date.now() - m.updated) / 1000) + ' s ago' : 'never'}.`);
    setText($('source-note'), m.source === 'demo' ? `Demo data, not the orchestrator.${O.demoReason ? ' ' + O.demoReason : ''}` : '');
    $('demo-tools').hidden = m.source !== 'demo';
  }

  function render(m) {
    M = m;
    if (!colors) readColors();
    // the node's own colour (black A, green B), whatever the source sent
    m.nodes.forEach((n) => { n.color = O.armColor(n.arm_id, n.color); });
    renderConn(m);
    renderLeft(m);
    renderTask(m);
    renderKpis(m);
    O.renderSchedule(m);
    renderCards(m);
    renderFeed(m);
  }

  // ------------------------------------------------------------------ selection

  function select(id) {
    selected = id;
    store('orchestra-selected', id);
    if (M) renderLeft(M);
    const c = cards.get(id);
    if (c && c.isConnected) {
      c.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      c.classList.remove('flash');
      void c.offsetWidth;
      c.classList.add('flash');
      say('');
    } else {
      say(`${nm(id)} is not on this task yet`);
    }
  }

  // ------------------------------------------------------------------ commands

  function say(text, bad) {
    const el = $('cmd-msg');
    setText(el, rn(text));
    el.classList.toggle('bad', !!bad);
  }

  async function run(name, args) {
    if (!source || busyCmd) return;
    busyCmd = true;
    if (M) render(M);
    try {
      const out = await source.command(name, args);
      say(out.accepted ? '' : out.message || 'Refused', !out.accepted);
    } finally {
      busyCmd = false;
      if (M) render(M);
    }
  }

  function confirmDialog(title, text, typed) {
    const d = $('confirm');
    if (typeof d.showModal !== 'function') return Promise.resolve(window.confirm(`${title}\n\n${text}`));
    setText($('confirm-title'), title);
    setText($('confirm-text'), text);
    $('confirm-type-wrap').hidden = !typed;
    const input = $('confirm-type');
    input.value = '';
    const ok = $('confirm-ok');
    ok.disabled = !!typed;
    input.oninput = () => { ok.disabled = typed && input.value.trim().toUpperCase() !== 'STOP'; };
    return new Promise((resolve) => {
      d.onclose = () => resolve(d.returnValue === 'ok');
      d.showModal();
      (typed ? input : ok).focus();
    });
  }

  const realArms = () => M && (M.backend === 'real' || M.backend === 'sorter');

  async function plugNext(armId) {
    const id = typeof armId === 'string' ? armId : (M && (M.nodes.find((n) => !n.connected && !n.joined) || {}).arm_id);
    if (realArms() && !(await confirmDialog(`Plug in ${id ? nm(id) : 'the next arm node'}?`, 'The arm may move to its start pose. A person must be at the rig.', false))) return;
    run('plug', { arm: id });
  }

  async function armCommand(cmd, arm) {
    if (cmd === 'remove' || cmd === 'remove-abort') {
      const abort = cmd === 'remove-abort';
      const yes = await confirmDialog(
        `${abort ? 'Remove now' : 'Remove'} ${nm(arm)}?`,
        abort
          ? 'Cancels its running step (it goes back to the other arms), then parks slowly and turns torque off.'
          : 'It finishes its step and places a part it holds, then parks slowly and turns torque off. Its remaining work goes to the other arms.',
        false,
      );
      if (yes) run('remove', { arm, abort });
    } else if (cmd === 'plug') plugNext(arm);
    else run(cmd, { arm });
  }

  function wireControls() {
    $('btn-go').addEventListener('click', async () => {
      if (realArms() && !(await confirmDialog('Go on real arms?', 'The arms will start moving. A person must be at the rig.', false))) return;
      run('go');
    });
    $('btn-pause').addEventListener('click', () => run('pause'));
    $('btn-stop').addEventListener('click', async () => {
      const yes = await confirmDialog(
        'Stop all arms?',
        "Every arm stops its current step now, goes back to its home pose, then turns torque off. This is a software stop. In an emergency, switch off the followers' 12 V supply.",
        realArms(),
      );
      // abort: cancel the running step (a real step can take 15 s; waiting for it to end
      // looked like Stop all doing nothing)
      if (yes) run('stop_all', { abort: true });
    });
    $('btn-auto').addEventListener('click', () => startDemo('once'));
    $('btn-reset').addEventListener('click', () => startDemo(''));
  }

  // ------------------------------------------------------------------ split pane, theme

  function wireSplitter() {
    const app = $('app');
    const sp = $('splitter');
    const set = (pct) => {
      pct = Math.min(40, Math.max(18, pct));
      app.style.setProperty('--left', `${pct}%`);
      sp.setAttribute('aria-valuenow', String(Math.round(pct)));
      store('orchestra-left', String(pct));
    };
    const saved = Number(store('orchestra-left'));
    if (saved) set(saved);
    sp.addEventListener('pointerdown', (e) => {
      sp.setPointerCapture(e.pointerId);
      sp.classList.add('drag');
      const move = (ev) => set((ev.clientX / app.clientWidth) * 100);
      const up = () => {
        sp.classList.remove('drag');
        if (M) O.renderSchedule(M);
        sp.removeEventListener('pointermove', move);
        sp.removeEventListener('pointerup', up);
      };
      sp.addEventListener('pointermove', move);
      sp.addEventListener('pointerup', up);
    });
    sp.addEventListener('dblclick', () => set(24));
    sp.addEventListener('keydown', (e) => {
      const now = Number(sp.getAttribute('aria-valuenow'));
      if (e.key === 'ArrowLeft') set(now - 2);
      if (e.key === 'ArrowRight') set(now + 2);
      if (e.key === 'Home') set(24);
    });
  }

  function wireTheme() {
    const root = document.documentElement;
    const saved = store('orchestra-theme');
    if (saved) root.dataset.theme = saved;
    $('theme').addEventListener('click', () => {
      const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
      root.dataset.theme = dark ? 'light' : 'dark';
      store('orchestra-theme', root.dataset.theme);
      readColors();
      if (M) render(M); // swatches and lanes pick the theme's shade of each node colour
    });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { readColors(); if (M) render(M); });
  }

  // ------------------------------------------------------------------ boot

  function reset() {
    rows.forEach((li) => li.remove());
    cards.forEach((c) => c.remove());
    rows.clear();
    cards.clear();
    joinOrder.clear();
    hist.clear();
    joinSeq = 0;
    lastJoined = null;
    const slot = document.querySelector('.slot');
    if (slot) slot.remove();
  }

  function startDemo(autoplay) {
    if (source) source.stop();
    reset();
    source = O.DemoSource({ speed: params.get('speed'), autoplay });
    source.start(render);
  }

  // ------------------------------------------------------------------ live camera
  // The overhead camera next to the table map: an MJPEG stream from the same feed the scans use.
  // Shown once the server has a camera; dimmed when frames stop, reconnected when they return.
  function wireCamera(base) {
    const panel = $('cam-panel');
    const img = $('cam-img');
    const state = $('cam-state');
    let streaming = false;
    const connect = () => { img.src = `${base}/api/camera/stream?t=${Date.now()}`; streaming = true; };
    img.addEventListener('error', () => { streaming = false; });
    async function poll() {
      try {
        const r = await fetch(`${base}/api/camera`, { cache: 'no-store' });
        const c = r.ok ? await r.json() : { available: false };
        panel.hidden = !c.available;
        if (c.available) {
          const stale = c.age_s === null || c.age_s > 3;
          panel.classList.toggle('stale', stale);
          setText(state, stale ? (c.error || 'waiting for frames…') : `${c.device}, live`);
          if (!streaming || (stale && c.age_s !== null)) connect();
        }
      } catch (e) {
        panel.classList.add('stale');
      }
    }
    poll();
    setInterval(poll, 2000);
  }

  async function boot() {
    wireSplitter();
    wireTheme();
    wireControls();
    const base = params.get('api') || (location.protocol.startsWith('http') ? '' : null);
    if (params.get('demo') !== '1' && base !== null) {
      const config = await O.probeLive(base, 900);
      if (config) {
        source = O.LiveSource(config, base);
        source.start(render);
        wireCamera(base);
        return;
      }
      O.demoReason = 'Start a run with python -m hfteama.demo and open the address it prints.';
    } else if (base === null) {
      O.demoReason = 'Serve the page from the Orchestra API for live data.';
    }
    startDemo(params.get('autoplay') || '');
  }

  setInterval(() => { if (M && M.source === 'live') render(M); }, 1000); // stale check ticks on
  let resizeT = null;
  window.addEventListener('resize', () => { // the Gantt and map are drawn at the pane's width
    clearTimeout(resizeT);
    resizeT = setTimeout(() => { if (M) O.renderSchedule(M); }, 120);
  });
  boot();
})();
