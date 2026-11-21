/* Orchestra live source: the Orchestra API in the demo process (docs/ORCHESTRA_DASHBOARD.md,
 * section 5). GET /api/config, SSE on GET /api/stream (hello, state, event, joints, ik, command),
 * GET /api/tasks for the object chips, POST /api/... for commands.
 *
 * It turns the API's messages into the same screen model the demo source produces, so app.js does
 * not care where the data comes from.
 */
(function () {
  'use strict';
  const O = (window.Orchestra = window.Orchestra || {});

  const COMMANDS = {
    go: () => ['/api/run/go', {}],
    pause: (a, paused) => [paused ? '/api/run/resume' : '/api/run/pause', {}],
    stop_all: (a) => ['/api/run/stop-all', { abort: !!a.abort }],
    plug: (a) => [`/api/arms/${encodeURIComponent(a.arm)}/plug`, {}],
    remove: (a) => [`/api/arms/${encodeURIComponent(a.arm)}/remove`, { abort: !!a.abort }],
    simulate_unplug: (a) => [`/api/arms/${encodeURIComponent(a.arm)}/simulate-unplug`, {}],
    inject_failure: (a) => [`/api/arms/${encodeURIComponent(a.arm)}/inject-failure`, {}],
  };

  async function probe(base, timeoutMs) {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), timeoutMs || 800);
    try {
      const r = await fetch(base + '/api/config', { signal: ctl.signal, cache: 'no-store' });
      return r.ok ? await r.json() : null;
    } catch (e) {
      return null;
    } finally {
      clearTimeout(t);
    }
  }

  // "Sorting objects on the desk" from the batch, when the server does not send a title.
  function taskTitle(task) {
    if (task.title) return { title: task.title, subtitle: task.subtitle || task.note || task.name };
    const steps = task.steps || [];
    const skills = new Set(steps.map((s) => s.skill));
    const objects = new Set(steps.map((s) => s.object).filter(Boolean));
    const targets = new Set(steps.filter((s) => s.skill === 'place' && s.target).map((s) => s.target));
    const sorting = steps.length && [...skills].every((k) => ['pick', 'place', 'pick_place'].includes(k));
    if (sorting || /sort/i.test(task.name || '')) {
      const into = targets.size === 1 ? `into the ${[...targets][0].replace(/_/g, ' ')}` : `into ${targets.size || 'their'} bins`;
      return { title: 'Sorting objects on the desk', subtitle: `${objects.size} objects ${into}` };
    }
    if (/assembl/i.test(task.name || '')) return { title: 'Assembly', subtitle: `${steps.length} steps` };
    return { title: task.name || 'Batch', subtitle: `${steps.length} steps` };
  }
  O.taskTitle = taskTitle;

  function LiveSource(config, base) {
    base = base || '';
    let listener = null;
    let es = null;
    let state = null;
    let tasks = [];
    const joints = {};
    const ik = {};
    const history = {};
    let connected = false;
    let updated = 0;
    let tasksTimer = null;

    const steps = (config.task && config.task.steps) || [];
    const stepById = Object.fromEntries(steps.map((s) => [s.id, s]));
    const names = taskTitle(config.task || {});

    function emit() {
      if (listener && state) listener(model());
    }

    function model() {
      const run = state.run || {};
      const nodes = ((state.nodes && state.nodes.items) || []).map((n) => {
        const step = n.task ? stepById[n.task] || { id: n.task } : null;
        const held = n.held ? stepById[n.held] || { id: n.held } : null;
        const j = joints[n.arm_id];
        return {
          arm_id: n.arm_id, label: n.label || '', color: n.color || '#5b6872', pair: n.pair, serial: n.serial, port: n.port || null,
          node_state: n.node_state, backend: n.backend || config.backend, connected: n.connected, joined: n.joined, parking: n.parking,
          task: step ? { id: step.id, skill: step.skill, label: step.label || step.object || step.id, target: step.target || null,
            elapsed: n.task_elapsed != null ? n.task_elapsed : null, estimate: n.task_estimate != null ? n.task_estimate : step.estimate } : null,
          held: held ? { task: held.id, label: held.label || held.object || held.id } : null,
          waiting_for: n.waiting_for || null,
          heartbeat_age_s: n.heartbeat_age_s, joined_at: n.joined_at, left_at: n.left_at,
          steps_ok: n.steps_ok || 0, steps_failed: n.steps_failed || 0, last_reason: n.last_reason, unblocked: n.unblocked || 0,
          history: history[n.arm_id] || [], fail_armed: false,
          joints: j ? { q: j.q, stale: Date.now() - j.at > 2000 } : null,
          tcp: j && j.tcp ? j.tcp : null, target: null,
          ik: n.task && ik[n.arm_id] && ik[n.arm_id].task === n.task ? ik[n.arm_id] : null,
          reach: (config.reach && config.reach[n.arm_id]) || null,
        };
      });
      const byStep = Object.fromEntries(tasks.map((t) => [t.id, t]));
      const objects = [];
      const seen = new Set();
      steps.forEach((s) => {
        if (!s.object || seen.has(s.object)) return;
        seen.add(s.object);
        const mine = steps.filter((x) => x.object === s.object).map((x) => byStep[x.id] || { state: 'pending' });
        const pick = mine[0] || {};
        const place = mine[mine.length - 1] || {};
        let st = 'waiting';
        let arm = null;
        if (place.state === 'done') { st = 'done'; arm = place.arm; }
        else if (mine.some((x) => x.state === 'failed')) st = 'failed';
        else if (place.state === 'running' || place.state === 'verifying') { st = 'placing'; arm = place.arm; }
        else if (pick.state === 'running') { st = 'picking'; arm = pick.arm; }
        else if (pick.state === 'done') { st = 'held'; arm = pick.arm; }
        else if ((run.blocked || []).includes(pick.id)) st = 'out_of_reach';
        objects.push({ id: s.object, label: s.label || s.object.replace(/_/g, ' '), state: st, arm });
      });
      const phase = run.paused ? 'running' : run.phase;
      return {
        source: 'live', connected, updated, backend: config.backend,
        task: { name: (config.task || {}).name, title: names.title, subtitle: names.subtitle, objects,
          done: run.done || 0, total: run.total || steps.length, blocked: (run.blocked || []).length },
        run: {
          phase, paused: !!run.paused, now: run.now || 0,
          remaining: run.phase === 'running' ? run.remaining : null,
          big: run.phase === 'finished' ? run.predicted_finish : run.estimate,
          limit: run.limit || '', whatif: run.whatif || {},
        },
        replanning: state.replanning || null,
        counts: { known: state.nodes ? state.nodes.known : 0, connected: state.nodes ? state.nodes.connected : 0, joined: state.nodes ? state.nodes.joined : 0 },
        nodes,
        feed: (state.feed || []).map((x) => (typeof x === 'string' ? { t: null, text: x, level: /lost|fail|timeout/i.test(x) ? 'bad' : 'info' } : x)),
        allowed: Object.assign({ go: run.phase === 'estimate', pause: run.phase === 'running', stop_all: run.phase !== 'finished', plug: true, simulate_unplug: false, inject_failure: false }, config.commands || {}),
      };
    }

    async function refreshTasks() {
      try {
        const r = await fetch(base + '/api/tasks', { cache: 'no-store' });
        if (r.ok) {
          tasks = await r.json();
          emit();
        }
      } catch (e) {
        /* the stream reports the connection; tasks retry next second */
      }
    }

    function on(type, fn) {
      es.addEventListener(type, (ev) => {
        try {
          fn(JSON.parse(ev.data));
        } catch (e) {
          console.warn('orchestra: bad', type, e);
        }
      });
    }

    function open() {
      es = new EventSource(base + '/api/stream');
      es.onopen = () => { connected = true; emit(); };
      es.onerror = () => { connected = false; emit(); }; // EventSource reconnects with Last-Event-ID
      on('hello', () => {});
      on('state', (s) => { state = s; updated = Date.now(); connected = true; emit(); });
      on('joints', (j) => {
        const at = Date.now();
        Object.entries(j.arms || {}).forEach(([id, v]) => { joints[id] = { q: v.q, tcp: v.tcp, at }; });
        emit();
      });
      on('ik', (m) => { ik[m.arm] = m; emit(); });
      on('event', (e) => {
        if (e.kind === 'task_end' && e.arm) {
          const h = (history[e.arm] = history[e.arm] || []);
          h.unshift({ task: e.task, outcome: e.outcome === 'ok' ? 'ok' : 'failed' });
          history[e.arm] = h.slice(0, 10);
        }
      });
    }

    return {
      kind: 'live',
      start(onModel) {
        listener = onModel;
        open();
        refreshTasks();
        tasksTimer = setInterval(refreshTasks, 1000);
      },
      stop() { if (es) es.close(); clearInterval(tasksTimer); },
      async command(name, args) {
        const make = COMMANDS[name];
        if (!make) return { accepted: false, message: `Not available live: ${name}` };
        const [path, body] = make(args || {}, state && state.run && state.run.paused);
        const headers = { 'Content-Type': 'application/json' };
        const token = O.token && O.token();
        if (token) headers['X-Orchestra-Token'] = token;
        try {
          const r = await fetch(base + path, { method: 'POST', headers, body: JSON.stringify(body) });
          const out = await r.json().catch(() => ({}));
          return { accepted: r.ok && out.accepted !== false, message: out.message || out.detail || `HTTP ${r.status}` };
        } catch (e) {
          return { accepted: false, message: 'No answer from the orchestrator' };
        }
      },
    };
  }

  O.probeLive = probe;
  O.LiveSource = LiveSource;
})();
