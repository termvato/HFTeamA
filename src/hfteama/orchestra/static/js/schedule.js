/* Orchestra schedule: plan against actual (Gantt, one lane per arm) and the table seen from above
 * (each arm's reach, the bins, every object in the colour of the arm it is planned for).
 *
 * Reads the screen model's `schedule` and `scene` (GET /api/state, docs/ORCHESTRA_DASHBOARD.md
 * section 5.1; js/demo.js makes the same shape). Plain SVG, no chart library. Colours come from
 * the CSS tokens so both themes work.
 */
(function () {
  'use strict';
  const O = (window.Orchestra = window.Orchestra || {});
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const FALLBACK = ['#3b6fb6', '#8a5cb8', '#c77c2e', '#2a8a8a', '#b3403a'];

  function armColours(m) {
    const out = {};
    (m.nodes || []).forEach((n) => { out[n.arm_id] = n.color; });
    return (id, i) => out[id] || FALLBACK[i % FALLBACK.length];
  }

  // a round step for the time axis: 5 s, 10 s, 15 s, 30 s, 1 min, ...
  function tickStep(span, px) {
    const want = span / Math.max(2, Math.floor(px / 90));
    return [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800].find((s) => s >= want) || 3600;
  }

  // ------------------------------------------------------------------ Gantt

  function gantt(m, width) {
    const s = m.schedule || {};
    const plan = s.plan || [];
    const actual = s.actual || [];
    const colour = armColours(m);
    const joined = (m.nodes || []).filter((n) => n.joined).map((n) => n.arm_id);
    const order = (m.nodes || []).map((n) => n.arm_id);
    const lanes = [...new Set([...joined, ...plan.map((b) => b.arm), ...actual.map((b) => b.arm)])]
      .sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
    if (!lanes.length || (!plan.length && !actual.length)) return null;

    const LABEL = Math.min(150, Math.max(96, width * 0.18));
    const LANE = 34;
    const TOP = 6;
    const AXIS = 22;
    const w = Math.max(240, width);
    const h = TOP + lanes.length * LANE + AXIS;
    const end = Math.max(1, s.now || 0, s.finish || 0, ...plan.map((b) => b.end), ...actual.map((b) => b.end)) * 1.03;
    const X = (t) => LABEL + ((w - LABEL - 8) * Math.max(0, t)) / end;
    const Y = (arm) => TOP + lanes.indexOf(arm) * LANE;
    const out = [];

    lanes.forEach((arm, i) => {
      const y = Y(arm);
      if (i % 2) out.push(`<rect x="0" y="${y}" width="${w}" height="${LANE}" class="g-alt"/>`);
      const gone = !joined.includes(arm);
      out.push(`<rect x="8" y="${y + 9}" width="5" height="${LANE - 18}" rx="2" fill="${esc(colour(arm, i))}"${gone ? ' opacity=".35"' : ''}/>`);
      out.push(`<text x="20" y="${y + LANE / 2 + 4}" class="g-lane${gone ? ' gone' : ''}">${esc(arm)}</text>`);
    });

    const step = tickStep(end, w - LABEL);
    for (let t = 0; t <= end; t += step) {
      const x = X(t);
      out.push(`<line x1="${x}" x2="${x}" y1="${TOP}" y2="${h - AXIS}" class="g-grid"/>`);
      out.push(`<text x="${x}" y="${h - 6}" class="g-tick" text-anchor="middle">${O.clock(t)}</text>`);
    }

    const label = (b, x0, x1, y, cls) => (x1 - x0 > 7 * b.task.length + 8
      ? `<text x="${(x0 + x1) / 2}" y="${y}" class="${cls}" text-anchor="middle">${esc(b.task)}</text>` : '');
    plan.forEach((b) => {
      const y = Y(b.arm);
      if (y == null) return;
      const x0 = X(b.start);
      const x1 = Math.max(x0 + 2, X(b.end));
      out.push(`<rect x="${x0}" y="${y + 4}" width="${x1 - x0}" height="${LANE - 8}" rx="3" class="g-plan"><title>${esc(b.task)} planned ${O.clock(b.start)} to ${O.clock(b.end)}</title></rect>`);
      if (!actual.some((a) => a.task === b.task)) out.push(label(b, x0, x1, y + LANE / 2 + 4, 'g-plan-t'));
    });
    actual.forEach((b) => {
      const y = Y(b.arm);
      const x0 = X(b.start);
      const x1 = Math.max(x0 + 2, X(b.end));
      const st = b.state || 'ok';
      out.push(`<rect x="${x0}" y="${y + 10}" width="${x1 - x0}" height="${LANE - 20}" rx="2" class="g-act g-${esc(st)}"><title>${esc(b.task)} ${esc(st)} on ${esc(b.arm)}: ${O.clock(b.start)} to ${O.clock(b.end)}</title></rect>`);
      out.push(label(b, x0, x1, y + LANE / 2 + 4, 'g-act-t'));
    });

    if (s.finish != null) {
      const x = X(s.finish);
      out.push(`<line x1="${x}" x2="${x}" y1="${TOP}" y2="${h - AXIS}" class="g-finish"><title>Finish ${O.clock(s.finish)}</title></line>`);
    }
    if (m.run && m.run.phase === 'running') {
      const x = X(s.now || 0);
      out.push(`<line x1="${x}" x2="${x}" y1="${TOP - 4}" y2="${h - AXIS}" class="g-now"/>`);
    }
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Plan against actual, one lane per arm">${out.join('')}</svg>`;
  }

  // ------------------------------------------------------------------ table map

  function tableMap(m, width) {
    const sc = m.scene;
    if (!sc || !((sc.arms || []).length || (sc.objects || []).length)) return null;
    const colour = armColours(m);
    const arms = sc.arms || [];
    const bins = sc.bins || [];
    const objs = sc.objects || [];
    // which arm takes each step: the current plan, else the arm that did (or is doing) it
    const planned = {};
    ((m.schedule && m.schedule.actual) || []).forEach((b) => { planned[b.task] = b.arm; });
    ((m.schedule && m.schedule.plan) || []).forEach((b) => { planned[b.task] = b.arm; });
    const doneTasks = new Set(((m.schedule && m.schedule.actual) || []).filter((b) => b.state === 'ok').map((b) => b.task));
    const live = new Set((m.nodes || []).filter((n) => n.joined).map((n) => n.arm_id));
    const armIndex = Object.fromEntries(arms.map((a, i) => [a.id, i]));

    // bounds: objects, bins and arm bases, plus a margin; reach circles may run off the edge
    const pts = [...objs.map((o) => o.xy), ...bins.map((b) => b.xy), ...arms.map((a) => a.xy)].filter(Boolean);
    if (!pts.length) return null;
    let x0 = Math.min(...pts.map((p) => p[0])) - 0.08;
    let x1 = Math.max(...pts.map((p) => p[0])) + 0.08;
    let y0 = Math.min(...pts.map((p) => p[1])) - 0.08;
    let y1 = Math.max(...pts.map((p) => p[1])) + 0.08;
    const w = Math.max(220, width);
    const h = Math.round(Math.min(420, Math.max(200, (w * (y1 - y0)) / (x1 - x0))));
    // keep metres square: widen whichever side is short
    const k = Math.min(w / (x1 - x0), h / (y1 - y0));
    const cx = (x0 + x1) / 2;
    const cy = (y0 + y1) / 2;
    x0 = cx - w / k / 2; x1 = cx + w / k / 2; y0 = cy - h / k / 2; y1 = cy + h / k / 2;
    const X = (x) => (x - x0) * k;
    const Y = (y) => (y1 - y) * k; // table y points up on screen
    const out = [`<defs><clipPath id="map-clip"><rect width="${w}" height="${h}"/></clipPath></defs><g clip-path="url(#map-clip)">`];

    arms.forEach((a, i) => {
      const c = esc(colour(a.id, i));
      const on = live.has(a.id);
      [a.r_min, a.r_max].forEach((r) => {
        if (r && r < 5) out.push(`<circle cx="${X(a.xy[0])}" cy="${Y(a.xy[1])}" r="${r * k}" fill="none" stroke="${c}" stroke-width="1.5"${on ? '' : ' stroke-dasharray="3 4" opacity=".4"'}/>`);
      });
    });
    bins.forEach((b) => {
      const r = (b.radius || 0.08) * k;
      out.push(`<circle cx="${X(b.xy[0])}" cy="${Y(b.xy[1])}" r="${r}" class="m-bin"/>`);
      out.push(`<text x="${X(b.xy[0])}" y="${Y(b.xy[1]) + 4}" class="m-bin-t" text-anchor="middle">${esc(b.id)}</text>`);
    });
    arms.forEach((a, i) => {
      const on = live.has(a.id);
      const x = X(a.xy[0]);
      const y = Y(a.xy[1]);
      out.push(`<rect x="${x - 8}" y="${y - 8}" width="16" height="16" rx="3" fill="${esc(colour(a.id, i))}"${on ? '' : ' opacity=".35"'}><title>${esc(a.id)}${on ? '' : ' (not in the run)'}</title></rect>`);
      out.push(`<text x="${x}" y="${y - 13}" class="m-arm${on ? '' : ' gone'}" text-anchor="middle">${esc(a.id)}</text>`);
    });
    objs.forEach((o) => {
      if (!o.xy) return;
      const tasks = o.tasks || [];
      const arm = o.arm || tasks.map((t) => planned[t]).find(Boolean);
      const finished = tasks.length && tasks.every((t) => doneTasks.has(t));
      const fill = arm ? esc(colour(arm, armIndex[arm] || 0)) : 'var(--faint)';
      const x = X(o.xy[0]);
      const y = Y(o.xy[1]);
      const name = o.label || o.id;
      const who = arm ? `planned for ${arm}` : 'no arm can take it now';
      out.push(`<g${finished ? ' opacity=".3"' : ''}><circle cx="${x}" cy="${y}" r="6.5" fill="${fill}" class="m-obj"><title>${esc(name)}: ${finished ? 'sorted' : esc(who)}</title></circle>`);
      out.push(`<text x="${x}" y="${y + 18}" class="m-obj-t" text-anchor="middle">${esc(name)}</text></g>`);
    });
    out.push('</g>');
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="Table seen from above: arms, reach, bins and objects">${out.join('')}</svg>`;
  }

  // ------------------------------------------------------------------ what-if chips

  function whatif(m) {
    const r = m.run || {};
    const n = m.counts ? m.counts.joined : 0;
    const items = Object.entries(r.whatif || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
    if (!items.length || r.phase === 'finished') return '';
    return items.map(([k, v]) => {
      const f = typeof v === 'object' && v ? v : { finish: v, blocked: 0 };
      const val = f.blocked ? `${f.blocked} out of reach` : O.clock(f.finish);
      const kk = Number(k);
      return `<span class="wi${kk === n ? ' cur' : ''}"><b>${esc(val)}</b><span>${kk} arm${kk === 1 ? '' : 's'}</span></span>`;
    }).join('');
  }

  // ------------------------------------------------------------------ render

  function put(el, html) {
    if (el._html !== html) {
      el.innerHTML = html;
      el._html = html;
    }
  }

  O.renderSchedule = function (m) {
    const $ = (id) => document.getElementById(id);
    const wi = whatif(m);
    $('whatif').hidden = !wi;
    put($('whatif'), wi ? `<span class="wi-h">Whole batch with</span>${wi}` : '');

    const gEl = $('gantt');
    const g = gantt(m, gEl.clientWidth || 600);
    $('gantt-panel').hidden = !g;
    if (g) put(gEl, g);

    const mEl = $('map');
    const t = tableMap(m, mEl.clientWidth || 360);
    $('map-panel').hidden = !t;
    if (t) put(mEl, t);
    $('schedule').hidden = !g && !t;
  };
})();
