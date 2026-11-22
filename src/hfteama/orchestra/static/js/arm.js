/* Orchestra arm nodes: display names and colours for the followers, and a live SO-101 drawing.
 *
 * Names are display only. The arm ids (vato_follower = hacklab, rakesh_follower) stay what the
 * configs, the runner (--arms ...) and the API use; the screen shows them as Arm Node A / B.
 *
 * ArmView draws an SO-101 from the side (base, shoulder lift, elbow flex, wrist flex, gripper) and
 * poses it from the joint angles every frame; shoulder pan and wrist roll cannot be seen from the
 * side, so the card shows pan on a small top-view dial. Joint convention as in the sim and demo
 * (Menagerie radians): lift 0 = upper arm up, elbow 0 = forearm at a right angle to it, wrist 0 =
 * in line with the forearm; the rest pose [_, -1.6, 1.6, 1.0, ...] folds the arm back on itself.
 * It is a schematic for the operator, not the IK model.
 */
(function () {
  'use strict';
  const O = (window.Orchestra = window.Orchestra || {});
  const R2D = 180 / Math.PI;

  // body: printed parts; edge: their outline; color / colorDark: swatches, lanes, reach circles.
  const NODES = {
    vato_follower: { name: 'Arm Node A', short: 'A', finish: 'Green SO-101 (hacklab)', body: '#2e8b57', edge: '#1b5a37', color: '#2e8b57', colorDark: '#4fbf80' },
    rakesh_follower: { name: 'Arm Node B', short: 'B', finish: 'Black SO-101 (rakesh)', body: '#23272c', edge: '#59626b', color: '#23272c', colorDark: '#8b949e' },
    third_follower: { name: 'Arm Node C', short: 'C', finish: 'Blue SO-101', body: '#3b6fb6', edge: '#264c80', color: '#3b6fb6', colorDark: '#6f9fe0' },
  };
  O.NODES = NODES;

  O.isDark = function () {
    const t = document.documentElement.dataset.theme;
    if (t) return t === 'dark';
    return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches;
  };
  O.armName = (id) => (NODES[id] ? NODES[id].name : id);
  O.armFinish = (id) => (NODES[id] ? NODES[id].finish : 'SO-101');
  O.armColor = (id, fallback) => {
    const n = NODES[id];
    if (!n) return fallback;
    return O.isDark() ? n.colorDark : n.color;
  };

  // "Arm rakesh_follower joined" -> "Arm Node A joined", in feed lines and messages.
  const RX = new RegExp(`(?:\\b[Aa]rm\\s+)?\\b(${Object.keys(NODES).join('|')})\\b`, 'g');
  O.rename = (s) => String(s == null ? '' : s).replace(RX, (_, id) => NODES[id].name);

  // ------------------------------------------------------------------ geometry (mm)

  const SH = 118; // shoulder axis above the table
  const L1 = 116; // shoulder -> elbow
  const L2 = 135; // elbow -> wrist
  const LIFT_OFFSET = 0.15; // the upper arm's bend in the model, as in demo.js
  const REST = [0, -1.6, 1.6, 1.0, 0, 0.3];
  O.REST_Q = REST;

  const GRIP = [-0.174533, 1.7453292];

  const SKELETON = `
    <rect class="a-floor" x="-400" y="0" width="900" height="60"/>
    <line class="a-floor-line" x1="-400" x2="500" y1="0" y2="0"/>
    <ellipse class="a-shadow" cx="0" cy="3" rx="86" ry="8"/>
    <rect class="a-link" x="-64" y="-12" width="128" height="12" rx="3"/>
    <rect class="a-servo" x="-36" y="-58" width="72" height="46" rx="6"/>
    <rect class="a-servo-hi" x="-30" y="-54" width="60" height="5" rx="2"/>
    <rect class="a-link" x="-42" y="-68" width="84" height="12" rx="4"/>
    <path class="a-link" d="M-28 -60 L-28 -118 A28 28 0 0 1 28 -118 L28 -60 Z"/>
    <g data-j="1">
      <rect class="a-link" x="-17" y="-15" width="${L1 + 34}" height="30" rx="15"/>
      <rect class="a-slot" x="26" y="-5" width="${L1 - 52}" height="10" rx="5"/>
      <rect class="a-servo" x="-22" y="-20" width="44" height="40" rx="6"/>
      <circle class="a-horn" r="10"/><circle class="a-pin" r="3"/>
      <g data-j="2" transform="translate(${L1} 0)">
        <rect class="a-link" x="-11" y="-12" width="${L2 + 22}" height="24" rx="12"/>
        <rect class="a-slot" x="28" y="-4" width="${L2 - 60}" height="8" rx="4"/>
        <rect class="a-servo" x="-20" y="-18" width="40" height="36" rx="6"/>
        <circle class="a-horn" r="9"/><circle class="a-pin" r="2.6"/>
        <g data-j="3" transform="translate(${L2} 0)">
          <rect class="a-servo" x="12" y="-14" width="34" height="28" rx="5"/>
          <rect class="a-link" x="44" y="-17" width="18" height="34" rx="4"/>
          <path class="a-link" d="M60 5 L114 9 Q119 10 118 15 L117 18 L60 17 Z"/>
          <g data-j="5" transform="translate(60 -7)">
            <path class="a-link" d="M0 0 L52 -3 Q57 -4 56 -9 L55 -12 L0 -12 Z"/>
          </g>
          <rect class="a-servo" x="-18" y="-16" width="36" height="32" rx="6"/>
          <circle class="a-horn" r="8"/><circle class="a-pin" r="2.4"/>
        </g>
      </g>
    </g>`;

  function make(tag, attrs) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs || {}).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
  }

  // One SO-101, posed by update(q). opts.mini: the small glyph in the robot list.
  function ArmView(opts) {
    opts = opts || {};
    const svg = make('svg', {
      class: `arm-svg${opts.mini ? ' mini' : ''}`,
      viewBox: opts.mini ? '-100 -290 340 300' : '-160 -335 500 360',
      preserveAspectRatio: 'xMidYMax meet',
      role: 'img',
      'aria-label': opts.label || 'SO-101 arm, side view',
    });
    svg.innerHTML = `<g>${SKELETON}</g>`; // origin: the base's centre on the table
    const root = svg.firstElementChild;
    const j1 = root.querySelector('[data-j="1"]');
    const j2 = root.querySelector('[data-j="2"]');
    const j3 = root.querySelector('[data-j="3"]');
    const j5 = root.querySelector('[data-j="5"]');
    let last = '';

    function update(q) {
      q = q && q.length >= 5 ? q : REST;
      const a1 = Math.PI / 2 - LIFT_OFFSET - q[1];
      const rel2 = -(q[2] + Math.PI / 2);
      const rel3 = -q[3];
      const g = q.length > 5 ? q[5] : REST[5];
      const open = Math.max(0, Math.min(1, (g - GRIP[0]) / (GRIP[1] - GRIP[0]))) * 38;
      const key = [a1, rel2, rel3, open].map((v) => v.toFixed(3)).join(',');
      if (key === last) return;
      last = key;
      j1.setAttribute('transform', `translate(0 ${-SH}) rotate(${(-a1 * R2D).toFixed(2)})`);
      j2.setAttribute('transform', `translate(${L1} 0) rotate(${(-rel2 * R2D).toFixed(2)})`);
      j3.setAttribute('transform', `translate(${L2} 0) rotate(${(-rel3 * R2D).toFixed(2)})`);
      j5.setAttribute('transform', `translate(60 -7) rotate(${(-open).toFixed(2)})`);
    }

    function paint(id, fallback) {
      const n = NODES[id];
      const body = n ? n.body : fallback || '#5b6872';
      svg.style.setProperty('--arm', body);
      svg.style.setProperty('--arm-edge', n ? n.edge : `color-mix(in srgb, ${body} 65%, #000)`);
    }

    update(REST);
    return { el: svg, update, paint };
  }
  O.ArmView = ArmView;

  // Top view of shoulder pan: a dial with the arm's heading.
  O.panDial = function () {
    const svg = make('svg', { class: 'pan-dial', viewBox: '-20 -20 40 40', 'aria-hidden': 'true' });
    svg.innerHTML = `
      <circle class="pd-ring" r="17"/>
      <path class="pd-range" d=""/>
      <line class="pd-zero" x1="0" y1="0" x2="0" y2="-17"/>
      <g class="pd-needle"><line x1="0" y1="0" x2="0" y2="-14"/><circle r="3"/></g>`;
    const needle = svg.querySelector('.pd-needle');
    // the pan limit (+-110 deg) as a faint arc
    const lim = 110 / R2D;
    const p = (a, r) => `${(r * Math.sin(a)).toFixed(2)} ${(-r * Math.cos(a)).toFixed(2)}`;
    svg.querySelector('.pd-range').setAttribute('d', `M${p(-lim, 17)} A17 17 0 1 1 ${p(lim, 17)}`);
    return {
      el: svg,
      update(rad) {
        needle.setAttribute('transform', `rotate(${(-(rad || 0) * R2D).toFixed(1)})`);
      },
    };
  };
})();
