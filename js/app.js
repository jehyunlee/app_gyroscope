import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import {
  SHAPES,
  inertia,
  initialState,
  observe,
  omegaOf,
  momentumFromOmega,
  torques,
  rk4Step,
  stableStep,
  steadyPrecession,
  gyroscopicApprox,
  nutationRate,
  sleepingThreshold,
  axisOf,
  vec,
} from './physics.js';

const DEG = Math.PI / 180;

// ---------- parameters ----------

const FIELDS = {
  spin: { label: '스핀 ω<sub>s</sub>', min: -200, max: 200, step: 1, unit: 'rad/s', initial: true, fmt: (v) => `${v.toFixed(0)} rad/s · ${Math.abs((v * 60) / (2 * Math.PI)).toFixed(0)} rpm` },
  theta: { label: '기울기 θ₀ <small>(수직에서)</small>', min: 0, max: 150, step: 1, unit: '°', initial: true, fmt: (v) => `${v.toFixed(0)}°` },
  phiDot: { label: '처음 세차 속도 φ̇₀', min: -15, max: 25, step: 0.1, unit: 'rad/s', initial: true, fmt: (v) => `${v.toFixed(2)} rad/s` },
  thetaDot: { label: '처음 끄덕임 θ̇₀', min: -10, max: 10, step: 0.1, unit: 'rad/s', initial: true, fmt: (v) => `${v.toFixed(1)} rad/s` },
  mass: { label: '바퀴 질량 m', min: 0.1, max: 5, step: 0.1, fmt: (v) => `${v.toFixed(1)} kg` },
  radius: { label: '바퀴 반지름 r', min: 0.03, max: 0.2, step: 0.005, fmt: (v) => `${(v * 100).toFixed(1)} cm` },
  arm: { label: '막대 길이 l <small>(피벗–바퀴 중심)</small>', min: 0, max: 0.3, step: 0.005, fmt: (v) => `${(v * 100).toFixed(1)} cm` },
  g: { label: '중력 가속도 g', min: 0, max: 30, step: 0.01, fmt: (v) => `${v.toFixed(2)} m/s²` },
  spinDamping: { label: '베어링 마찰 <small>(스핀 감쇠)</small>', min: 0, max: 0.002, step: 0.00005, fmt: (v) => (v === 0 ? '없음' : `${(v * 1e4).toFixed(1)}`) },
  pivotDamping: { label: '피벗 마찰 <small>(세차·장동 감쇠)</small>', min: 0, max: 0.02, step: 0.0005, fmt: (v) => (v === 0 ? '없음' : `${(v * 1e3).toFixed(1)}`) },
  timeScale: { label: '시간 배속', min: 0.05, max: 2, step: 0.05, fmt: (v) => `×${v.toFixed(2)}` },
};

const DEFAULTS = {
  spin: 100, theta: 70, phiDot: 0, thetaDot: 0,
  mass: 1, radius: 0.1, arm: 0.15, shape: 'ring', g: 9.81,
  spinDamping: 0, pivotDamping: 0, timeScale: 1,
};

// `steady: true` sets φ̇₀ to the exact steady-precession rate for the chosen θ₀, ω_s.
const PRESETS = [
  { id: 'steady', label: '정상 세차', steady: true, set: { spin: 100, theta: 70 } },
  // Nutation at ~3 Hz is too quick to follow, so these start in slow motion.
  { id: 'cusp', label: '그냥 놓기 (뾰족점)', set: { spin: 50, theta: 60, phiDot: 0, timeScale: 0.25 } },
  { id: 'loop', label: '고리 장동', set: { spin: 50, theta: 60, phiDot: -2, timeScale: 0.25 } },
  { id: 'wave', label: '물결 장동', set: { spin: 50, theta: 60, phiDot: 5, timeScale: 0.25 } },
  { id: 'nospin', label: '스핀 0 (넘어짐)', set: { spin: 0, theta: 60, phiDot: 0 } },
  { id: 'sleep', label: '잠자는 팽이', set: { spin: 70, theta: 3, phiDot: 0 } },
  { id: 'free', label: '무중력', set: { spin: 40, theta: 30, phiDot: 4, g: 0 } },
  { id: 'friction', label: '마찰 있는 현실', steady: true, set: { spin: 120, theta: 80, spinDamping: 0.0006, pivotDamping: 0.004 } },
];

const params = { ...DEFAULTS };
const physicsParams = () => ({
  mass: params.mass, radius: params.radius, arm: params.arm, shape: params.shape,
  g: params.g, spinDamping: params.spinDamping, pivotDamping: params.pivotDamping,
});

// ---------- DOM: controls ----------

const inputs = {};
for (const el of document.querySelectorAll('.field')) {
  const key = el.dataset.key;
  const f = FIELDS[key];
  el.innerHTML = `<div class="head"><label for="f-${key}">${f.label}</label><output id="o-${key}"></output></div>
    <input type="range" id="f-${key}" min="${f.min}" max="${f.max}" step="${f.step}">`;
  const input = el.querySelector('input');
  inputs[key] = input;
  input.addEventListener('input', () => {
    setParam(key, Number(input.value));
    markPreset(null);
  });
}

const shapeSelect = document.getElementById('shape');
for (const [key, s] of Object.entries(SHAPES)) shapeSelect.add(new Option(s.label, key));
shapeSelect.addEventListener('change', () => {
  setParam('shape', shapeSelect.value);
  markPreset(null);
});

function syncControls() {
  for (const [key, input] of Object.entries(inputs)) {
    input.value = params[key];
    document.getElementById(`o-${key}`).textContent = FIELDS[key].fmt(params[key]);
  }
  shapeSelect.value = params.shape;
}

// Initial conditions restart the run. Physical parameters apply live while
// keeping the current angular velocity, so you can watch e.g. g change mid-flight.
function setParam(key, value) {
  if (FIELDS[key]?.initial) {
    params[key] = value;
    syncControls();
    reset();
    return;
  }
  const omega = omegaOf(sim.state, physicsParams());
  params[key] = value;
  if (key !== 'timeScale') {
    sim.state = { q: sim.state.q, L: momentumFromOmega(sim.state.q, omega, physicsParams()) };
    // A new g, m or I legitimately changes the energy; measure drift from here.
    sim.E0 = observe(sim.state, physicsParams()).energy;
  }
  syncControls();
  updateGeometry();
  updateTexts();
}

const presetNav = document.getElementById('presets');
for (const preset of PRESETS) {
  const b = document.createElement('button');
  b.type = 'button';
  b.textContent = preset.label;
  b.dataset.preset = preset.id;
  presetNav.append(b);
}
document.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-preset]');
  if (button) applyPreset(button.dataset.preset);
  const gravity = event.target.closest('button[data-g]');
  if (gravity) {
    setParam('g', Number(gravity.dataset.g));
    markPreset(null);
  }
});

function applyPreset(id) {
  const preset = PRESETS.find((p) => p.id === id);
  Object.assign(params, DEFAULTS, preset.set);
  if (preset.steady) params.phiDot = steadyPrecession(params.theta * DEG, params.spin, physicsParams());
  syncControls();
  updateGeometry();
  reset();
  markPreset(id);
}

function markPreset(id) {
  for (const b of presetNav.children) b.classList.toggle('active', b.dataset.preset === id);
}

document.getElementById('steady').addEventListener('click', () => {
  const rate = steadyPrecession(params.theta * DEG, params.spin, physicsParams());
  if (rate === null || !Number.isFinite(rate)) return;
  // Kept exact (not snapped to the slider step) so the start is truly nutation-free.
  params.phiDot = Math.max(FIELDS.phiDot.min, Math.min(FIELDS.phiDot.max, rate));
  syncControls();
  reset();
  markPreset(null);
});

// ---------- simulation state ----------

const sim = { state: null, t: 0, running: true, E0: 0, Lref: 1, Wref: 1, spinAngle: 0, history: [] };

function reset() {
  const p = physicsParams();
  sim.state = initialState(
    { theta: params.theta * DEG, phi: 0, spin: params.spin, phiDot: params.phiDot, thetaDot: params.thetaDot },
    p,
  );
  sim.t = 0;
  sim.spinAngle = 0;
  sim.E0 = observe(sim.state, p).energy;
  // Arrow scales are fixed per run so shrinking L (friction) is visible.
  const { I1, I3 } = inertia(p);
  sim.Lref = Math.max(vec.norm(sim.state.L), I3 * 30, I1 * 3);
  sim.Wref = Math.max(vec.norm(omegaOf(sim.state, p)), 5);
  sim.history = [];
  trail.count = 0;
  updateTexts();
}

const playButton = document.getElementById('play');
playButton.addEventListener('click', () => {
  sim.running = !sim.running;
  playButton.textContent = sim.running ? '⏸ 일시정지' : '▶ 재생';
});
document.getElementById('reset').addEventListener('click', reset);

// A short tap on the wheel: angular impulse ΔL = r × J, sized to give the
// non-spinning wheel about 1.5 rad/s of swing.
function kick(direction) {
  const p = physicsParams();
  const { I1 } = inertia(p);
  const e = axisOf(sim.state.q);
  const lever = vec.scale(e, Math.max(p.arm, p.radius));
  const horizontal = vec.cross([0, 1, 0], e);
  const side = vec.norm(horizontal) > 1e-6 ? vec.scale(horizontal, 1 / vec.norm(horizontal)) : [0, 0, 1];
  const dir = direction === 'down' ? [0, -1, 0] : side;
  const impulse = (1.5 * I1) / vec.norm(lever);
  sim.state = { q: sim.state.q, L: vec.add(sim.state.L, vec.cross(lever, vec.scale(dir, impulse))) };
  sim.E0 = observe(sim.state, p).energy;
  markPreset(null);
}
document.getElementById('kickDown').addEventListener('click', () => kick('down'));
document.getElementById('kickSide').addEventListener('click', () => kick('side'));

// ---------- three.js scene ----------

const viewport = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0e14);
scene.fog = new THREE.Fog(0x0a0e14, 3, 9);

const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
camera.position.set(0.62, 0.36, 0.78);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, -0.03, 0);
controls.enableDamping = true;
controls.minDistance = 0.4;
controls.maxDistance = 5;

scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x20242c, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.position.set(1.5, 3, 1.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.5, far: 6 });
scene.add(sun);

const GROUND = -0.62;
const ground = new THREE.Mesh(new THREE.CircleGeometry(3, 64), new THREE.MeshStandardMaterial({ color: 0x141a22, roughness: 0.95 }));
ground.rotation.x = -Math.PI / 2;
ground.position.y = GROUND;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.PolarGridHelper(1.2, 12, 6, 64, 0x2a3442, 0x1d2530);
grid.position.y = GROUND + 0.001;
scene.add(grid);

const metal = new THREE.MeshStandardMaterial({ color: 0x9aa7b8, metalness: 0.8, roughness: 0.3 });
const stand = new THREE.Group();
const post = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.018, -GROUND - 0.02, 24), metal);
post.position.y = (GROUND - 0.02) / 2 - 0.0;
post.castShadow = true;
const base = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.14, 0.03, 48), new THREE.MeshStandardMaterial({ color: 0x2c3544, metalness: 0.4, roughness: 0.6 }));
base.position.y = GROUND + 0.015;
base.receiveShadow = true;
base.castShadow = true;
const pivot = new THREE.Mesh(new THREE.SphereGeometry(0.02, 24, 16), new THREE.MeshStandardMaterial({ color: 0xe6edf3, metalness: 0.6, roughness: 0.25 }));
stand.add(post, base, pivot);
scene.add(stand);

// The axle frame carries precession and nutation; the wheel spins inside it
// by a separately integrated display angle (see slowSpin).
const axle = new THREE.Group();
scene.add(axle);
const rod = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, 1, 16), metal);
rod.castShadow = true;
axle.add(rod);
const tip = new THREE.Mesh(new THREE.SphereGeometry(0.01, 16, 12), new THREE.MeshStandardMaterial({ color: 0xc792ea, emissive: 0x4b2b66 }));
axle.add(tip);

const wheel = new THREE.Group();
axle.add(wheel);
const rimMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, metalness: 0.55, roughness: 0.35 });
const markMat = new THREE.MeshStandardMaterial({ color: 0xffd166, metalness: 0.3, roughness: 0.4, emissive: 0x3a2a00 });
const spokeMat = new THREE.MeshStandardMaterial({ color: 0xc8d1dc, metalness: 0.7, roughness: 0.3 });
let wheelParts = [];

function buildWheel() {
  for (const part of wheelParts) {
    wheel.remove(part);
    part.geometry.dispose();
  }
  wheelParts = [];
  const r = params.radius;
  const disc = params.shape === 'disc';
  const tube = disc ? r * 0.06 : r * 0.1;
  const rim = new THREE.Mesh(new THREE.TorusGeometry(r, tube, 16, 96), rimMat);
  rim.rotation.x = Math.PI / 2;
  wheelParts.push(rim);
  if (disc) {
    const plate = new THREE.Mesh(new THREE.CylinderGeometry(r, r, tube * 1.2, 64), rimMat);
    wheelParts.push(plate);
  }
  const spokes = disc ? 0 : 6;
  for (let i = 0; i < spokes; i++) {
    const s = new THREE.Mesh(new THREE.BoxGeometry(r, r * 0.05, r * 0.05), spokeMat);
    const a = (i / spokes) * Math.PI * 2;
    s.position.set((Math.cos(a) * r) / 2, 0, (-Math.sin(a) * r) / 2);
    s.rotation.y = a;
    wheelParts.push(s);
  }
  // Two coloured markers make the spin direction and speed readable.
  for (const a of [0, Math.PI]) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(tube * 2.6, tube * 2.6, tube * 2.6), a === 0 ? markMat : spokeMat);
    m.position.set(Math.cos(a) * r, disc ? tube * 0.9 : 0, -Math.sin(a) * r);
    wheelParts.push(m);
  }
  const hub = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.14, r * 0.14, r * 0.22, 24), spokeMat);
  wheelParts.push(hub);
  for (const part of wheelParts) {
    part.castShadow = true;
    wheel.add(part);
  }
}

function updateGeometry() {
  const arm = params.arm;
  const rodLength = arm + 0.05;
  rod.scale.set(1, rodLength, 1);
  rod.position.y = rodLength / 2 - 0.02;
  tip.position.y = rodLength - 0.02;
  wheel.position.y = arm;
  const s = Math.cbrt(params.mass);
  pivot.scale.setScalar(Math.max(0.8, Math.min(1.4, s)));
  buildWheel();
}

// Arrows share the pivot as origin; τ is drawn from the tip of L (τ·Δt).
function makeArrow(color) {
  const a = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.1, color, 0.04, 0.025);
  a.line.material.linewidth = 2;
  a.cone.material = new THREE.MeshBasicMaterial({ color });
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.0045, 0.0045, 1, 8), new THREE.MeshBasicMaterial({ color }));
  a.add(shaft);
  a.userData.shaft = shaft;
  scene.add(a);
  return a;
}
const arrows = { L: makeArrow(0xff9f1c), tau: makeArrow(0xff4d6d), w: makeArrow(0x3ddbd9), g: makeArrow(0x7bd88f) };
const TAU_DT = 0.3;
const L_LENGTH = 0.3;
const tmpV = new THREE.Vector3();

function setArrow(arrow, origin, v, visible) {
  const length = vec.norm(v);
  arrow.visible = visible && length > 0.004;
  if (!arrow.visible) return;
  arrow.position.set(...origin);
  arrow.setDirection(tmpV.set(v[0] / length, v[1] / length, v[2] / length));
  const head = Math.min(0.045, length * 0.45);
  arrow.setLength(length, head, head * 0.55);
  arrow.userData.shaft.scale.y = Math.max(length - head, 1e-4);
  arrow.userData.shaft.position.y = (length - head) / 2;
}

// Trail of the axle tip.
const TRAIL_MAX = 1500;
const trail = { positions: new Float32Array(TRAIL_MAX * 3), colors: new Float32Array(TRAIL_MAX * 3), count: 0 };
const trailGeometry = new THREE.BufferGeometry();
trailGeometry.setAttribute('position', new THREE.BufferAttribute(trail.positions, 3));
trailGeometry.setAttribute('color', new THREE.BufferAttribute(trail.colors, 3));
const trailLine = new THREE.Line(trailGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 }));
trailLine.frustumCulled = false;
scene.add(trailLine);
const trailColor = new THREE.Color(0xc792ea);
const bg = new THREE.Color(0x0a0e14);

function pushTrail(point) {
  if (trail.count === TRAIL_MAX) {
    trail.positions.copyWithin(0, 3);
    trail.count -= 1;
  }
  trail.positions.set(point, trail.count * 3);
  trail.count += 1;
}

function refreshTrail() {
  const n = trail.count;
  const c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    c.copy(bg).lerp(trailColor, 0.15 + (0.85 * i) / Math.max(1, n - 1));
    trail.colors.set([c.r, c.g, c.b], i * 3);
  }
  trailGeometry.setDrawRange(0, n);
  trailGeometry.attributes.position.needsUpdate = true;
  trailGeometry.attributes.color.needsUpdate = true;
}

// ---------- per-frame ----------

const checks = Object.fromEntries(['showL', 'showTau', 'showW', 'showG', 'showTrail', 'slowSpin'].map((id) => [id, document.getElementById(id)]));
checks.showTrail.addEventListener('change', () => {
  trailLine.visible = checks.showTrail.checked;
});

const historySeconds = () => Math.max(2, Math.min(8, 8 * params.timeScale));
let last = performance.now();
let hudTimer = 0;

function frame(now) {
  const dt = Math.min((now - last) / 1000, 1 / 20);
  last = now;
  const p = physicsParams();

  if (sim.running) {
    let remaining = dt * params.timeScale;
    let guard = 0;
    while (remaining > 1e-9 && guard < 6000) {
      const h = Math.min(stableStep(sim.state, p), remaining);
      sim.state = rk4Step(sim.state, p, h);
      remaining -= h;
      sim.t += h;
      guard += 1;
    }
    const o = observe(sim.state, p);
    const visualRate = checks.slowSpin.checked ? Math.sign(o.spin) * Math.min(Math.abs(o.spin), 2 * Math.PI * 1.2) : o.spin;
    sim.spinAngle = (sim.spinAngle + visualRate * dt * params.timeScale) % (2 * Math.PI);
    sim.history.push({ t: sim.t, theta: o.theta / DEG, phiDot: o.phiDot });
    while (sim.history.length && sim.history[0].t < sim.t - 8) sim.history.shift();
    pushTrail(vec.scale(o.e, params.arm + 0.03));
  }

  const o = observe(sim.state, p);
  // Axle orientation without spin: rotate ŷ onto e by the shortest arc.
  axle.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tmpV.set(...o.e));
  wheel.rotation.y = sim.spinAngle;

  const T = torques(sim.state, p);
  const Lvis = vec.scale(sim.state.L, L_LENGTH / sim.Lref);
  setArrow(arrows.L, [0, 0, 0], Lvis, checks.showL.checked);
  setArrow(arrows.tau, checks.showL.checked ? Lvis : [0, 0, 0], vec.scale(T.total, (L_LENGTH * TAU_DT) / sim.Lref), checks.showTau.checked);
  setArrow(arrows.w, [0, 0, 0], vec.scale(o.omega, 0.3 / sim.Wref), checks.showW.checked);
  setArrow(arrows.g, vec.scale(o.e, params.arm), [0, -0.012 * params.g, 0], checks.showG.checked && params.g > 0);
  refreshTrail();

  controls.update();
  renderer.render(scene, camera);

  hudTimer += dt;
  if (hudTimer > 0.1) {
    hudTimer = 0;
    updateTexts();
    drawChart();
  }
  requestAnimationFrame(frame);
}

function resize() {
  const { clientWidth: w, clientHeight: h } = viewport;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // Keep the gyroscope in frame on narrow (portrait) screens.
  camera.fov = w / h < 1 ? 2 * Math.atan(Math.tan(20 * DEG) / (w / h)) / DEG : 40;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

// ---------- readouts ----------

const hud = document.getElementById('hud');
const fmt = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const joules = (v) => (Math.abs(v) >= 1 ? `${fmt(v, 2)} J` : `${fmt(v * 1000, 1)} mJ`);

function updateTexts() {
  const p = physicsParams();
  const o = observe(sim.state, p);
  const { I1, I3 } = inertia(p);
  const theory = steadyPrecession(o.theta, o.spin, p);
  const approx = gyroscopicApprox(o.spin, p);
  const threshold = sleepingThreshold(p);
  const drift = o.energy - sim.E0;
  const damped = p.spinDamping > 0 || p.pivotDamping > 0;
  const torqueFree = p.g === 0 || p.arm === 0;
  hud.innerHTML = `
    <dt>시간</dt><dd>${fmt(sim.t, 2)} s${params.timeScale !== 1 ? ` <span class="slow">×${fmt(params.timeScale, 2)}</span>` : ''}</dd>
    <dt>기울기 θ</dt><dd>${fmt(o.theta / DEG, 1)}°</dd>
    <dt>세차 속도 Ω</dt><dd>${fmt(o.phiDot, 2)} rad/s</dd>
    <dt>정상 세차 이론값</dt><dd>${torqueFree ? '토크 없음' : theory === null || Math.abs(o.spin) < 1 ? '해 없음' : `${fmt(theory, 2)} rad/s`}</dd>
    <dt>스핀 ω<sub>s</sub></dt><dd>${fmt(o.spin, 1)} rad/s</dd>
    <dt>장동 주파수 ≈</dt><dd>${fmt(nutationRate(o.w3, p) / (2 * Math.PI), 2)} Hz</dd>
    <dt>|L|</dt><dd>${fmt(vec.norm(sim.state.L) * 1000, 1)} g·m²/s</dd>
    <dt>${damped ? '마찰로 잃은 에너지' : '에너지 오차'}</dt><dd class="${!damped && Math.abs(drift) > 1e-3 ? 'warn' : ''}">${damped ? joules(drift) : `${fmt(drift * 1000, 3)} mJ`}</dd>`;

  document.getElementById('liveFormula').innerHTML = torqueFree
    ? '지금은 중력 토크가 0이라 L이 고정됩니다. 세차는 토크가 아니라 처음 조건에서만 생깁니다.'
    : `m g l = ${fmt(p.mass, 1)}×${fmt(p.g, 2)}×${fmt(p.arm, 3)} = <b>${fmt(p.mass * p.g * p.arm, 3)}</b> N·m<br>
       I<sub>s</sub> ω<sub>s</sub> = ${fmt(I3 * 1000, 2)}×10⁻³ × ${fmt(o.spin, 1)} = <b>${fmt(I3 * o.spin, 4)}</b> kg·m²/s<br>
       근사 Ω = <b>${fmt(approx, 2)}</b> rad/s · 정확한 해 = <b>${theory === null ? '없음' : fmt(theory, 2)}</b> rad/s · 측정값 = <b>${fmt(o.phiDot, 2)}</b> rad/s<br>
       한 바퀴 도는 데 ${Number.isFinite(approx) && approx !== 0 ? fmt((2 * Math.PI) / Math.abs(approx), 2) : '∞'} 초`;
  document.getElementById('liveNutation').innerHTML = `빠른 팽이 근사: 장동 각속도 ≈ I<sub>s</sub>ω₃ / I<sub>⊥</sub> = <b>${fmt(nutationRate(o.w3, p), 1)}</b> rad/s (${fmt(nutationRate(o.w3, p) / (2 * Math.PI), 2)} Hz)`;
  document.getElementById('liveSleep').innerHTML = `수직으로 선 팽이가 안정하려면 ω₃ &gt; √(4 I<sub>⊥</sub> m g l) / I<sub>s</sub> = <b>${fmt(threshold, 1)}</b> rad/s · 지금 ω₃ = <b>${fmt(Math.abs(o.w3), 1)}</b> rad/s ${Math.abs(o.w3) > threshold ? '✅ 안정' : '⚠️ 불안정'}`;
}

// ---------- chart ----------

const chart = document.getElementById('chart');
const ctx = chart.getContext('2d');

function drawChart() {
  const ratio = Math.min(window.devicePixelRatio, 2);
  const w = chart.clientWidth;
  const h = chart.clientHeight;
  if (chart.width !== Math.round(w * ratio)) {
    chart.width = Math.round(w * ratio);
    chart.height = Math.round(h * ratio);
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const half = (h - 10) / 2;
  const span = historySeconds();
  const t1 = Math.max(sim.t, span);
  const t0 = t1 - span;
  const hist = sim.history.filter((s) => s.t >= t0);
  const left = 46;
  const x = (t) => left + ((t - t0) / span) * (w - left - 8);

  const panel = (top, key, color, title, unit, reference) => {
    const values = hist.map((s) => s[key]).filter(Number.isFinite);
    let lo = Math.min(...values, reference ?? Infinity);
    let hi = Math.max(...values, reference ?? -Infinity);
    if (!values.length) [lo, hi] = [0, 1];
    const pad = Math.max((hi - lo) * 0.15, key === 'theta' ? 1 : 0.2);
    lo -= pad;
    hi += pad;
    const y = (v) => top + 16 + (1 - (v - lo) / (hi - lo)) * (half - 24);
    ctx.strokeStyle = '#2a3442';
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top + 16, w - left - 8, half - 24);
    ctx.fillStyle = '#8b98a8';
    ctx.font = '11px -apple-system, sans-serif';
    ctx.fillText(title, left, top + 11);
    ctx.textAlign = 'right';
    ctx.fillText(`${hi.toFixed(1)}${unit}`, left - 4, top + 24);
    ctx.fillText(`${lo.toFixed(1)}${unit}`, left - 4, top + half - 8);
    ctx.textAlign = 'left';
    if (reference !== null && Number.isFinite(reference) && reference > lo && reference < hi) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = '#58a6ff88';
      ctx.beginPath();
      ctx.moveTo(left, y(reference));
      ctx.lineTo(w - 8, y(reference));
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let started = false;
    for (const s of hist) {
      if (!Number.isFinite(s[key])) continue;
      if (started) ctx.lineTo(x(s.t), y(s[key]));
      else ctx.moveTo(x(s.t), y(s[key]));
      started = true;
    }
    ctx.stroke();
  };

  const p = physicsParams();
  const o = observe(sim.state, p);
  const theory = p.g === 0 || p.arm === 0 ? null : steadyPrecession(o.theta, o.spin, p);
  panel(0, 'theta', '#c792ea', '기울기 θ (클수록 아래로 처짐) — 흔들림이 장동', '°', null);
  panel(half + 10, 'phiDot', '#ff9f1c', '세차 속도 Ω = φ̇ — 점선: 정상 세차 이론값', '', theory);
}

// ---------- start ----------

applyPreset('steady');
requestAnimationFrame(frame);
window.gyro = { sim, params, applyPreset, reset };
