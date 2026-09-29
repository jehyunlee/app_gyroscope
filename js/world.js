// The gyroscope in its surroundings: a stand (post, base) and the floor.
//
// Phase 'pivot': the axle tip rests in the pivot cup on top of the post, and
// the heavy-top equations of physics.js apply. The cup only holds the tip; it
// cannot hold the wheel away from the post. So the first time any part of the
// wheel or axle touches the stand, the gyroscope leaves the pivot.
//
// Phase 'free': a free rigid body (wheel + massless axle) under gravity,
// bouncing and sliding on the stand and floor with impulse-based contacts.
// Phase 'rest': it has come to rest on the floor.

import { SHAPES, inertia, axisOf, omegaOf, torques, rk4Step, stableStep, axisKinematics, observe as observePivot, quat, vec } from './physics.js';

const Y = [0, 1, 0];

// Stand geometry, shared with the renderer so what you see is what collides.
export const STAND = {
  pivotRadius: 0.02,
  postRadius: 0.014,
  baseRadius: 0.13,
  baseHeight: 0.03,
  ground: -0.62,
  // A low wall around the floor, like a tray, keeps a rolling wheel in view.
  trayRadius: 1.1,
  trayHeight: 0.04,
};

const RESTITUTION = 0.3;
const FRICTION = 0.5;
const ROLLING_DAMPING = 1.5; // 1/s, only while touching something

// Visual sizes of the moving parts, also shared with the renderer.
export function partSizes(p) {
  const disc = p.shape === 'disc';
  return {
    tube: disc ? p.radius * 0.06 : p.radius * 0.1,
    plate: disc ? p.radius * 0.036 : 0,
    rodRadius: 0.006,
    rodStart: -0.02,
    rodEnd: p.arm + 0.03,
  };
}

// Collision spheres on the wheel and axle, in body coordinates relative to the pivot
// (body +y is the axle). `nearPivot` marks the axle piece that sits in the cup:
// it touches the post by construction and must not trigger a release.
const sampleCache = new Map();
export function contactSamples(p) {
  const key = `${p.shape}|${p.radius}|${p.arm}`;
  if (sampleCache.has(key)) return sampleCache.get(key);
  const s = partSizes(p);
  const samples = [];
  const ring = (radius, count, size) => {
    for (let i = 0; i < count; i++) {
      const a = (i / count) * 2 * Math.PI;
      samples.push({ at: [radius * Math.cos(a), p.arm, radius * Math.sin(a)], r: size });
    }
  };
  ring(p.radius, 40, s.tube);
  if (p.shape === 'disc') ring(p.radius * 0.55, 20, Math.max(s.plate, p.radius * 0.45 * 0.5));
  samples.push({ at: [0, p.arm, 0], r: p.radius * 0.14 });
  const pieces = 8;
  for (let i = 0; i <= pieces; i++) {
    const y = s.rodStart + ((s.rodEnd - s.rodStart) * i) / pieces;
    samples.push({ at: [0, y, 0], r: s.rodRadius, nearPivot: y < STAND.pivotRadius + 0.012 });
  }
  sampleCache.set(key, samples);
  return samples;
}

// Penetrations of a sphere (centre c, radius r) into the stand and floor.
function obstaclesHit(c, r, out, sample) {
  const topOfBase = STAND.ground + STAND.baseHeight;
  if (c[1] - r < STAND.ground) out.push({ sample, normal: Y, depth: STAND.ground - (c[1] - r), floor: true });
  const radial = Math.hypot(c[0], c[2]);
  if (radial < STAND.baseRadius + r && c[1] - r < topOfBase && c[1] > STAND.ground) {
    const top = topOfBase - (c[1] - r);
    const side = STAND.baseRadius + r - radial;
    if (top <= side || radial < 1e-9) out.push({ sample, normal: Y, depth: top, floor: true });
    else out.push({ sample, normal: [c[0] / radial, 0, c[2] / radial], depth: side });
  }
  if (radial > STAND.trayRadius - r && c[1] - r < STAND.ground + STAND.trayHeight) {
    out.push({ sample, normal: [-c[0] / radial, 0, -c[2] / radial], depth: radial - (STAND.trayRadius - r) });
  }
  // Post: a capsule from the base up to the pivot centre, capped by the pivot ball.
  const y = Math.max(topOfBase, Math.min(0, c[1]));
  const d = vec.sub(c, [0, y, 0]);
  const dist = vec.norm(d);
  const reach = (y === 0 ? STAND.pivotRadius : STAND.postRadius) + r;
  if (dist < reach) {
    const normal = dist > 1e-9 ? vec.scale(d, 1 / dist) : Y;
    out.push({ sample, normal, depth: reach - dist });
  }
}

// Does the pivoted gyroscope touch the stand anywhere other than at the cup?
export function touchesStand(q, p) {
  const hits = [];
  for (const s of contactSamples(p)) {
    if (s.nearPivot) continue;
    obstaclesHit(quat.rotate(q, s.at), s.r, hits, s);
    if (hits.length) return true;
  }
  return false;
}

// ---------- free rigid body ----------
// State: centre of mass x, velocity v, orientation q, angular momentum about the COM L.

function comInertia(p) {
  const shape = SHAPES[p.shape];
  return { A: shape.i1 * p.mass * p.radius * p.radius, C: shape.i3 * p.mass * p.radius * p.radius };
}

function applyInverseInertia(q, vector, p) {
  const { A, C } = comInertia(p);
  const e = axisOf(q);
  const along = vec.dot(vector, e);
  return vec.add(vec.scale(vec.sub(vector, vec.scale(e, along)), 1 / A), vec.scale(e, along / C));
}

export const freeOmega = (body, p) => applyInverseInertia(body.q, body.L, p);

// Leaving the pivot: same motion, now described about the centre of mass.
export function detach(state, p) {
  const e = axisOf(state.q);
  const omega = omegaOf(state, p);
  const x = vec.scale(e, p.arm);
  const v = vec.cross(omega, x);
  // L_pivot = L_com + x × m v
  const L = vec.sub(state.L, vec.cross(x, vec.scale(v, p.mass)));
  return { x, v, q: state.q, L };
}

function resolveContact(body, contact, p, restitution) {
  const r = vec.sub(contact.point, body.x);
  const omega = freeOmega(body, p);
  const velocity = vec.add(body.v, vec.cross(omega, r));
  const vn = vec.dot(velocity, contact.normal);
  if (vn >= 0) return;
  const response = (direction) => {
    const angular = vec.cross(applyInverseInertia(body.q, vec.cross(r, direction), p), r);
    return 1 / p.mass + vec.dot(angular, direction);
  };
  const bounce = vn < -0.3 ? restitution : 0;
  const jn = (-(1 + bounce) * vn) / response(contact.normal);
  let impulse = vec.scale(contact.normal, jn);
  const tangential = vec.sub(velocity, vec.scale(contact.normal, vn));
  const speed = vec.norm(tangential);
  if (speed > 1e-9) {
    const t = vec.scale(tangential, 1 / speed);
    const jt = Math.min(speed / response(t), FRICTION * jn);
    impulse = vec.sub(impulse, vec.scale(t, jt));
  }
  body.v = vec.add(body.v, vec.scale(impulse, 1 / p.mass));
  body.L = vec.add(body.L, vec.cross(r, impulse));
}

function freeContacts(body, p) {
  const hits = [];
  for (const s of contactSamples(p)) {
    const c = vec.add(body.x, quat.rotate(body.q, vec.sub(s.at, [0, p.arm, 0])));
    const before = hits.length;
    obstaclesHit(c, s.r, hits, s);
    for (let i = before; i < hits.length; i++) hits[i].point = vec.sub(c, vec.scale(hits[i].normal, s.r));
  }
  return hits;
}

export function freeStep(body, p, h) {
  body.v = vec.add(body.v, [0, -p.g * h, 0]);
  const contacts = freeContacts(body, p);
  for (let iteration = 0; iteration < 4; iteration++) {
    for (const c of contacts) resolveContact(body, c, p, iteration === 0 ? RESTITUTION : 0);
  }
  if (contacts.length) {
    const keep = Math.exp(-ROLLING_DAMPING * h);
    body.v = vec.scale(body.v, keep);
    body.L = vec.scale(body.L, keep);
    // Push out of the deepest overlap so resting contact does not sink.
    const deepest = contacts.reduce((a, b) => (b.depth > a.depth ? b : a));
    body.x = vec.add(body.x, vec.scale(deepest.normal, Math.max(0, deepest.depth - 5e-4) * 0.6));
  }
  body.x = vec.add(body.x, vec.scale(body.v, h));
  const omega = freeOmega(body, p);
  const angle = vec.norm(omega) * h;
  if (angle > 0) body.q = quat.normalize(quat.mul(quat.axisAngle(vec.scale(omega, 1 / vec.norm(omega)), angle), body.q));
  return contacts.some((c) => c.floor);
}

function freeKinetic(body, p) {
  return 0.5 * p.mass * vec.dot(body.v, body.v) + 0.5 * vec.dot(body.L, freeOmega(body, p));
}

// ---------- the world ----------

export function createWorld(state, p) {
  const world = { phase: 'pivot', t: 0, state, body: null, calm: 0, detachedAt: null };
  if (touchesStand(state.q, p)) release(world, p);
  return world;
}

function release(world, p) {
  world.body = detach(world.state, p);
  world.phase = 'free';
  world.detachedAt = world.t;
}

export function stepWorld(world, p, duration) {
  let remaining = duration;
  let guard = 0;
  while (remaining > 1e-9 && guard < 8000 && world.phase !== 'rest') {
    guard += 1;
    if (world.phase === 'pivot') {
      const h = Math.min(stableStep(world.state, p), remaining);
      world.state = rk4Step(world.state, p, h);
      world.t += h;
      remaining -= h;
      if (touchesStand(world.state.q, p)) release(world, p);
      continue;
    }
    const spin = vec.norm(freeOmega(world.body, p));
    const h = Math.min(1e-3, 0.03 / Math.max(spin, 1), remaining);
    const onFloor = freeStep(world.body, p, h);
    world.t += h;
    remaining -= h;
    world.calm = onFloor && freeKinetic(world.body, p) < 2e-4 * p.mass ? world.calm + h : 0;
    if (world.calm > 0.4) {
      world.body.v = [0, 0, 0];
      world.body.L = [0, 0, 0];
      world.phase = 'rest';
    }
  }
  return world;
}

// A live change of m, r, l, shape or g keeps the current angular velocity.
export function retune(world, before, after) {
  if (world.phase === 'pivot') {
    const omega = omegaOf(world.state, before);
    const e = axisOf(world.state.q);
    const w3 = vec.dot(omega, e);
    const { I1, I3 } = inertia(after);
    world.state = { q: world.state.q, L: vec.add(vec.scale(vec.sub(omega, vec.scale(e, w3)), I1), vec.scale(e, I3 * w3)) };
    return;
  }
  const omega = freeOmega(world.body, before);
  const e = axisOf(world.body.q);
  const w3 = vec.dot(omega, e);
  const { A, C } = comInertia(after);
  world.body.L = vec.add(vec.scale(vec.sub(omega, vec.scale(e, w3)), A), vec.scale(e, C * w3));
  if (world.phase === 'rest' && before.shape === after.shape && before.radius === after.radius) return;
  if (world.phase === 'rest') world.phase = 'free';
}

// Everything the display needs, in one shape for all phases.
export function observeWorld(world, p) {
  if (world.phase === 'pivot') {
    const o = observePivot(world.state, p);
    return { ...o, phase: 'pivot', center: vec.scale(o.e, p.arm), q: world.state.q, L: world.state.L, torque: torques(world.state, p).total, height: o.e[1] * p.arm };
  }
  const body = world.body;
  const e = axisOf(body.q);
  const omega = freeOmega(body, p);
  const kinetic = freeKinetic(body, p);
  const potential = p.mass * p.g * body.x[1];
  return {
    phase: world.phase,
    e,
    omega,
    ...axisKinematics(e, omega),
    center: body.x,
    q: body.q,
    L: body.L,
    torque: [0, 0, 0],
    kinetic,
    potential,
    energy: kinetic + potential,
    height: body.x[1],
  };
}
