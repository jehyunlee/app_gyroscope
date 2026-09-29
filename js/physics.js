// Heavy symmetric gyroscope on a fixed frictionless (or damped) pivot.
//
// World frame is y-up (same as three.js). The pivot sits at the origin.
// The wheel's symmetry axis is the body +y axis; e = q·ŷ in world space.
// State is (q, L): orientation quaternion and angular momentum about the pivot,
// both in world coordinates. Integrating L instead of Euler angles keeps the
// model free of the θ = 0 singularity, so sleeping tops and loops work too.

export const vec = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  norm: (a) => Math.hypot(a[0], a[1], a[2]),
};

// Quaternions are [x, y, z, w].
export const quat = {
  mul(a, b) {
    const [ax, ay, az, aw] = a;
    const [bx, by, bz, bw] = b;
    return [
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
      aw * bw - ax * bx - ay * by - az * bz,
    ];
  },
  axisAngle(axis, angle) {
    const s = Math.sin(angle / 2);
    return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
  },
  normalize(q) {
    const n = Math.hypot(q[0], q[1], q[2], q[3]);
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  },
  rotate(q, v) {
    const [x, y, z, w] = q;
    const t = vec.scale(vec.cross([x, y, z], v), 2);
    return vec.add(vec.add(v, vec.scale(t, w)), vec.cross([x, y, z], t));
  },
};

const Y = [0, 1, 0];

export const SHAPES = {
  // Thin ring: all mass on the rim. Thin uniform disc: mass spread evenly.
  ring: { label: '얇은 고리 (림에 질량 집중)', i3: 1, i1: 0.5 },
  disc: { label: '얇은 원판 (균일한 질량)', i3: 0.5, i1: 0.25 },
};

// Moments of inertia about the pivot. I3: symmetry axis. I1: any axis ⟂ to it
// through the pivot (parallel-axis theorem adds m·l²). Rod mass is neglected,
// as in the reference derivation.
export function inertia(p) {
  const shape = SHAPES[p.shape];
  const I3 = shape.i3 * p.mass * p.radius * p.radius;
  const I1 = shape.i1 * p.mass * p.radius * p.radius + p.mass * p.arm * p.arm;
  return { I1, I3 };
}

export function axisOf(q) {
  return quat.rotate(q, Y);
}

// Angular velocity from angular momentum: ω = L⊥/I1 + (L·e/I3) e.
export function omegaOf(state, p) {
  const { I1, I3 } = inertia(p);
  const e = axisOf(state.q);
  const L3 = vec.dot(state.L, e);
  const Lperp = vec.sub(state.L, vec.scale(e, L3));
  return vec.add(vec.scale(Lperp, 1 / I1), vec.scale(e, L3 / I3));
}

export function momentumFromOmega(q, omega, p) {
  const { I1, I3 } = inertia(p);
  const e = axisOf(q);
  const w3 = vec.dot(omega, e);
  const wperp = vec.sub(omega, vec.scale(e, w3));
  return vec.add(vec.scale(wperp, I1), vec.scale(e, I3 * w3));
}

// Gravity acts at the centre of the wheel, l·e from the pivot: τ = l e × (−m g ŷ).
export function gravityTorque(q, p) {
  return vec.cross(vec.scale(axisOf(q), p.arm), [0, -p.mass * p.g, 0]);
}

export function torques(state, p) {
  const e = axisOf(state.q);
  const omega = omegaOf(state, p);
  const w3 = vec.dot(omega, e);
  const wperp = vec.sub(omega, vec.scale(e, w3));
  const gravity = gravityTorque(state.q, p);
  const spinFriction = vec.scale(e, -p.spinDamping * w3);
  const pivotFriction = vec.scale(wperp, -p.pivotDamping);
  return { gravity, spinFriction, pivotFriction, total: vec.add(gravity, vec.add(spinFriction, pivotFriction)) };
}

function derivative(state, p) {
  const omega = omegaOf(state, p);
  const dq = quat.mul([omega[0] / 2, omega[1] / 2, omega[2] / 2, 0], state.q);
  return { dq, dL: torques(state, p).total };
}

function advance(state, d, h) {
  return {
    q: state.q.map((v, i) => v + d.dq[i] * h),
    L: state.L.map((v, i) => v + d.dL[i] * h),
  };
}

export function rk4Step(state, p, h) {
  const k1 = derivative(state, p);
  const k2 = derivative(advance(state, k1, h / 2), p);
  const k3 = derivative(advance(state, k2, h / 2), p);
  const k4 = derivative(advance(state, k3, h), p);
  const q = state.q.map((v, i) => v + (h / 6) * (k1.dq[i] + 2 * k2.dq[i] + 2 * k3.dq[i] + k4.dq[i]));
  const L = state.L.map((v, i) => v + (h / 6) * (k1.dL[i] + 2 * k2.dL[i] + 2 * k3.dL[i] + k4.dL[i]));
  return { q: quat.normalize(q), L };
}

// Step size resolves the fastest motion (spin and nutation) with margin.
export function stableStep(state, p) {
  const { I1, I3 } = inertia(p);
  const omega = vec.norm(omegaOf(state, p));
  const nutation = Math.abs(vec.dot(state.L, axisOf(state.q))) / I1;
  const pendulum = Math.sqrt((p.mass * p.g * p.arm) / I1);
  const fastest = Math.max(omega, nutation, pendulum, 1);
  return Math.min(2e-3, 0.02 / fastest, (0.5 * I3) / Math.max(p.spinDamping, 1e-12), (0.5 * I1) / Math.max(p.pivotDamping, 1e-12));
}

export function simulate(state, p, duration) {
  let s = state;
  let t = 0;
  let steps = 0;
  while (t < duration - 1e-12) {
    const h = Math.min(stableStep(s, p), duration - t);
    s = rk4Step(s, p, h);
    t += h;
    steps += 1;
  }
  return { state: s, steps };
}

// Euler-angle initial conditions → (q, L).
// θ: tilt of the axle from vertical, φ: azimuth (precession angle),
// spin: ψ̇, the wheel's spin relative to the axle, φ̇, θ̇: precession and nutation rates.
export function initialState(ic, p) {
  const { theta, phi = 0, spin, phiDot, thetaDot } = ic;
  const q = quat.normalize(quat.mul(quat.axisAngle(Y, phi), quat.axisAngle([0, 0, 1], -theta)));
  const e = axisOf(q);
  const dEdTheta = quat.rotate(quat.axisAngle(Y, phi), [Math.cos(theta), -Math.sin(theta), 0]);
  const nodeAxis = vec.cross(e, dEdTheta);
  const omega = vec.add(vec.add(vec.scale(Y, phiDot), vec.scale(nodeAxis, thetaDot)), vec.scale(e, spin));
  return { q, L: momentumFromOmega(q, omega, p) };
}

// Observable angles and rates, computed from the state rather than stored.
export function observe(state, p) {
  const { I1, I3 } = inertia(p);
  const e = axisOf(state.q);
  const omega = omegaOf(state, p);
  const w3 = vec.dot(omega, e);
  const wperp = vec.sub(omega, vec.scale(e, w3));
  const horizontal = 1 - e[1] * e[1];
  const theta = Math.acos(Math.max(-1, Math.min(1, e[1])));
  // ė = ω × e, and e × ė = ω⊥, so φ̇ = ω⊥·ŷ / sin²θ.
  const phiDot = horizontal > 1e-8 ? wperp[1] / horizontal : 0;
  const thetaDot = horizontal > 1e-8 ? -vec.dot(vec.cross(omega, e), Y) / Math.sqrt(horizontal) : 0;
  const Lperp = vec.sub(state.L, vec.scale(e, vec.dot(state.L, e)));
  const kinetic = 0.5 * (vec.dot(Lperp, Lperp) / I1 + (vec.dot(state.L, e) ** 2) / I3);
  const potential = p.mass * p.g * p.arm * e[1];
  return {
    e,
    omega,
    w3,
    theta,
    phi: Math.atan2(-e[2], e[0]),
    phiDot,
    thetaDot,
    spin: w3 - phiDot * e[1],
    kinetic,
    potential,
    energy: kinetic + potential,
    Lvert: state.L[1],
  };
}

// Steady (uniform) precession at fixed θ with relative spin ψ̇.
// θ̈ = 0 requires  (I3 − I1) cosθ φ̇² + I3 ψ̇ φ̇ − m g l = 0.
// Returns the slow root (the one the gyroscope actually uses), or null if no real root.
export function steadyPrecession(theta, spin, p) {
  const { I1, I3 } = inertia(p);
  const a = (I3 - I1) * Math.cos(theta);
  const b = I3 * spin;
  const c = -p.mass * p.g * p.arm;
  if (Math.abs(a) < 1e-12) return b === 0 ? null : -c / b;
  const disc = b * b - 4 * a * c;
  if (disc < 0) return null;
  const root = Math.sqrt(disc);
  if (b === 0) return (-b + root) / (2 * a);
  // Numerically stable form of the root that → m g l / (I3 ψ̇) as spin grows.
  return (2 * -c) / (b + Math.sign(b) * root);
}

// The textbook fast-spin approximation: Ω ≈ m g l / (I3 ω_s).
export function gyroscopicApprox(spin, p) {
  const { I3 } = inertia(p);
  return spin === 0 ? Infinity : (p.mass * p.g * p.arm) / (I3 * spin);
}

// Fast-top nutation frequency ≈ I3 ω3 / I1.
export function nutationRate(w3, p) {
  const { I1, I3 } = inertia(p);
  return Math.abs((I3 * w3) / I1);
}

// A vertical ("sleeping") top is stable when I3² ω3² > 4 I1 m g l.
export function sleepingThreshold(p) {
  const { I1, I3 } = inertia(p);
  return Math.sqrt((4 * I1 * p.mass * p.g * Math.max(p.arm, 0)) / (I3 * I3));
}
