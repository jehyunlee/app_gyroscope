import test from 'node:test';
import assert from 'node:assert/strict';
import {
  initialState,
  observe,
  simulate,
  steadyPrecession,
  gyroscopicApprox,
  sleepingThreshold,
  inertia,
  vec,
} from '../js/physics.js';
import { createWorld, stepWorld, observeWorld, detach, freeStep, STAND } from '../js/world.js';

const runWorld = (world, p, seconds) => {
  for (let t = 0; t < seconds; t += 0.05) stepWorld(world, p, 0.05);
  return world;
};

const base = { mass: 1, radius: 0.1, arm: 0.15, shape: 'ring', g: 9.81, spinDamping: 0, pivotDamping: 0 };
const deg = Math.PI / 180;

test('initial conditions round-trip through the observer', () => {
  const ic = { theta: 60 * deg, phi: 0.7, spin: 80, phiDot: 1.3, thetaDot: -0.4 };
  const o = observe(initialState(ic, base), base);
  assert.ok(Math.abs(o.theta - ic.theta) < 1e-9);
  assert.ok(Math.abs(o.phi - ic.phi) < 1e-9);
  assert.ok(Math.abs(o.phiDot - ic.phiDot) < 1e-9);
  assert.ok(Math.abs(o.thetaDot - ic.thetaDot) < 1e-9);
  assert.ok(Math.abs(o.spin - ic.spin) < 1e-9);
});

test('horizontal steady precession matches Ω = m g l / (I3 ω_s)', () => {
  const spin = 150;
  const phiDot = steadyPrecession(Math.PI / 2, spin, base);
  assert.ok(Math.abs(phiDot - gyroscopicApprox(spin, base)) < 1e-12);
  const start = initialState({ theta: Math.PI / 2, spin, phiDot, thetaDot: 0 }, base);
  const { state } = simulate(start, base, 3);
  const o = observe(state, base);
  assert.ok(Math.abs(o.theta - Math.PI / 2) < 1e-4, `θ drifted to ${o.theta}`);
  assert.ok(phiDot * 3 < Math.PI);
  assert.ok(Math.abs(o.phi - phiDot * 3) < 1e-3, `φ = ${o.phi}, expected ${phiDot * 3}`);
});

test('tilted steady precession holds θ using the exact slow root', () => {
  const theta = 50 * deg;
  const spin = 60;
  const phiDot = steadyPrecession(theta, spin, base);
  const { state } = simulate(initialState({ theta, spin, phiDot, thetaDot: 0 }, base), base, 4);
  assert.ok(Math.abs(observe(state, base).theta - theta) < 1e-4);
});

test('released from rest it nutates with cusps and never rises above the start', () => {
  const theta = 70 * deg;
  let s = initialState({ theta, spin: 60, phiDot: 0, thetaDot: 0 }, base);
  let minTheta = Infinity;
  let maxTheta = -Infinity;
  let minPhiDot = Infinity;
  for (let i = 0; i < 400; i++) {
    s = simulate(s, base, 0.005).state;
    const o = observe(s, base);
    minTheta = Math.min(minTheta, o.theta);
    maxTheta = Math.max(maxTheta, o.theta);
    minPhiDot = Math.min(minPhiDot, o.phiDot);
  }
  assert.ok(minTheta > theta - 1e-4, 'energy forbids rising above the release angle');
  assert.ok(maxTheta - theta > 1 * deg, 'the axle dips (nutation)');
  assert.ok(minPhiDot > -1e-3, 'precession never reverses for cusps');
});

test('frictionless motion conserves energy and vertical angular momentum', () => {
  const s0 = initialState({ theta: 40 * deg, spin: 90, phiDot: -2, thetaDot: 1 }, base);
  const o0 = observe(s0, base);
  const { state } = simulate(s0, base, 5);
  const o1 = observe(state, base);
  assert.ok(Math.abs(o1.energy - o0.energy) / Math.abs(o0.energy) < 1e-6);
  assert.ok(Math.abs(o1.Lvert - o0.Lvert) < 1e-8);
  assert.ok(Math.abs(o1.w3 - o0.w3) < 1e-8, 'spin component ω3 is conserved');
});

test('without spin the wheel falls like a pendulum', () => {
  const { state } = simulate(initialState({ theta: 60 * deg, spin: 0, phiDot: 0, thetaDot: 0 }, base), base, 0.3);
  assert.ok(observe(state, base).theta > 100 * deg);
});

test('without gravity the angular momentum vector is fixed in space', () => {
  const p = { ...base, g: 0 };
  const s0 = initialState({ theta: 30 * deg, spin: 40, phiDot: 3, thetaDot: 0.5 }, p);
  const { state } = simulate(s0, p, 4);
  assert.ok(vec.norm(vec.sub(state.L, s0.L)) < 1e-12);
});

test('sleeping top is stable above the threshold spin and falls below it', () => {
  const threshold = sleepingThreshold(base);
  const run = (spin) => {
    const s = initialState({ theta: 2 * deg, spin, phiDot: 0, thetaDot: 0 }, base);
    let maxTheta = 0;
    let cur = s;
    for (let i = 0; i < 200; i++) {
      cur = simulate(cur, base, 0.02).state;
      maxTheta = Math.max(maxTheta, observe(cur, base).theta);
    }
    return maxTheta;
  };
  assert.ok(run(threshold * 1.3) < 5 * deg);
  assert.ok(run(threshold * 0.7) > 20 * deg);
});

test('friction slows the spin and lets the gyroscope sag', () => {
  const p = { ...base, spinDamping: 5e-4, pivotDamping: 2e-3 };
  const phiDot = steadyPrecession(70 * deg, 100, p);
  const s0 = initialState({ theta: 70 * deg, spin: 100, phiDot, thetaDot: 0 }, p);
  const { state } = simulate(s0, p, 4);
  const o = observe(state, p);
  assert.ok(o.w3 < observe(s0, p).w3 * 0.9);
  assert.ok(o.theta > 70 * deg + 1 * deg);
  assert.ok(o.energy < observe(s0, p).energy);
});

test('inertia follows the ring and disc formulas', () => {
  const ring = inertia(base);
  assert.ok(Math.abs(ring.I3 - 0.01) < 1e-15 && Math.abs(ring.I1 - (0.005 + 0.0225)) < 1e-15);
  const disc = inertia({ ...base, shape: 'disc' });
  assert.ok(Math.abs(disc.I3 - 0.005) < 1e-15 && Math.abs(disc.I1 - (0.0025 + 0.0225)) < 1e-15);
});

test('a weak spin (1 rad/s) hits the post, leaves the pivot and comes to rest on the floor', () => {
  const world = createWorld(initialState({ theta: 70 * deg, spin: 1, phiDot: 1.48, thetaDot: 0 }, base), base);
  runWorld(world, base, 0.5);
  assert.ok(world.detachedAt > 0.1 && world.detachedAt < 0.4, `released at ${world.detachedAt}`);
  runWorld(world, base, 5);
  const o = observeWorld(world, base);
  assert.equal(o.phase, 'rest');
  assert.ok(o.center[1] < STAND.ground + 0.1, 'lying low, on the floor or the base');
  assert.ok(o.energy < observeWorld(createWorld(initialState({ theta: 70 * deg, spin: 1, phiDot: 1.48, thetaDot: 0 }, base), base), base).energy);
});

test('a fast gyroscope in steady precession never touches the stand', () => {
  const phiDot = steadyPrecession(70 * deg, 100, base);
  const world = runWorld(createWorld(initialState({ theta: 70 * deg, spin: 100, phiDot, thetaDot: 0 }, base), base), base, 5);
  assert.equal(world.phase, 'pivot');
  assert.ok(Math.abs(observeWorld(world, base).theta - 70 * deg) < 1e-3);
});

test('friction slowly drains the spin until the gyroscope sags onto the post and falls off', () => {
  const p = { ...base, spinDamping: 6e-4, pivotDamping: 4e-3 };
  const phiDot = steadyPrecession(80 * deg, 120, p);
  const world = runWorld(createWorld(initialState({ theta: 80 * deg, spin: 120, phiDot, thetaDot: 0 }, p), p), p, 45);
  assert.ok(world.detachedAt > 10, `held on for a while, released at ${world.detachedAt}`);
  assert.equal(world.phase, 'rest');
});

test('leaving the pivot keeps the velocity, energy and angular momentum of the motion', () => {
  const s = initialState({ theta: 130 * deg, spin: 40, phiDot: 2, thetaDot: 3 }, base);
  const body = detach(s, base);
  const o = observe(s, base);
  const kinetic = 0.5 * base.mass * vec.dot(body.v, body.v) + 0.5 * vec.dot(body.L, o.omega);
  assert.ok(Math.abs(kinetic - o.kinetic) < 1e-12);
  assert.ok(vec.norm(vec.sub(body.v, vec.cross(o.omega, body.x))) < 1e-12);
});

test('in free flight the centre of mass follows a parabola and L about it is constant', () => {
  const s = initialState({ theta: 40 * deg, spin: 60, phiDot: 3, thetaDot: 1 }, base);
  const body = detach(s, base);
  body.x = [0, 2, 0];
  const L0 = body.L;
  const v0 = body.v;
  for (let i = 0; i < 200; i++) freeStep(body, base, 1e-3);
  assert.ok(vec.norm(vec.sub(body.L, L0)) < 1e-12);
  const expected = [v0[0] * 0.2, 2 + v0[1] * 0.2 - 0.5 * base.g * 0.04, v0[2] * 0.2];
  assert.ok(vec.norm(vec.sub(body.x, expected)) < 2e-3);
});
