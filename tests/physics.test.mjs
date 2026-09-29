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

test('a weak spin (1 rad/s) falls; frictionless it swings back, with pivot friction it stays down', () => {
  const ic = { theta: 70 * deg, spin: 1, phiDot: 1.48, thetaDot: 0 };
  const free = simulate(initialState(ic, base), base, 0.3).state;
  assert.ok(observe(free, base).theta > 140 * deg, 'drops through horizontal within 0.3 s');
  const p = { ...base, pivotDamping: 0.015 };
  const damped = simulate(initialState(ic, p), p, 10).state;
  assert.ok(observe(damped, p).theta > 160 * deg, 'settles hanging down');
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
