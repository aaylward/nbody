/**
 * Tests for temporal interpolation functions
 */

import { describe, it, expect } from 'vitest';
import {
  interpolateParticles,
  interpolateParticlesSmooth,
  calculateInterpolationAlpha,
} from '../interpolation';
import { createParticleArray, setParticle } from '../../particleData';

describe('interpolateParticles', () => {
  it('should return frame0 when alpha is 0', () => {
    const frame0 = createParticleArray(2);
    const frame1 = createParticleArray(2);

    setParticle(frame0, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
    setParticle(frame1, 0, { x: 10, y: 10, z: 10, vx: 1, vy: 1, vz: 1, mass: 1 });

    const result = interpolateParticles(frame0, frame1, 0);

    expect(result[0]).toBeCloseTo(0); // x
    expect(result[1]).toBeCloseTo(0); // y
    expect(result[2]).toBeCloseTo(0); // z
  });

  it('should return frame1 when alpha is 1', () => {
    const frame0 = createParticleArray(2);
    const frame1 = createParticleArray(2);

    setParticle(frame0, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
    setParticle(frame1, 0, { x: 10, y: 10, z: 10, vx: 1, vy: 1, vz: 1, mass: 1 });

    const result = interpolateParticles(frame0, frame1, 1);

    expect(result[0]).toBeCloseTo(10); // x
    expect(result[1]).toBeCloseTo(10); // y
    expect(result[2]).toBeCloseTo(10); // z
  });

  it('should interpolate halfway when alpha is 0.5', () => {
    const frame0 = createParticleArray(1);
    const frame1 = createParticleArray(1);

    setParticle(frame0, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
    setParticle(frame1, 0, { x: 10, y: 20, z: 30, vx: 2, vy: 4, vz: 6, mass: 1 });

    const result = interpolateParticles(frame0, frame1, 0.5);

    expect(result[0]).toBeCloseTo(5); // x
    expect(result[1]).toBeCloseTo(10); // y
    expect(result[2]).toBeCloseTo(15); // z
    expect(result[3]).toBeCloseTo(1); // vx
    expect(result[4]).toBeCloseTo(2); // vy
    expect(result[5]).toBeCloseTo(3); // vz
  });

  it('should handle multiple particles', () => {
    const frame0 = createParticleArray(3);
    const frame1 = createParticleArray(3);

    for (let i = 0; i < 3; i++) {
      setParticle(frame0, i, { x: i * 10, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(frame1, i, { x: i * 10 + 5, y: 5, z: 5, vx: 1, vy: 1, vz: 1, mass: 1 });
    }

    const result = interpolateParticles(frame0, frame1, 0.5);

    // Check first particle
    expect(result[0]).toBeCloseTo(2.5); // x
    expect(result[1]).toBeCloseTo(2.5); // y

    // Check second particle
    expect(result[7]).toBeCloseTo(12.5); // x
    expect(result[8]).toBeCloseTo(2.5); // y
  });

  it('should throw error if arrays have different lengths', () => {
    const frame0 = createParticleArray(2);
    const frame1 = createParticleArray(3);

    expect(() => interpolateParticles(frame0, frame1, 0.5)).toThrow('Frame arrays must have the same length');
  });
});

describe('interpolateParticlesSmooth', () => {
  it('should use Hermite interpolation', () => {
    const frame0 = createParticleArray(1);
    const frame1 = createParticleArray(1);

    setParticle(frame0, 0, { x: 0, y: 0, z: 0, vx: 10, vy: 0, vz: 0, mass: 1 });
    setParticle(frame1, 0, { x: 10, y: 0, z: 0, vx: 10, vy: 0, vz: 0, mass: 1 });

    const result = interpolateParticlesSmooth(frame0, frame1, 0.5);

    // With Hermite interpolation and velocity, the midpoint should be influenced by velocities
    expect(result[0]).toBeGreaterThan(4); // x should be > 4 due to velocity influence
    expect(result[0]).toBeLessThan(6); // but still < 6
  });

  it('should preserve mass during interpolation', () => {
    const frame0 = createParticleArray(1);
    const frame1 = createParticleArray(1);

    setParticle(frame0, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 5 });
    setParticle(frame1, 0, { x: 10, y: 10, z: 10, vx: 1, vy: 1, vz: 1, mass: 5 });

    const result = interpolateParticlesSmooth(frame0, frame1, 0.5);

    expect(result[6]).toBe(5); // mass unchanged
  });
});

describe('calculateInterpolationAlpha', () => {
  it('should return 0 when time equals last update', () => {
    const alpha = calculateInterpolationAlpha(1000, 1000, 50);
    expect(alpha).toBe(0);
  });

  it('should return 0.5 halfway through interval', () => {
    const alpha = calculateInterpolationAlpha(1000, 1025, 50);
    expect(alpha).toBeCloseTo(0.5);
  });

  it('should return 1 at end of interval', () => {
    const alpha = calculateInterpolationAlpha(1000, 1050, 50);
    expect(alpha).toBeCloseTo(1.0);
  });

  it('should clamp alpha to [0, 1]', () => {
    const alphaNegative = calculateInterpolationAlpha(1000, 900, 50);
    expect(alphaNegative).toBe(0);

    const alphaOver = calculateInterpolationAlpha(1000, 1200, 50);
    expect(alphaOver).toBe(1);
  });
});
