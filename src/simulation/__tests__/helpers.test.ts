import { describe, it, expect } from 'vitest';
import { Particle } from '../../types';

// Helper function to test particle interpolation logic
function interpolateParticles(p1: Particle, p2: Particle, t: number): Particle {
  const oneMinusT = 1 - t;
  return {
    x: p1.x * oneMinusT + p2.x * t,
    y: p1.y * oneMinusT + p2.y * t,
    z: p1.z * oneMinusT + p2.z * t,
    vx: p1.vx * oneMinusT + p2.vx * t,
    vy: p1.vy * oneMinusT + p2.vy * t,
    vz: p1.vz * oneMinusT + p2.vz * t,
  };
}

describe('Particle Interpolation', () => {
  it('should return first particle when t=0', () => {
    const p1: Particle = { x: 0, y: 0, z: 0, vx: 1, vy: 1, vz: 1 };
    const p2: Particle = { x: 10, y: 10, z: 10, vx: 5, vy: 5, vz: 5 };

    const result = interpolateParticles(p1, p2, 0);

    expect(result.x).toBeCloseTo(p1.x);
    expect(result.y).toBeCloseTo(p1.y);
    expect(result.z).toBeCloseTo(p1.z);
  });

  it('should return second particle when t=1', () => {
    const p1: Particle = { x: 0, y: 0, z: 0, vx: 1, vy: 1, vz: 1 };
    const p2: Particle = { x: 10, y: 10, z: 10, vx: 5, vy: 5, vz: 5 };

    const result = interpolateParticles(p1, p2, 1);

    expect(result.x).toBeCloseTo(p2.x);
    expect(result.y).toBeCloseTo(p2.y);
    expect(result.z).toBeCloseTo(p2.z);
  });

  it('should interpolate midpoint when t=0.5', () => {
    const p1: Particle = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    const p2: Particle = { x: 10, y: 20, z: 30, vx: 2, vy: 4, vz: 6 };

    const result = interpolateParticles(p1, p2, 0.5);

    expect(result.x).toBeCloseTo(5);
    expect(result.y).toBeCloseTo(10);
    expect(result.z).toBeCloseTo(15);
    expect(result.vx).toBeCloseTo(1);
    expect(result.vy).toBeCloseTo(2);
    expect(result.vz).toBeCloseTo(3);
  });

  it('should handle negative coordinates', () => {
    const p1: Particle = { x: -10, y: -5, z: -15, vx: -1, vy: -2, vz: -3 };
    const p2: Particle = { x: 10, y: 5, z: 15, vx: 1, vy: 2, vz: 3 };

    const result = interpolateParticles(p1, p2, 0.25);

    expect(result.x).toBeCloseTo(-5);
    expect(result.y).toBeCloseTo(-2.5);
    expect(result.z).toBeCloseTo(-7.5);
  });
});

describe('Physics Calculations', () => {
  it('should calculate distance between two particles', () => {
    const p1: Particle = { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 };
    const p2: Particle = { x: 3, y: 4, z: 0, vx: 0, vy: 0, vz: 0 };

    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const dz = p2.z - p1.z;
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    expect(distance).toBeCloseTo(5);
  });

  it('should calculate velocity magnitude', () => {
    const p: Particle = { x: 0, y: 0, z: 0, vx: 3, vy: 4, vz: 0 };
    const speed = Math.sqrt(p.vx ** 2 + p.vy ** 2 + p.vz ** 2);

    expect(speed).toBeCloseTo(5);
  });

  it('should calculate circular orbital velocity', () => {
    const centralMass = 5000;
    const radius = 50;
    const G = 1.0;

    // v = sqrt(GM/r)
    const expectedV = Math.sqrt((G * centralMass) / radius);

    expect(expectedV).toBeGreaterThan(0);
    expect(expectedV).toBeCloseTo(10, 0);
  });
});
