import { describe, it, expect } from 'vitest';
import { initializeNBodyParticles } from '../initialization';
import { getParticle, getParticleCount, getCenterOfMassVelocity } from '../particleData';

describe('N-Body Initialization', () => {
  it('should initialize correct number of particles', () => {
    const numParticles = 100;
    const particles = initializeNBodyParticles(numParticles);
    expect(getParticleCount(particles)).toBe(numParticles);
  });

  it('should place a massive central body at the origin', () => {
    const particles = initializeNBodyParticles(100);
    const central = getParticle(particles, 0);
    expect(central.x).toBeCloseTo(0);
    expect(central.y).toBeCloseTo(0);
    expect(central.z).toBeCloseTo(0);
    expect(central.mass).toBe(5000);
  });

  it('should scale radius for large particle counts', () => {
    // Small count (scale = 1)
    const smallCount = 100;
    const smallParticles = initializeNBodyParticles(smallCount);
    let maxR_small = 0;
    for (let i = 1; i < smallCount; i++) {
      const p = getParticle(smallParticles, i);
      const r = Math.sqrt(p.x * p.x + p.y * p.y);
      if (r > maxR_small) maxR_small = r;
    }
    // Expected max radius approx 80 (since range is 20-80)
    // We allow some margin because random numbers could be close to bounds
    expect(maxR_small).toBeLessThan(85);

    // Large count (scale > 1)
    // 20000 particles -> scale = sqrt(4) = 2. Max radius ~ 160.
    const largeCount = 20000;
    const largeParticles = initializeNBodyParticles(largeCount);
    let maxR_large = 0;
    // Check a subset to save time in test
    for (let i = 1; i < largeCount; i+=100) {
      const p = getParticle(largeParticles, i);
      const r = Math.sqrt(p.x * p.x + p.y * p.y);
      if (r > maxR_large) maxR_large = r;
    }

    expect(maxR_large).toBeGreaterThan(80); // Should be significantly larger than base scale
    // With random distribution, getting exactly close to 160 is likely but not guaranteed,
    // but definitely > 80.
  });

  it('should increase velocities for large particle counts (cloud mass effect)', () => {
    // Small count
    const smallCount = 100;
    const smallParticles = initializeNBodyParticles(smallCount);
    let avgV_small = 0;
    for (let i = 1; i < smallCount; i++) {
      const p = getParticle(smallParticles, i);
      avgV_small += Math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
    }
    avgV_small /= (smallCount - 1);

    // Large count
    // With 20k particles, total mass is 25k (vs 5k central).
    // Radius doubles (scale=2).
    // M_enclosed roughly 4x larger on average?
    // r roughly 2x larger.
    // v ~ sqrt(M/r) ~ sqrt(4/2) ~ sqrt(2) ~ 1.4x larger.
    const largeCount = 20000;
    const largeParticles = initializeNBodyParticles(largeCount);
    let avgV_large = 0;
    for (let i = 1; i < largeCount; i+=100) {
      const p = getParticle(largeParticles, i);
      avgV_large += Math.sqrt(p.vx * p.vx + p.vy * p.vy + p.vz * p.vz);
    }
    avgV_large /= Math.floor((largeCount - 1) / 100);

    expect(avgV_large).toBeGreaterThan(avgV_small);
  });

  it('should remove center of mass velocity', () => {
    const particles = initializeNBodyParticles(1000);
    const comVel = getCenterOfMassVelocity(particles);

    expect(Math.abs(comVel.vx)).toBeLessThan(1e-5);
    expect(Math.abs(comVel.vy)).toBeLessThan(1e-5);
    expect(Math.abs(comVel.vz)).toBeLessThan(1e-5);
  });
});
