import { describe, test, expect } from 'vitest';
import { Octree } from '../octree';
import { createParticleArray, setParticle, getParticle } from '../../particleData';

describe('Octree', () => {
  describe('Construction', () => {
    test('should build octree for single particle', () => {
      const particles = createParticleArray(1);
      setParticle(particles, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const octree = new Octree(particles);
      const root = octree.getRoot();

      expect(root.isLeaf).toBe(true);
      expect(root.particleCount).toBe(1);
      expect(root.totalMass).toBe(1);
      expect(root.centerOfMass).toEqual({ x: 0, y: 0, z: 0 });
    });

    test('should build octree for two particles', () => {
      const particles = createParticleArray(2);
      setParticle(particles, 0, { x: -1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(particles, 1, { x: 1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const octree = new Octree(particles, 1);
      const root = octree.getRoot();

      expect(root.particleCount).toBe(2);
      expect(root.totalMass).toBe(2);
      // Center of mass should be at origin
      expect(root.centerOfMass.x).toBeCloseTo(0, 5);
      expect(root.centerOfMass.y).toBeCloseTo(0, 5);
      expect(root.centerOfMass.z).toBeCloseTo(0, 5);
    });

    test('should subdivide when maxParticlesPerNode exceeded', () => {
      const particles = createParticleArray(9);
      // Place particles in different octants
      const positions = [
        { x: -1, y: -1, z: -1 },
        { x: 1, y: -1, z: -1 },
        { x: -1, y: 1, z: -1 },
        { x: 1, y: 1, z: -1 },
        { x: -1, y: -1, z: 1 },
        { x: 1, y: -1, z: 1 },
        { x: -1, y: 1, z: 1 },
        { x: 1, y: 1, z: 1 },
        { x: 0, y: 0, z: 0 },
      ];

      for (let i = 0; i < 9; i++) {
        setParticle(particles, i, {
          ...positions[i],
          vx: 0,
          vy: 0,
          vz: 0,
          mass: 1,
        });
      }

      const octree = new Octree(particles, 1);
      const root = octree.getRoot();

      expect(root.isLeaf).toBe(false);
      expect(root.particleCount).toBe(9);
      expect(root.children).not.toBeNull();
      expect(root.children).toHaveLength(8);
    });

    test('should compute correct tree depth', () => {
      const particles = createParticleArray(100);
      for (let i = 0; i < 100; i++) {
        setParticle(particles, i, {
          x: Math.random() * 100 - 50,
          y: Math.random() * 100 - 50,
          z: Math.random() * 100 - 50,
          vx: 0,
          vy: 0,
          vz: 0,
          mass: 1,
        });
      }

      const octree = new Octree(particles, 1);
      const depth = octree.getMaxDepth();
      const nodeCount = octree.countNodes();

      // With 100 particles and max 1 per leaf, we expect reasonable depth
      expect(depth).toBeGreaterThan(0);
      expect(depth).toBeLessThan(20); // Should not be pathologically deep
      expect(nodeCount).toBeGreaterThan(1);
    });
  });

  describe('Center of Mass', () => {
    test('should compute center of mass for uniform distribution', () => {
      const particles = createParticleArray(8);
      // 8 particles at corners of cube centered at origin
      const corners = [
        { x: -1, y: -1, z: -1 },
        { x: 1, y: -1, z: -1 },
        { x: -1, y: 1, z: -1 },
        { x: 1, y: 1, z: -1 },
        { x: -1, y: -1, z: 1 },
        { x: 1, y: -1, z: 1 },
        { x: -1, y: 1, z: 1 },
        { x: 1, y: 1, z: 1 },
      ];

      for (let i = 0; i < 8; i++) {
        setParticle(particles, i, { ...corners[i], vx: 0, vy: 0, vz: 0, mass: 1 });
      }

      const octree = new Octree(particles);
      const root = octree.getRoot();

      // Center of mass should be at origin for symmetric distribution
      expect(root.centerOfMass.x).toBeCloseTo(0, 5);
      expect(root.centerOfMass.y).toBeCloseTo(0, 5);
      expect(root.centerOfMass.z).toBeCloseTo(0, 5);
      expect(root.totalMass).toBe(8);
    });

    test('should compute center of mass with different masses', () => {
      const particles = createParticleArray(2);
      setParticle(particles, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 3 });
      setParticle(particles, 1, { x: 4, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const octree = new Octree(particles);
      const root = octree.getRoot();

      // COM = (3*0 + 1*4) / (3+1) = 1
      expect(root.centerOfMass.x).toBeCloseTo(1, 5);
      expect(root.centerOfMass.y).toBeCloseTo(0, 5);
      expect(root.centerOfMass.z).toBeCloseTo(0, 5);
      expect(root.totalMass).toBe(4);
    });

    test('should propagate center of mass up tree', () => {
      const particles = createParticleArray(4);
      // 4 particles in 2 different octants
      setParticle(particles, 0, { x: -10, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(particles, 1, { x: -9, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(particles, 2, { x: 9, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(particles, 3, { x: 10, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const octree = new Octree(particles, 1);
      const root = octree.getRoot();

      // COM of all 4 particles
      const expectedX = (-10 + -9 + 9 + 10) / 4;
      expect(root.centerOfMass.x).toBeCloseTo(expectedX, 5);
      expect(root.totalMass).toBe(4);
    });
  });

  describe('Force Calculation', () => {
    test('should compute zero force for single particle', () => {
      const particles = createParticleArray(1);
      setParticle(particles, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const octree = new Octree(particles);
      const force = octree.computeForce(0, 0.5);

      expect(force.x).toBe(0);
      expect(force.y).toBe(0);
      expect(force.z).toBe(0);
    });

    test('should compute correct force for two particles', () => {
      const particles = createParticleArray(2);
      const G = 1.0;
      const softening = 0.0; // No softening for exact test
      const distance = 10.0;

      setParticle(particles, 0, {
        x: 0,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        mass: 1,
      });
      setParticle(particles, 1, {
        x: distance,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        mass: 1,
      });

      const octree = new Octree(particles);
      const force = octree.computeForce(0, 0.5, G, softening);

      // F = G * m1 * m2 / r^2
      // Expected force magnitude: 1 * 1 * 1 / 100 = 0.01
      // Direction: positive x (towards particle 1)
      const expectedMag = (G * 1 * 1) / (distance * distance);

      expect(force.x).toBeCloseTo(expectedMag, 5);
      expect(force.y).toBeCloseTo(0, 5);
      expect(force.z).toBeCloseTo(0, 5);
    });

    test('should compute force with softening parameter', () => {
      const particles = createParticleArray(2);
      const G = 1.0;
      const softening = 2.0;
      const distance = 1.0;

      setParticle(particles, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(particles, 1, { x: distance, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const octree = new Octree(particles);
      const force = octree.computeForce(0, 0.5, G, softening);

      // F = G * m1 * m2 / (r^2 + softening^2)^(3/2)
      const r2Soft = distance * distance + softening * softening;
      const expectedMag = (G * 1 * 1) / (Math.sqrt(r2Soft) * r2Soft);

      expect(force.x).toBeCloseTo(expectedMag, 5);
      expect(force.y).toBeCloseTo(0, 5);
      expect(force.z).toBeCloseTo(0, 5);
    });

    test('should compute force in 3D', () => {
      const particles = createParticleArray(2);
      setParticle(particles, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(particles, 1, { x: 3, y: 4, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const octree = new Octree(particles);
      const force = octree.computeForce(0, 0.5, 1.0, 0.0);

      // Distance = 5
      const distance = 5;
      const forceMag = 1 / (distance * distance);

      // Force components
      const fx = forceMag * (3 / distance);
      const fy = forceMag * (4 / distance);

      expect(force.x).toBeCloseTo(fx, 5);
      expect(force.y).toBeCloseTo(fy, 5);
      expect(force.z).toBeCloseTo(0, 5);
    });

    test('Barnes-Hut approximation should be reasonably accurate', () => {
      // Test that Barnes-Hut with theta=0.5 gives accurate results
      const numParticles = 100;
      const particles = createParticleArray(numParticles);

      // Create particles in a random distribution
      for (let i = 0; i < numParticles; i++) {
        setParticle(particles, i, {
          x: (Math.random() - 0.5) * 100,
          y: (Math.random() - 0.5) * 100,
          z: (Math.random() - 0.5) * 100,
          vx: 0,
          vy: 0,
          vz: 0,
          mass: 1,
        });
      }

      const octree = new Octree(particles);

      // Compute force using Barnes-Hut
      const bhForce = octree.computeForce(0, 0.5);

      // Compute exact force by brute force
      const exactForce = computeBruteForce(particles, 0);

      // Barnes-Hut should be within 10% of exact for theta=0.5
      const error = Math.sqrt(
        Math.pow(bhForce.x - exactForce.x, 2) +
          Math.pow(bhForce.y - exactForce.y, 2) +
          Math.pow(bhForce.z - exactForce.z, 2)
      );
      const exactMag = Math.sqrt(
        exactForce.x * exactForce.x + exactForce.y * exactForce.y + exactForce.z * exactForce.z
      );

      const relativeError = error / exactMag;
      expect(relativeError).toBeLessThan(0.1); // < 10% error
    });

    test('tighter theta should give better accuracy', () => {
      const numParticles = 50;
      const particles = createParticleArray(numParticles);

      for (let i = 0; i < numParticles; i++) {
        setParticle(particles, i, {
          x: (Math.random() - 0.5) * 100,
          y: (Math.random() - 0.5) * 100,
          z: (Math.random() - 0.5) * 100,
          vx: 0,
          vy: 0,
          vz: 0,
          mass: 1,
        });
      }

      const octree = new Octree(particles);
      const exactForce = computeBruteForce(particles, 0);

      // Compute error for different theta values
      const error05 = computeError(octree, exactForce, 0, 0.5);
      const error03 = computeError(octree, exactForce, 0, 0.3);

      // Tighter theta (0.3) should give better accuracy than 0.5
      expect(error03).toBeLessThan(error05);
    });
  });

  describe('Performance', () => {
    test('should scale better than brute force for large N', () => {
      const numParticles = 5000;
      const particles = createParticleArray(numParticles);

      for (let i = 0; i < numParticles; i++) {
        setParticle(particles, i, {
          x: (Math.random() - 0.5) * 100,
          y: (Math.random() - 0.5) * 100,
          z: (Math.random() - 0.5) * 100,
          vx: 0,
          vy: 0,
          vz: 0,
          mass: 1,
        });
      }

      // Build octree once
      const octree = new Octree(particles);

      // Time Barnes-Hut force calculation (sample subset to keep test fast)
      const sampleSize = 100;
      const bhStart = performance.now();
      for (let i = 0; i < sampleSize; i++) {
        octree.computeForce(i, 0.5);
      }
      const bhTime = performance.now() - bhStart;

      // Time brute force for same particles
      const bfStart = performance.now();
      for (let i = 0; i < sampleSize; i++) {
        computeBruteForce(particles, i);
      }
      const bfTime = performance.now() - bfStart;

      // Barnes-Hut should be faster (with large enough N)
      expect(bhTime).toBeLessThan(bfTime);

      // Log for visibility
      console.log(
        `Barnes-Hut time: ${bhTime.toFixed(2)}ms (${sampleSize} forces, N=${numParticles})`
      );
      console.log(
        `Brute force time: ${bfTime.toFixed(2)}ms (${sampleSize} forces, N=${numParticles})`
      );
      console.log(`Speedup: ${(bfTime / bhTime).toFixed(2)}x`);
    });

    test('tree depth should scale as O(log N)', () => {
      const sizes = [100, 1000, 10000];
      const depths: number[] = [];

      for (const size of sizes) {
        const particles = createParticleArray(size);
        for (let i = 0; i < size; i++) {
          setParticle(particles, i, {
            x: (Math.random() - 0.5) * 100,
            y: (Math.random() - 0.5) * 100,
            z: (Math.random() - 0.5) * 100,
            vx: 0,
            vy: 0,
            vz: 0,
            mass: 1,
          });
        }

        const octree = new Octree(particles);
        depths.push(octree.getMaxDepth());
      }

      console.log('Tree depths:', depths);
      console.log('N values:', sizes);

      // Depth should scale sublinearly with N
      // With 100x increase in particles, depth should grow much less than 100x
      const depthRatio = depths[2] / depths[0];
      const sizeRatio = sizes[2] / sizes[0];
      expect(depthRatio).toBeLessThan(sizeRatio / 10); // Much better than linear
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty particle array', () => {
      const particles = createParticleArray(0);
      const octree = new Octree(particles);

      expect(octree.countNodes()).toBeGreaterThan(0);
      expect(octree.getRoot().particleCount).toBe(0);
    });

    test('should handle particles at same location', () => {
      const particles = createParticleArray(3);
      for (let i = 0; i < 3; i++) {
        setParticle(particles, i, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      }

      const octree = new Octree(particles);
      const root = octree.getRoot();

      expect(root.centerOfMass).toEqual({ x: 0, y: 0, z: 0 });
      expect(root.totalMass).toBe(3);
    });

    test('should handle very large coordinate values', () => {
      const particles = createParticleArray(2);
      setParticle(particles, 0, {
        x: 1e6,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        mass: 1,
      });
      setParticle(particles, 1, {
        x: -1e6,
        y: 0,
        z: 0,
        vx: 0,
        vy: 0,
        vz: 0,
        mass: 1,
      });

      const octree = new Octree(particles);
      const root = octree.getRoot();

      expect(root.centerOfMass.x).toBeCloseTo(0, 1);
      expect(root.totalMass).toBe(2);
    });
  });
});

// Helper functions

/**
 * Compute exact force using brute-force O(N²) algorithm
 */
function computeBruteForce(
  particles: Float32Array,
  particleIndex: number,
  G = 1.0,
  softening = 2.0
) {
  const p = getParticle(particles, particleIndex);
  let fx = 0,
    fy = 0,
    fz = 0;

  const numParticles = particles.length / 7;

  for (let j = 0; j < numParticles; j++) {
    if (j === particleIndex) continue;

    const q = getParticle(particles, j);
    const dx = q.x - p.x;
    const dy = q.y - p.y;
    const dz = q.z - p.z;

    const r2 = dx * dx + dy * dy + dz * dz + softening * softening;
    const r = Math.sqrt(r2);
    const invR3 = 1 / (r * r2);
    const f = G * p.mass * q.mass * invR3;

    fx += f * dx;
    fy += f * dy;
    fz += f * dz;
  }

  return { x: fx, y: fy, z: fz };
}

/**
 * Compute relative error between Barnes-Hut and exact force
 */
function computeError(
  octree: Octree,
  exactForce: { x: number; y: number; z: number },
  particleIndex: number,
  theta: number
): number {
  const bhForce = octree.computeForce(particleIndex, theta);

  const error = Math.sqrt(
    Math.pow(bhForce.x - exactForce.x, 2) +
      Math.pow(bhForce.y - exactForce.y, 2) +
      Math.pow(bhForce.z - exactForce.z, 2)
  );

  const exactMag = Math.sqrt(
    exactForce.x * exactForce.x + exactForce.y * exactForce.y + exactForce.z * exactForce.z
  );

  return exactMag > 0 ? error / exactMag : 0;
}
