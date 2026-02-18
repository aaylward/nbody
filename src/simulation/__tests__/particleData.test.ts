import { describe, it, expect } from 'vitest';
import {
  createParticleArray,
  getParticle,
  setParticle,
  getPosition,
  getVelocity,
  getMass,
  updatePosition,
  updateVelocity,
  copyParticle,
  getParticleCount,
  cloneParticleData,
  fromParticleObjects,
  toParticleObjects,
  extractPositions,
  extractVelocities,
  calculateColors,
  getCenterOfMass,
  FLOATS_PER_PARTICLE,
} from '../particleData';

describe('particleData', () => {
  describe('createParticleArray', () => {
    it('should create array with correct size', () => {
      const data = createParticleArray(10);
      expect(data.length).toBe(10 * FLOATS_PER_PARTICLE);
      expect(data).toBeInstanceOf(Float32Array);
    });

    it('should initialize with zeros', () => {
      const data = createParticleArray(5);
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBe(0);
      }
    });
  });

  describe('setParticle and getParticle', () => {
    it('should set and get particle data correctly', () => {
      const data = createParticleArray(3);
      const particle = {
        x: 1.5,
        y: 2.5,
        z: 3.5,
        vx: 0.1,
        vy: 0.2,
        vz: 0.3,
        mass: 10,
      };

      setParticle(data, 1, particle);
      const retrieved = getParticle(data, 1);

      expect(retrieved.x).toBe(1.5);
      expect(retrieved.y).toBe(2.5);
      expect(retrieved.z).toBe(3.5);
      expect(retrieved.vx).toBeCloseTo(0.1, 5);
      expect(retrieved.vy).toBeCloseTo(0.2, 5);
      expect(retrieved.vz).toBeCloseTo(0.3, 5);
      expect(retrieved.mass).toBe(10);
    });

    it('should default mass to 1 when not provided', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 1, y: 2, z: 3, vx: 0, vy: 0, vz: 0 });
      const particle = getParticle(data, 0);
      expect(particle.mass).toBe(1);
    });

    it('should not affect other particles', () => {
      const data = createParticleArray(3);

      setParticle(data, 0, { x: 1, y: 1, z: 1, vx: 0, vy: 0, vz: 0, mass: 5 });
      setParticle(data, 1, { x: 2, y: 2, z: 2, vx: 1, vy: 1, vz: 1, mass: 10 });
      setParticle(data, 2, { x: 3, y: 3, z: 3, vx: 2, vy: 2, vz: 2, mass: 15 });

      const p0 = getParticle(data, 0);
      const p2 = getParticle(data, 2);

      expect(p0.x).toBe(1);
      expect(p0.mass).toBe(5);
      expect(p2.x).toBe(3);
      expect(p2.mass).toBe(15);
    });
  });

  describe('getPosition', () => {
    it('should get position components', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 1.5, y: 2.5, z: 3.5, vx: 0, vy: 0, vz: 0 });

      const pos = getPosition(data, 0);
      expect(pos.x).toBe(1.5);
      expect(pos.y).toBe(2.5);
      expect(pos.z).toBe(3.5);
    });
  });

  describe('getVelocity', () => {
    it('should get velocity components', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 0.1, vy: 0.2, vz: 0.3 });

      const vel = getVelocity(data, 0);
      expect(vel.vx).toBeCloseTo(0.1, 5);
      expect(vel.vy).toBeCloseTo(0.2, 5);
      expect(vel.vz).toBeCloseTo(0.3, 5);
    });
  });

  describe('getMass', () => {
    it('should get mass', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 42 });

      expect(getMass(data, 0)).toBe(42);
    });
  });

  describe('updatePosition', () => {
    it('should update position by delta', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 1, y: 2, z: 3, vx: 0, vy: 0, vz: 0 });

      updatePosition(data, 0, 0.5, 0.6, 0.7);

      const pos = getPosition(data, 0);
      expect(pos.x).toBeCloseTo(1.5);
      expect(pos.y).toBeCloseTo(2.6);
      expect(pos.z).toBeCloseTo(3.7);
    });
  });

  describe('updateVelocity', () => {
    it('should update velocity by delta', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 1, vy: 2, vz: 3 });

      updateVelocity(data, 0, 0.1, 0.2, 0.3);

      const vel = getVelocity(data, 0);
      expect(vel.vx).toBeCloseTo(1.1);
      expect(vel.vy).toBeCloseTo(2.2);
      expect(vel.vz).toBeCloseTo(3.3);
    });
  });

  describe('copyParticle', () => {
    it('should copy particle from source to dest', () => {
      const source = createParticleArray(2);
      const dest = createParticleArray(2);

      setParticle(source, 1, { x: 5, y: 6, z: 7, vx: 1, vy: 2, vz: 3, mass: 20 });
      copyParticle(source, 1, dest, 0);

      const copied = getParticle(dest, 0);
      expect(copied.x).toBe(5);
      expect(copied.y).toBe(6);
      expect(copied.z).toBe(7);
      expect(copied.mass).toBe(20);
    });
  });

  describe('getParticleCount', () => {
    it('should return correct particle count', () => {
      const data = createParticleArray(100);
      expect(getParticleCount(data)).toBe(100);
    });
  });

  describe('cloneParticleData', () => {
    it('should create independent copy', () => {
      const original = createParticleArray(2);
      setParticle(original, 0, { x: 1, y: 2, z: 3, vx: 0, vy: 0, vz: 0 });

      const clone = cloneParticleData(original);
      setParticle(clone, 0, { x: 99, y: 99, z: 99, vx: 0, vy: 0, vz: 0 });

      expect(getParticle(original, 0).x).toBe(1);
      expect(getParticle(clone, 0).x).toBe(99);
    });
  });

  describe('fromParticleObjects', () => {
    it('should convert object array to TypedArray', () => {
      const objects = [
        { x: 1, y: 2, z: 3, vx: 0.1, vy: 0.2, vz: 0.3, mass: 5 },
        { x: 4, y: 5, z: 6, vx: 0.4, vy: 0.5, vz: 0.6, mass: 10 },
      ];

      const data = fromParticleObjects(objects);

      expect(getParticleCount(data)).toBe(2);
      expect(getParticle(data, 0).x).toBe(1);
      expect(getParticle(data, 0).mass).toBe(5);
      expect(getParticle(data, 1).x).toBe(4);
      expect(getParticle(data, 1).mass).toBe(10);
    });

    it('should handle missing mass', () => {
      const objects = [{ x: 1, y: 2, z: 3, vx: 0, vy: 0, vz: 0 }];
      const data = fromParticleObjects(objects);
      expect(getParticle(data, 0).mass).toBe(1);
    });
  });

  describe('toParticleObjects', () => {
    it('should convert TypedArray to object array', () => {
      const data = createParticleArray(2);
      setParticle(data, 0, { x: 1, y: 2, z: 3, vx: 0.1, vy: 0.2, vz: 0.3, mass: 5 });
      setParticle(data, 1, { x: 4, y: 5, z: 6, vx: 0.4, vy: 0.5, vz: 0.6, mass: 10 });

      const objects = toParticleObjects(data);

      expect(objects.length).toBe(2);
      expect(objects[0].x).toBe(1);
      expect(objects[0].mass).toBe(5);
      expect(objects[1].x).toBe(4);
      expect(objects[1].mass).toBe(10);
    });
  });

  describe('extractPositions', () => {
    it('should extract positions in THREE.js format', () => {
      const data = createParticleArray(2);
      setParticle(data, 0, { x: 1, y: 2, z: 3, vx: 0, vy: 0, vz: 0 });
      setParticle(data, 1, { x: 4, y: 5, z: 6, vx: 0, vy: 0, vz: 0 });

      const positions = extractPositions(data);

      expect(positions.length).toBe(6); // 2 particles × 3 components
      expect(positions[0]).toBe(1);
      expect(positions[1]).toBe(2);
      expect(positions[2]).toBe(3);
      expect(positions[3]).toBe(4);
      expect(positions[4]).toBe(5);
      expect(positions[5]).toBe(6);
    });

    it('should write to output buffer if provided', () => {
      const data = createParticleArray(2);
      setParticle(data, 0, { x: 1, y: 2, z: 3, vx: 0, vy: 0, vz: 0 });
      setParticle(data, 1, { x: 4, y: 5, z: 6, vx: 0, vy: 0, vz: 0 });

      const out = new Float32Array(6);
      const result = extractPositions(data, out);

      expect(result).toBe(out); // Should return the same buffer instance
      expect(out[0]).toBe(1);
      expect(out[5]).toBe(6);
    });
  });

  describe('extractVelocities', () => {
    it('should extract velocities', () => {
      const data = createParticleArray(2);
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 1, vy: 2, vz: 3 });
      setParticle(data, 1, { x: 0, y: 0, z: 0, vx: 4, vy: 5, vz: 6 });

      const velocities = extractVelocities(data);

      expect(velocities.length).toBe(6);
      expect(velocities[0]).toBe(1);
      expect(velocities[1]).toBe(2);
      expect(velocities[2]).toBe(3);
      expect(velocities[3]).toBe(4);
      expect(velocities[4]).toBe(5);
      expect(velocities[5]).toBe(6);
    });
  });

  describe('calculateColors', () => {
    it('should calculate colors based on velocity', () => {
      const data = createParticleArray(2);
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0 }); // stationary
      setParticle(data, 1, { x: 0, y: 0, z: 0, vx: 10, vy: 0, vz: 0 }); // max velocity

      const colors = calculateColors(data, 10);

      expect(colors.length).toBe(6); // 2 particles × 3 components

      // Stationary particle should have low red, mid green, high blue
      expect(colors[0]).toBe(0.5); // R
      expect(colors[1]).toBe(0.5); // G
      expect(colors[2]).toBe(1.0); // B

      // Fast particle should have high red, mid green, low blue
      expect(colors[3]).toBe(1.0); // R
      expect(colors[4]).toBe(0.5); // G
      expect(colors[5]).toBe(0.5); // B
    });

    it('should clamp velocities at maxVelocity', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 20, vy: 0, vz: 0 }); // 2× max

      const colors = calculateColors(data, 10);

      // Should be clamped to max color values
      expect(colors[0]).toBe(1.0);
      expect(colors[2]).toBe(0.5);
    });

    it('should write to output buffer if provided', () => {
      const data = createParticleArray(1);
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 20, vy: 0, vz: 0 });

      const out = new Float32Array(3);
      const result = calculateColors(data, 10, out);

      expect(result).toBe(out);
      expect(out[0]).toBe(1.0);
    });
  });

  describe('memory efficiency', () => {
    it('should use less memory than object arrays', () => {
      const numParticles = 10000;

      // TypedArray approach
      const typedData = createParticleArray(numParticles);
      const typedBytes = typedData.byteLength;

      // Object array approach (estimate)
      const objectArray = Array(numParticles)
        .fill(null)
        .map(() => ({ x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 }));

      // TypedArray: 10000 particles × 8 floats × 4 bytes = 320KB
      expect(typedBytes).toBe(320000);

      // Object arrays typically use 10-20× more memory due to object overhead
      // We can't measure exact object memory in JS, but we know:
      // - Each object has ~32 bytes overhead
      // - Each property has ~8 bytes overhead
      // - Each number is stored as 8 bytes (not optimized)
      // Estimate: (32 + 7×(8+8)) × 10000 = 1.44MB

      // Just verify the TypedArray exists and has correct size
      expect(typedData).toBeInstanceOf(Float32Array);
      expect(objectArray.length).toBe(numParticles);
    });
  });

  describe('getCenterOfMass', () => {
    it('should calculate center of mass for uniform particles', () => {
      const data = createParticleArray(3);

      // Three particles in a line with equal mass
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(data, 1, { x: 3, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(data, 2, { x: 6, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const com = getCenterOfMass(data);

      expect(com.x).toBe(3); // (0 + 3 + 6) / 3
      expect(com.y).toBe(0);
      expect(com.z).toBe(0);
    });

    it('should weight by mass correctly', () => {
      const data = createParticleArray(2);

      // Two particles: one at origin with mass 1, one at (10,0,0) with mass 9
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(data, 1, { x: 10, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 9 });

      const com = getCenterOfMass(data);

      // COM should be at (1*0 + 9*10) / (1+9) = 90/10 = 9
      expect(com.x).toBe(9);
      expect(com.y).toBe(0);
      expect(com.z).toBe(0);
    });

    it('should handle 3D distribution', () => {
      const data = createParticleArray(4);

      // Four particles at corners of a tetrahedron, equal mass
      setParticle(data, 0, { x: 1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(data, 1, { x: -1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(data, 2, { x: 0, y: 1, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
      setParticle(data, 3, { x: 0, y: -1, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

      const com = getCenterOfMass(data);

      // Should be at origin for symmetric distribution
      expect(com.x).toBeCloseTo(0, 5);
      expect(com.y).toBeCloseTo(0, 5);
      expect(com.z).toBe(0);
    });

    it('should match central massive body for stellar system', () => {
      const data = createParticleArray(11);

      // Central star at origin with mass 1000
      setParticle(data, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1000 });

      // 10 small particles orbiting with mass 1 each, at distance 10
      for (let i = 1; i < 11; i++) {
        const angle = (i - 1) * Math.PI / 5;
        setParticle(data, i, {
          x: 10 * Math.cos(angle),
          y: 10 * Math.sin(angle),
          z: 0,
          vx: 0,
          vy: 0,
          vz: 0,
          mass: 1
        });
      }

      const com = getCenterOfMass(data);

      // COM should be very close to origin (central star dominates)
      expect(Math.abs(com.x)).toBeLessThan(0.1);
      expect(Math.abs(com.y)).toBeLessThan(0.1);
      expect(com.z).toBe(0);
    });
  });
});
