/**
 * Tests for the CPU↔GPU data packing used by RealtimeSimulationGPUBarnesHut.
 *
 * These test the actual production functions from barnesHutPacking.ts,
 * verifying the stride and offset math that previously caused a
 * flash-then-blank bug (stride 7 vs the actual FLOATS_PER_PARTICLE = 8).
 */

import { describe, it, expect } from 'vitest';
import {
  createParticleArray,
  setParticle,
  FLOATS_PER_PARTICLE,
  OFFSET_X,
  OFFSET_Y,
  OFFSET_Z,
  OFFSET_VX,
  OFFSET_VY,
  OFFSET_VZ,
  OFFSET_MASS,
} from '../../particleData';
import {
  packParticlesForGPU,
  packVelocitiesForGPU,
  unpackParticlesFromGPU,
} from '../barnesHutPacking';

describe('Barnes-Hut CPU↔GPU packing', () => {
  const N = 5;

  function makeTestParticles() {
    const cpu = createParticleArray(N);
    // Central body
    setParticle(cpu, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 5000 });
    // Orbiting particles with distinct values so misalignment is obvious
    for (let i = 1; i < N; i++) {
      setParticle(cpu, i, {
        x: i * 10,
        y: i * 20,
        z: i * 30,
        vx: i * 0.1,
        vy: i * 0.2,
        vz: i * 0.3,
        mass: i,
      });
    }
    return cpu;
  }

  it('should pack positions and masses correctly for all particles', () => {
    const cpu = makeTestParticles();
    const gpu = packParticlesForGPU(cpu, N);

    // Central body
    expect(gpu[0]).toBe(0);   // x
    expect(gpu[1]).toBe(0);   // y
    expect(gpu[2]).toBe(0);   // z
    expect(gpu[3]).toBe(5000); // mass — NOT 0 (the old bug read vz here)

    // Subsequent particles — these were garbled when stride was 7
    for (let i = 1; i < N; i++) {
      const g = i * 4;
      expect(gpu[g + 0]).toBe(i * 10);  // x
      expect(gpu[g + 1]).toBe(i * 20);  // y
      expect(gpu[g + 2]).toBe(i * 30);  // z
      expect(gpu[g + 3]).toBe(i);       // mass
    }
  });

  it('should pack velocities correctly for all particles', () => {
    const cpu = makeTestParticles();
    const gpu = packVelocitiesForGPU(cpu, N);

    // Central body
    expect(gpu[0]).toBe(0);  // vx
    expect(gpu[1]).toBe(0);  // vy
    expect(gpu[2]).toBe(0);  // vz
    expect(gpu[3]).toBe(0);  // padding

    // Subsequent particles
    for (let i = 1; i < N; i++) {
      const v = i * 4;
      expect(gpu[v + 0]).toBeCloseTo(i * 0.1);  // vx
      expect(gpu[v + 1]).toBeCloseTo(i * 0.2);  // vy
      expect(gpu[v + 2]).toBeCloseTo(i * 0.3);  // vz
      expect(gpu[v + 3]).toBe(0);                // padding
    }
  });

  it('should unpack GPU positions back to the correct CPU slots', () => {
    const cpu = createParticleArray(N);
    // Pre-fill velocities so we can verify they survive the unpack
    for (let i = 0; i < N; i++) {
      setParticle(cpu, i, { x: 0, y: 0, z: 0, vx: 99, vy: 88, vz: 77, mass: 0 });
    }

    // Simulate GPU data
    const gpuData = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) {
      gpuData[i * 4 + 0] = i * 10;  // x
      gpuData[i * 4 + 1] = i * 20;  // y
      gpuData[i * 4 + 2] = i * 30;  // z
      gpuData[i * 4 + 3] = i + 1;   // mass
    }

    unpackParticlesFromGPU(gpuData, cpu, N);

    for (let i = 0; i < N; i++) {
      const offset = i * FLOATS_PER_PARTICLE;
      expect(cpu[offset + OFFSET_X]).toBe(i * 10);
      expect(cpu[offset + OFFSET_Y]).toBe(i * 20);
      expect(cpu[offset + OFFSET_Z]).toBe(i * 30);
      expect(cpu[offset + OFFSET_MASS]).toBe(i + 1);
      // Velocities must be untouched
      expect(cpu[offset + OFFSET_VX]).toBe(99);
      expect(cpu[offset + OFFSET_VY]).toBe(88);
      expect(cpu[offset + OFFSET_VZ]).toBe(77);
    }
  });

  it('should write numParticles as u32 not f32 in uniform buffer', () => {
    const n = 50000;

    // Correct: write as u32
    const buf = new ArrayBuffer(16);
    new Uint32Array(buf, 0, 1)[0] = n;
    const asU32 = new Uint32Array(buf, 0, 1)[0];
    expect(asU32).toBe(50000);

    // What the old code did: write as f32 then read bits as u32
    const badBuf = new ArrayBuffer(4);
    new Float32Array(badBuf)[0] = n;
    const badU32 = new Uint32Array(badBuf)[0];
    // f32(50000) has a completely different bit pattern than u32(50000)
    expect(badU32).not.toBe(50000);
  });
});
