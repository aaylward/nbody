import { describe, test, expect } from 'vitest';
import { buildFlatOctree, NODE_WORD_STRIDE, WORD_COM_X, WORD_COM_Y, WORD_COM_Z, WORD_MASS, WORD_CELL_WIDTH, WORD_CHILD_START, WORD_CHILD_COUNT, WORD_PARTICLE_COUNT } from '../flatOctree';
import { createParticleArray, setParticle } from '../../particleData';

function readNode(buf: ArrayBuffer, idx: number) {
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const o = idx * NODE_WORD_STRIDE;
  return {
    comX: f[o + WORD_COM_X],
    comY: f[o + WORD_COM_Y],
    comZ: f[o + WORD_COM_Z],
    mass: f[o + WORD_MASS],
    cellWidth: f[o + WORD_CELL_WIDTH],
    childStart: u[o + WORD_CHILD_START],
    childCount: u[o + WORD_CHILD_COUNT],
    particleCount: u[o + WORD_PARTICLE_COUNT],
  };
}

describe('FlatOctree', () => {
  test('single particle → one leaf node with particle mass and position as COM', () => {
    const particles = createParticleArray(1);
    setParticle(particles, 0, { x: 2, y: 3, z: 5, vx: 0, vy: 0, vz: 0, mass: 7 });

    const { buffer, nodeCount } = buildFlatOctree(particles);

    expect(nodeCount).toBe(1);
    const root = readNode(buffer, 0);
    expect(root.mass).toBe(7);
    expect(root.comX).toBeCloseTo(2, 5);
    expect(root.comY).toBeCloseTo(3, 5);
    expect(root.comZ).toBeCloseTo(5, 5);
    expect(root.childCount).toBe(0);
    expect(root.particleCount).toBe(1);
  });

  test('two particles → root aggregates mass and COM; has children', () => {
    const particles = createParticleArray(2);
    setParticle(particles, 0, { x: -1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });
    setParticle(particles, 1, { x: 1, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

    const { buffer, nodeCount } = buildFlatOctree(particles);

    expect(nodeCount).toBeGreaterThan(1);
    const root = readNode(buffer, 0);
    expect(root.mass).toBe(2);
    expect(root.comX).toBeCloseTo(0, 5);
    expect(root.particleCount).toBe(2);
    expect(root.childCount).toBeGreaterThan(0);
    expect(root.childStart).toBe(1);
  });

  test('center of mass weighted by mass', () => {
    const particles = createParticleArray(2);
    setParticle(particles, 0, { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 3 });
    setParticle(particles, 1, { x: 4, y: 0, z: 0, vx: 0, vy: 0, vz: 0, mass: 1 });

    const { buffer } = buildFlatOctree(particles);
    const root = readNode(buffer, 0);

    expect(root.comX).toBeCloseTo(1, 4);
    expect(root.mass).toBe(4);
  });

  test('mass conservation across random distribution', () => {
    const N = 500;
    const particles = createParticleArray(N);
    let expectedMass = 0;
    for (let i = 0; i < N; i++) {
      const m = 0.5 + Math.random();
      expectedMass += m;
      setParticle(particles, i, {
        x: (Math.random() - 0.5) * 100,
        y: (Math.random() - 0.5) * 100,
        z: (Math.random() - 0.5) * 100,
        vx: 0, vy: 0, vz: 0, mass: m,
      });
    }

    const { buffer } = buildFlatOctree(particles);
    const root = readNode(buffer, 0);
    expect(root.mass).toBeCloseTo(expectedMass, 2);
    expect(root.particleCount).toBe(N);
  });

  test('COM matches brute-force computation', () => {
    const N = 200;
    const particles = createParticleArray(N);
    let sumX = 0, sumY = 0, sumZ = 0, sumM = 0;
    for (let i = 0; i < N; i++) {
      const m = 1;
      const x = (Math.random() - 0.5) * 50;
      const y = (Math.random() - 0.5) * 50;
      const z = (Math.random() - 0.5) * 50;
      sumX += x * m; sumY += y * m; sumZ += z * m; sumM += m;
      setParticle(particles, i, { x, y, z, vx: 0, vy: 0, vz: 0, mass: m });
    }

    const { buffer } = buildFlatOctree(particles);
    const root = readNode(buffer, 0);
    expect(root.comX).toBeCloseTo(sumX / sumM, 2);
    expect(root.comY).toBeCloseTo(sumY / sumM, 2);
    expect(root.comZ).toBeCloseTo(sumZ / sumM, 2);
  });

  test('leaf cell widths positive; internal node child ranges are contiguous', () => {
    const N = 100;
    const particles = createParticleArray(N);
    for (let i = 0; i < N; i++) {
      setParticle(particles, i, {
        x: (Math.random() - 0.5) * 20,
        y: (Math.random() - 0.5) * 20,
        z: (Math.random() - 0.5) * 20,
        vx: 0, vy: 0, vz: 0, mass: 1,
      });
    }

    const { buffer, nodeCount } = buildFlatOctree(particles);

    for (let i = 0; i < nodeCount; i++) {
      const n = readNode(buffer, i);
      expect(n.cellWidth).toBeGreaterThan(0);
      if (n.childCount > 0) {
        expect(n.childStart).toBeGreaterThanOrEqual(1);
        expect(n.childStart + n.childCount).toBeLessThanOrEqual(nodeCount);
        expect(n.childCount).toBeLessThanOrEqual(8);
      }
    }
  });

  test('particle count sums match at each level', () => {
    const N = 300;
    const particles = createParticleArray(N);
    for (let i = 0; i < N; i++) {
      setParticle(particles, i, {
        x: (Math.random() - 0.5) * 40,
        y: (Math.random() - 0.5) * 40,
        z: (Math.random() - 0.5) * 40,
        vx: 0, vy: 0, vz: 0, mass: 1,
      });
    }

    const { buffer, nodeCount } = buildFlatOctree(particles);
    const root = readNode(buffer, 0);
    expect(root.particleCount).toBe(N);

    // For any internal node, sum of children particleCounts equals its own
    for (let i = 0; i < nodeCount; i++) {
      const n = readNode(buffer, i);
      if (n.childCount > 0) {
        let sum = 0;
        for (let c = 0; c < n.childCount; c++) {
          sum += readNode(buffer, n.childStart + c).particleCount;
        }
        expect(sum).toBe(n.particleCount);
      }
    }
  });

  test('empty particle array produces valid single empty node', () => {
    const particles = createParticleArray(0);
    const { buffer, nodeCount } = buildFlatOctree(particles);
    expect(nodeCount).toBe(1);
    const root = readNode(buffer, 0);
    expect(root.particleCount).toBe(0);
    expect(root.mass).toBe(0);
    expect(root.childCount).toBe(0);
  });
});
