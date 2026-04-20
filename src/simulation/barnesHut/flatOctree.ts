/**
 * Flat Barnes-Hut octree.
 *
 * Builds directly into a GPU-ready node buffer with no per-node JS objects
 * and no per-level index arrays. Children of a node live in a contiguous
 * run of node indices, so the buffer is the serialization.
 *
 * Node layout (32 bytes = 8 × 32-bit words):
 *   word 0-2  : centerOfMass.x/y/z (f32)
 *   word 3    : totalMass          (f32)
 *   word 4    : cellWidth          (f32)
 *   word 5    : childStart         (u32, 0 if leaf)
 *   word 6    : childCount         (u32, 0 if leaf)
 *   word 7    : particleCount      (u32)
 */

import {
  FLOATS_PER_PARTICLE,
  OFFSET_X,
  OFFSET_Y,
  OFFSET_Z,
  OFFSET_MASS,
  getParticleCount,
} from '../particleData';

export const NODE_WORD_STRIDE = 8;
export const BYTES_PER_NODE = NODE_WORD_STRIDE * 4;

export const WORD_COM_X = 0;
export const WORD_COM_Y = 1;
export const WORD_COM_Z = 2;
export const WORD_MASS = 3;
export const WORD_CELL_WIDTH = 4;
export const WORD_CHILD_START = 5;
export const WORD_CHILD_COUNT = 6;
export const WORD_PARTICLE_COUNT = 7;

const MAX_DEPTH_DEFAULT = 30;
const MAX_PARTICLES_PER_LEAF_DEFAULT = 1;
const PADDING = 0.01;

export interface FlatOctreeResult {
  buffer: ArrayBuffer;
  nodeCount: number;
}

export interface BuildOptions {
  maxParticlesPerLeaf?: number;
  maxDepth?: number;
  // Preallocated scratch. Caller may reuse across rebuilds to eliminate alloc.
  nodeScratch?: ArrayBuffer;
  indexScratch?: Int32Array;
  maxNodes?: number;
}

// Reusable scratch (module-level for tests; production path passes its own).
let defaultNodeScratch: ArrayBuffer | null = null;
let defaultIndexScratch: Int32Array | null = null;
let defaultMaxNodes = 0;

export function buildFlatOctree(
  particles: Float32Array,
  options: BuildOptions = {}
): FlatOctreeResult {
  const n = getParticleCount(particles);
  const maxParticlesPerLeaf = options.maxParticlesPerLeaf ?? MAX_PARTICLES_PER_LEAF_DEFAULT;
  const maxDepth = options.maxDepth ?? MAX_DEPTH_DEFAULT;
  const maxNodes = options.maxNodes ?? Math.max(8, n * 4 + 1);

  // Scratch: node buffer (caller-provided or reused module default).
  let nodeBuffer: ArrayBuffer;
  if (options.nodeScratch) {
    nodeBuffer = options.nodeScratch;
  } else {
    if (!defaultNodeScratch || defaultMaxNodes < maxNodes) {
      defaultNodeScratch = new ArrayBuffer(maxNodes * BYTES_PER_NODE);
      defaultMaxNodes = maxNodes;
    }
    nodeBuffer = defaultNodeScratch;
  }
  const floats = new Float32Array(nodeBuffer);
  const uints = new Uint32Array(nodeBuffer);

  // Index scratch.
  let idx: Int32Array;
  if (options.indexScratch && options.indexScratch.length >= Math.max(1, n)) {
    idx = options.indexScratch;
  } else {
    if (!defaultIndexScratch || defaultIndexScratch.length < Math.max(1, n)) {
      defaultIndexScratch = new Int32Array(Math.max(1, n));
    }
    idx = defaultIndexScratch;
  }
  for (let i = 0; i < n; i++) idx[i] = i;

  // Handle empty case: emit a single empty root node.
  if (n === 0) {
    writeLeafEmpty(floats, uints, 0);
    // Copy to right-sized output.
    return sliceResult(nodeBuffer, 1);
  }

  // Compute root bounds in a single scan.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < n; i++) {
    const o = i * FLOATS_PER_PARTICLE;
    const x = particles[o + OFFSET_X];
    const y = particles[o + OFFSET_Y];
    const z = particles[o + OFFSET_Z];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  minX -= PADDING; minY -= PADDING; minZ -= PADDING;
  maxX += PADDING; maxY += PADDING; maxZ += PADDING;

  // Builder state — closed over by buildNode. Single-pass DFS, nodes
  // allocated contiguously so each parent's children form a run.
  const state = {
    nextIdx: 1, // 0 reserved for root
    particles,
    idx,
    floats,
    uints,
    maxParticlesPerLeaf,
    maxDepth,
    maxNodes,
  };

  buildNode(state, 0, 0, n, minX, minY, minZ, maxX, maxY, maxZ, 0);

  return sliceResult(nodeBuffer, state.nextIdx);
}

function sliceResult(buf: ArrayBuffer, nodeCount: number): FlatOctreeResult {
  const usedBytes = nodeCount * BYTES_PER_NODE;
  // Return a copy sized to used bytes so callers can transfer it without
  // dragging the full scratch along. Callers that need zero-copy can pass
  // their own nodeScratch and slice themselves.
  const out = new ArrayBuffer(usedBytes);
  new Uint8Array(out).set(new Uint8Array(buf, 0, usedBytes));
  return { buffer: out, nodeCount };
}

interface BuildState {
  nextIdx: number;
  particles: Float32Array;
  idx: Int32Array;
  floats: Float32Array;
  uints: Uint32Array;
  maxParticlesPerLeaf: number;
  maxDepth: number;
  maxNodes: number;
}

function buildNode(
  s: BuildState,
  nodeIdx: number,
  start: number,
  end: number,
  minX: number, minY: number, minZ: number,
  maxX: number, maxY: number, maxZ: number,
  depth: number
): void {
  const count = end - start;
  const cellWidth = Math.max(maxX - minX, maxY - minY, maxZ - minZ);

  if (count <= s.maxParticlesPerLeaf || depth >= s.maxDepth) {
    writeLeaf(s, nodeIdx, start, end, cellWidth);
    return;
  }

  const midX = (minX + maxX) * 0.5;
  const midY = (minY + maxY) * 0.5;
  const midZ = (minZ + maxZ) * 0.5;

  // Partition indices [start, end) into 8 octants via 7 axis-partitions:
  // first by Z, then Y within each Z-half, then X within each YZ-quadrant.
  const zMid = partitionByAxis(s.idx, s.particles, start, end, OFFSET_Z, midZ);
  const yLo = partitionByAxis(s.idx, s.particles, start, zMid, OFFSET_Y, midY);
  const yHi = partitionByAxis(s.idx, s.particles, zMid, end, OFFSET_Y, midY);
  const xA = partitionByAxis(s.idx, s.particles, start, yLo, OFFSET_X, midX);
  const xB = partitionByAxis(s.idx, s.particles, yLo, zMid, OFFSET_X, midX);
  const xC = partitionByAxis(s.idx, s.particles, zMid, yHi, OFFSET_X, midX);
  const xD = partitionByAxis(s.idx, s.particles, yHi, end, OFFSET_X, midX);

  // Octant order: same bit pattern as the existing octree (bit 0=x, 1=y, 2=z).
  // Ranges:
  //   0: [start, xA)  --- (x<mid, y<mid, z<mid)
  //   1: [xA, yLo)    +-- (x>=mid, y<mid, z<mid)
  //   2: [yLo, xB)    -+- (x<mid, y>=mid, z<mid)
  //   3: [xB, zMid)   ++- (x>=mid, y>=mid, z<mid)
  //   4: [zMid, xC)   --+ (x<mid, y<mid, z>=mid)
  //   5: [xC, yHi)    +-+ (x>=mid, y<mid, z>=mid)
  //   6: [yHi, xD)    -++ (x<mid, y>=mid, z>=mid)
  //   7: [xD, end)    +++ (x>=mid, y>=mid, z>=mid)
  const rangeStart = [start, xA, yLo, xB, zMid, xC, yHi, xD];
  const rangeEnd = [xA, yLo, xB, zMid, xC, yHi, xD, end];

  // Reserve contiguous child slots for non-empty octants.
  let childCount = 0;
  for (let i = 0; i < 8; i++) if (rangeEnd[i] > rangeStart[i]) childCount++;

  const childStart = s.nextIdx;
  s.nextIdx += childCount;
  if (s.nextIdx > s.maxNodes) {
    throw new Error(`flatOctree: nodeCount exceeded maxNodes (${s.maxNodes})`);
  }

  // Recurse into children. After each returns, we aggregate its COM/mass.
  let totalMass = 0, cX = 0, cY = 0, cZ = 0;
  let totalParticles = 0;
  let c = 0;
  for (let i = 0; i < 8; i++) {
    const rs = rangeStart[i];
    const re = rangeEnd[i];
    if (re <= rs) continue;

    // Child bounds from parent bounds + octant bits.
    const bx0 = (i & 1) ? midX : minX;
    const bx1 = (i & 1) ? maxX : midX;
    const by0 = (i & 2) ? midY : minY;
    const by1 = (i & 2) ? maxY : midY;
    const bz0 = (i & 4) ? midZ : minZ;
    const bz1 = (i & 4) ? maxZ : midZ;

    const childIdx = childStart + c;
    buildNode(s, childIdx, rs, re, bx0, by0, bz0, bx1, by1, bz1, depth + 1);

    const offset = childIdx * NODE_WORD_STRIDE;
    const childMass = s.floats[offset + WORD_MASS];
    if (childMass > 0) {
      totalMass += childMass;
      cX += s.floats[offset + WORD_COM_X] * childMass;
      cY += s.floats[offset + WORD_COM_Y] * childMass;
      cZ += s.floats[offset + WORD_COM_Z] * childMass;
    }
    totalParticles += s.uints[offset + WORD_PARTICLE_COUNT];
    c++;
  }

  const nodeOffset = nodeIdx * NODE_WORD_STRIDE;
  if (totalMass > 0) {
    s.floats[nodeOffset + WORD_COM_X] = cX / totalMass;
    s.floats[nodeOffset + WORD_COM_Y] = cY / totalMass;
    s.floats[nodeOffset + WORD_COM_Z] = cZ / totalMass;
  } else {
    s.floats[nodeOffset + WORD_COM_X] = 0;
    s.floats[nodeOffset + WORD_COM_Y] = 0;
    s.floats[nodeOffset + WORD_COM_Z] = 0;
  }
  s.floats[nodeOffset + WORD_MASS] = totalMass;
  s.floats[nodeOffset + WORD_CELL_WIDTH] = cellWidth;
  s.uints[nodeOffset + WORD_CHILD_START] = childCount > 0 ? childStart : 0;
  s.uints[nodeOffset + WORD_CHILD_COUNT] = childCount;
  s.uints[nodeOffset + WORD_PARTICLE_COUNT] = totalParticles;
}

function writeLeaf(
  s: BuildState,
  nodeIdx: number,
  start: number,
  end: number,
  cellWidth: number
): void {
  let mass = 0, cX = 0, cY = 0, cZ = 0;
  for (let k = start; k < end; k++) {
    const o = s.idx[k] * FLOATS_PER_PARTICLE;
    const m = s.particles[o + OFFSET_MASS];
    mass += m;
    cX += s.particles[o + OFFSET_X] * m;
    cY += s.particles[o + OFFSET_Y] * m;
    cZ += s.particles[o + OFFSET_Z] * m;
  }

  const off = nodeIdx * NODE_WORD_STRIDE;
  if (mass > 0) {
    s.floats[off + WORD_COM_X] = cX / mass;
    s.floats[off + WORD_COM_Y] = cY / mass;
    s.floats[off + WORD_COM_Z] = cZ / mass;
  } else {
    s.floats[off + WORD_COM_X] = 0;
    s.floats[off + WORD_COM_Y] = 0;
    s.floats[off + WORD_COM_Z] = 0;
  }
  s.floats[off + WORD_MASS] = mass;
  s.floats[off + WORD_CELL_WIDTH] = cellWidth;
  s.uints[off + WORD_CHILD_START] = 0;
  s.uints[off + WORD_CHILD_COUNT] = 0;
  s.uints[off + WORD_PARTICLE_COUNT] = end - start;
}

function writeLeafEmpty(floats: Float32Array, uints: Uint32Array, nodeIdx: number): void {
  const o = nodeIdx * NODE_WORD_STRIDE;
  floats[o + WORD_COM_X] = 0;
  floats[o + WORD_COM_Y] = 0;
  floats[o + WORD_COM_Z] = 0;
  floats[o + WORD_MASS] = 0;
  floats[o + WORD_CELL_WIDTH] = 2 + 2 * PADDING;
  uints[o + WORD_CHILD_START] = 0;
  uints[o + WORD_CHILD_COUNT] = 0;
  uints[o + WORD_PARTICLE_COUNT] = 0;
}

/**
 * In-place Hoare partition of idx[start..end) by a particle axis value.
 * Indices with particles[idx[i] * stride + axisOffset] < mid end up first.
 * Returns the first index of the "upper" half.
 */
function partitionByAxis(
  idx: Int32Array,
  particles: Float32Array,
  start: number,
  end: number,
  axisOffset: number,
  mid: number
): number {
  let i = start;
  let j = end - 1;
  while (i <= j) {
    while (i <= j && particles[idx[i] * FLOATS_PER_PARTICLE + axisOffset] < mid) i++;
    while (i <= j && particles[idx[j] * FLOATS_PER_PARTICLE + axisOffset] >= mid) j--;
    if (i < j) {
      const t = idx[i];
      idx[i] = idx[j];
      idx[j] = t;
      i++;
      j--;
    }
  }
  return i;
}
