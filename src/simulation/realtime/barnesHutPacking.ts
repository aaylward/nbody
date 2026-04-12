/**
 * Pure typed-array transforms for moving particle data between
 * the CPU layout (stride 8: [x,y,z,pad,vx,vy,vz,mass]) and the
 * GPU layout used by the Barnes-Hut shaders (stride 4: [x,y,z,mass]
 * for positions, [vx,vy,vz,pad] for velocities).
 */

import {
  FLOATS_PER_PARTICLE,
  OFFSET_X,
  OFFSET_Y,
  OFFSET_Z,
  OFFSET_VX,
  OFFSET_VY,
  OFFSET_VZ,
  OFFSET_MASS,
} from '../particleData';

/** CPU stride-8 → GPU [x, y, z, mass] stride-4 */
export function packParticlesForGPU(cpu: Float32Array, n: number): Float32Array<ArrayBuffer> {
  const gpu = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const src = i * FLOATS_PER_PARTICLE;
    const dst = i * 4;
    gpu[dst + 0] = cpu[src + OFFSET_X];
    gpu[dst + 1] = cpu[src + OFFSET_Y];
    gpu[dst + 2] = cpu[src + OFFSET_Z];
    gpu[dst + 3] = cpu[src + OFFSET_MASS];
  }
  return gpu;
}

/** CPU stride-8 → GPU [vx, vy, vz, 0] stride-4 (vec3f 16-byte stride) */
export function packVelocitiesForGPU(cpu: Float32Array, n: number): Float32Array<ArrayBuffer> {
  const gpu = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    const src = i * FLOATS_PER_PARTICLE;
    const dst = i * 4;
    gpu[dst + 0] = cpu[src + OFFSET_VX];
    gpu[dst + 1] = cpu[src + OFFSET_VY];
    gpu[dst + 2] = cpu[src + OFFSET_VZ];
  }
  return gpu;
}

/** GPU [x, y, z, mass] stride-4 → CPU stride-8 (velocities untouched) */
export function unpackParticlesFromGPU(
  gpuData: Float32Array,
  cpu: Float32Array,
  n: number,
): void {
  for (let i = 0; i < n; i++) {
    const src = i * 4;
    const dst = i * FLOATS_PER_PARTICLE;
    cpu[dst + OFFSET_X] = gpuData[src + 0];
    cpu[dst + OFFSET_Y] = gpuData[src + 1];
    cpu[dst + OFFSET_Z] = gpuData[src + 2];
    cpu[dst + OFFSET_MASS] = gpuData[src + 3];
  }
}
