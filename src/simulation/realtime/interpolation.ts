/**
 * Temporal interpolation for smooth rendering at 60 FPS
 * from lower physics framerate (e.g., 20 FPS)
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

/**
 * Simple linear interpolation between two particle frames
 * @param frame0 First particle frame
 * @param frame1 Second particle frame
 * @param alpha Interpolation factor (0.0 = frame0, 1.0 = frame1)
 * @returns Interpolated frame
 */
export function interpolateParticles(
  frame0: Float32Array,
  frame1: Float32Array,
  alpha: number
): Float32Array {
  if (frame0.length !== frame1.length) {
    throw new Error('Frame arrays must have the same length');
  }

  const result = new Float32Array(frame0.length);
  // Optimization: Fast copy of initial state (preserves mass and padding)
  result.set(frame0);

  // Linear interpolation for all changing components using offset iterations
  for (let offset = 0; offset < frame0.length; offset += FLOATS_PER_PARTICLE) {
    result[offset + OFFSET_X] += (frame1[offset + OFFSET_X] - frame0[offset + OFFSET_X]) * alpha;
    result[offset + OFFSET_Y] += (frame1[offset + OFFSET_Y] - frame0[offset + OFFSET_Y]) * alpha;
    result[offset + OFFSET_Z] += (frame1[offset + OFFSET_Z] - frame0[offset + OFFSET_Z]) * alpha;

    result[offset + OFFSET_VX] += (frame1[offset + OFFSET_VX] - frame0[offset + OFFSET_VX]) * alpha;
    result[offset + OFFSET_VY] += (frame1[offset + OFFSET_VY] - frame0[offset + OFFSET_VY]) * alpha;
    result[offset + OFFSET_VZ] += (frame1[offset + OFFSET_VZ] - frame0[offset + OFFSET_VZ]) * alpha;
  }

  return result;
}

/**
 * Hermite interpolation using velocities for smoother motion
 * @param frame0 First particle frame
 * @param frame1 Second particle frame
 * @param alpha Interpolation factor (0.0 = frame0, 1.0 = frame1)
 * @returns Smoothly interpolated frame
 */
export function interpolateParticlesSmooth(
  frame0: Float32Array,
  frame1: Float32Array,
  alpha: number
): Float32Array {
  if (frame0.length !== frame1.length) {
    throw new Error('Frame arrays must have the same length');
  }

  const result = new Float32Array(frame0.length);
  // Optimization: Fast copy of initial state (preserves mass and padding)
  result.set(frame0);

  // Hermite basis functions
  const t = alpha;
  const t2 = t * t;
  const t3 = t2 * t;

  const h00 = 2 * t3 - 3 * t2 + 1; // Position at t=0
  const h10 = t3 - 2 * t2 + t; // Velocity at t=0
  const h01 = -2 * t3 + 3 * t2; // Position at t=1
  const h11 = t3 - t2; // Velocity at t=1

  const oneMinusAlpha = 1 - alpha;

  for (let offset = 0; offset < frame0.length; offset += FLOATS_PER_PARTICLE) {
    // Hermite interpolation for position (x, y, z) using corresponding velocity components
    const v0x = frame0[offset + OFFSET_VX];
    const v0y = frame0[offset + OFFSET_VY];
    const v0z = frame0[offset + OFFSET_VZ];

    result[offset + OFFSET_X] = h00 * frame0[offset + OFFSET_X] + h10 * v0x + h01 * frame1[offset + OFFSET_X] + h11 * frame1[offset + OFFSET_VX];
    result[offset + OFFSET_Y] = h00 * frame0[offset + OFFSET_Y] + h10 * v0y + h01 * frame1[offset + OFFSET_Y] + h11 * frame1[offset + OFFSET_VY];
    result[offset + OFFSET_Z] = h00 * frame0[offset + OFFSET_Z] + h10 * v0z + h01 * frame1[offset + OFFSET_Z] + h11 * frame1[offset + OFFSET_VZ];

    // Linear interpolation for velocity (vx, vy, vz)
    result[offset + OFFSET_VX] = v0x * oneMinusAlpha + frame1[offset + OFFSET_VX] * alpha;
    result[offset + OFFSET_VY] = v0y * oneMinusAlpha + frame1[offset + OFFSET_VY] * alpha;
    result[offset + OFFSET_VZ] = v0z * oneMinusAlpha + frame1[offset + OFFSET_VZ] * alpha;
  }

  return result;
}

/**
 * Estimate the optimal interpolation alpha based on time
 * @param lastUpdateTime Time of last physics update (ms)
 * @param currentTime Current time (ms)
 * @param physicsInterval Expected physics update interval (ms)
 * @returns Alpha value clamped to [0, 1]
 */
export function calculateInterpolationAlpha(
  lastUpdateTime: number,
  currentTime: number,
  physicsInterval: number
): number {
  const elapsed = currentTime - lastUpdateTime;
  const alpha = elapsed / physicsInterval;
  return Math.max(0, Math.min(1, alpha));
}
