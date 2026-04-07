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
  const oneMinusAlpha = 1 - alpha;

  // Linear interpolation for all components
  for (let i = 0; i < frame0.length; i++) {
    result[i] = frame0[i] * oneMinusAlpha + frame1[i] * alpha;
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

  const numParticles = frame0.length / FLOATS_PER_PARTICLE;
  const result = new Float32Array(frame0.length);

  // Hermite basis functions
  const t = alpha;
  const t2 = t * t;
  const t3 = t2 * t;

  const h00 = 2 * t3 - 3 * t2 + 1; // Position at t=0
  const h10 = t3 - 2 * t2 + t; // Velocity at t=0
  const h01 = -2 * t3 + 3 * t2; // Position at t=1
  const h11 = t3 - t2; // Velocity at t=1

  const posOffsets = [OFFSET_X, OFFSET_Y, OFFSET_Z];
  const velOffsets = [OFFSET_VX, OFFSET_VY, OFFSET_VZ];
  const oneMinusAlpha = 1 - alpha;

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;

    // Hermite interpolation for position (x, y, z) using corresponding velocity components
    for (let j = 0; j < 3; j++) {
      const pos0 = frame0[offset + posOffsets[j]];
      const vel0 = frame0[offset + velOffsets[j]];
      const pos1 = frame1[offset + posOffsets[j]];
      const vel1 = frame1[offset + velOffsets[j]];

      result[offset + posOffsets[j]] = h00 * pos0 + h10 * vel0 + h01 * pos1 + h11 * vel1;
    }

    // Linear interpolation for velocity (vx, vy, vz)
    for (let j = 0; j < 3; j++) {
      const v = velOffsets[j];
      result[offset + v] = frame0[offset + v] * oneMinusAlpha + frame1[offset + v] * alpha;
    }

    // Mass doesn't interpolate
    result[offset + OFFSET_MASS] = frame0[offset + OFFSET_MASS];
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
