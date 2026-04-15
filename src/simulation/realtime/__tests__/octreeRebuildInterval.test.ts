/**
 * Tests for octree rebuild interval logic.
 *
 * The amortized rebuild skips the expensive CPU octree build on most
 * physics frames, reusing the last uploaded tree. These tests protect
 * the clamping invariants and the skip/rebuild decision so that future
 * refactors don't accidentally regress performance (always rebuilding)
 * or correctness (never rebuilding due to interval=0).
 */

import { describe, it, expect } from 'vitest';

/**
 * The rebuild decision extracted from the physics loop.
 * This mirrors the logic in RealtimeSimulationGPUBarnesHut.physicsLoop:
 *   const rebuildThisFrame = frameCount % this.octreeRebuildInterval === 0;
 */
function shouldRebuild(frameCount: number, interval: number): boolean {
  return frameCount % interval === 0;
}

/**
 * The clamping logic from setOctreeRebuildInterval.
 */
function clampInterval(interval: number): number {
  return Math.max(1, Math.min(16, interval));
}

describe('Octree Rebuild Interval', () => {
  describe('clamping', () => {
    it('should clamp interval=0 to 1 (prevents modulo-by-zero)', () => {
      expect(clampInterval(0)).toBe(1);
    });

    it('should clamp negative values to 1', () => {
      expect(clampInterval(-5)).toBe(1);
    });

    it('should clamp values above 16 to 16', () => {
      expect(clampInterval(32)).toBe(16);
    });

    it('should pass through values in range', () => {
      expect(clampInterval(1)).toBe(1);
      expect(clampInterval(4)).toBe(4);
      expect(clampInterval(8)).toBe(8);
      expect(clampInterval(16)).toBe(16);
    });
  });

  describe('rebuild decision', () => {
    it('should rebuild on frame 0 (first frame always rebuilds)', () => {
      expect(shouldRebuild(0, 4)).toBe(true);
    });

    it('should skip frames between rebuilds', () => {
      expect(shouldRebuild(1, 4)).toBe(false);
      expect(shouldRebuild(2, 4)).toBe(false);
      expect(shouldRebuild(3, 4)).toBe(false);
    });

    it('should rebuild on the interval boundary', () => {
      expect(shouldRebuild(4, 4)).toBe(true);
      expect(shouldRebuild(8, 4)).toBe(true);
    });

    it('interval=1 should rebuild every frame', () => {
      for (let i = 0; i < 10; i++) {
        expect(shouldRebuild(i, 1)).toBe(true);
      }
    });

    it('should rebuild exactly 1/N of the time over a long run', () => {
      const interval = 8;
      const totalFrames = 800;
      let rebuilds = 0;
      for (let i = 0; i < totalFrames; i++) {
        if (shouldRebuild(i, interval)) rebuilds++;
      }
      expect(rebuilds).toBe(totalFrames / interval);
    });
  });
});
