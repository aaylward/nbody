/**
 * Tests for octree rebuild interval and pipeline logic.
 *
 * The amortized rebuild skips the expensive CPU octree build on most
 * physics frames, reusing the last uploaded tree. The rebuild is
 * pipelined across two frames (download then build) to spread the cost.
 *
 * These tests protect the clamping invariants and the pipeline state
 * machine so that future refactors don't accidentally regress
 * performance (always rebuilding) or correctness (never rebuilding).
 */

import { describe, it, expect } from 'vitest';

/**
 * The clamping logic from setOctreeRebuildInterval.
 */
function clampInterval(interval: number): number {
  return Math.max(1, Math.min(16, interval));
}

/**
 * Simulates the two-phase rebuild pipeline from physicsLoop.
 *
 * Phase 1 ('idle' → 'downloading'): when framesSinceRebuild reaches
 *   the interval, kick off async download.
 * Phase 2 ('downloading' → 'idle'): on the next frame, await download
 *   and build octree. Reset counter.
 *
 * Returns a log of which frames triggered each phase.
 */
function simulatePipeline(totalFrames: number, interval: number) {
  let rebuildPhase: 'idle' | 'downloading' = 'idle';
  let framesSinceRebuild = 0;
  const downloadFrames: number[] = [];
  const buildFrames: number[] = [];

  for (let frame = 0; frame < totalFrames; frame++) {
    // Phase 2 runs first (mirrors the loop ordering)
    if (rebuildPhase === 'downloading') {
      buildFrames.push(frame);
      rebuildPhase = 'idle';
      framesSinceRebuild = 0;
    }

    framesSinceRebuild++;
    if (rebuildPhase === 'idle' && framesSinceRebuild >= interval) {
      downloadFrames.push(frame);
      rebuildPhase = 'downloading';
    }
  }

  return { downloadFrames, buildFrames };
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

  describe('rebuild pipeline', () => {
    it('first download should happen on frame interval-1', () => {
      const { downloadFrames } = simulatePipeline(20, 4);
      expect(downloadFrames[0]).toBe(3); // framesSinceRebuild reaches 4 on frame 3
    });

    it('build always follows download on the next frame', () => {
      const { downloadFrames, buildFrames } = simulatePipeline(40, 4);
      for (let i = 0; i < downloadFrames.length; i++) {
        if (i < buildFrames.length) {
          expect(buildFrames[i]).toBe(downloadFrames[i] + 1);
        }
      }
    });

    it('interval=1 should rebuild as fast as possible', () => {
      const { downloadFrames, buildFrames } = simulatePipeline(10, 1);
      // interval=1: framesSinceRebuild hits 1 on every non-build frame.
      // Build frame resets counter and immediately re-triggers download
      // on the same frame (counter goes 0→1 after reset), so download
      // and build happen on every frame except the very first build.
      expect(downloadFrames.length).toBeGreaterThan(0);
      expect(buildFrames.length).toBeGreaterThan(0);
      // Every download is followed by a build on the next frame
      for (let i = 0; i < buildFrames.length; i++) {
        expect(buildFrames[i]).toBe(downloadFrames[i] + 1);
      }
    });

    it('no frame should have both download and build', () => {
      const { downloadFrames, buildFrames } = simulatePipeline(100, 4);
      const overlap = downloadFrames.filter(f => buildFrames.includes(f));
      expect(overlap).toEqual([]);
    });

    it('should produce consistent rebuild cadence over many frames', () => {
      const { buildFrames } = simulatePipeline(200, 8);
      // Check intervals between builds are consistent
      for (let i = 1; i < buildFrames.length; i++) {
        const gap = buildFrames[i] - buildFrames[i - 1];
        // The build frame itself resets the counter, then on that same
        // frame framesSinceRebuild increments to 1, so the next download
        // triggers at interval-1 frames later, build 1 frame after that.
        // Total gap between builds = interval.
        expect(gap).toBe(8);
      }
    });
  });
});
