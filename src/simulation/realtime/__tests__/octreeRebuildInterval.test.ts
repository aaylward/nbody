/**
 * Tests for octree rebuild interval and pipeline logic.
 *
 * The rebuild is pipelined across three frames and uses a Web Worker:
 *   Frame N:   kick off GPU→CPU download (async mapAsync)
 *   Frame N+1: read downloaded data, send to worker
 *   Frame N+2+: worker posts back serialized octree, upload to GPU
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

type RebuildPhase = 'idle' | 'downloading' | 'building';

/**
 * Simulates the three-phase rebuild pipeline from physicsLoop.
 *
 * Phase 1 ('idle' → 'downloading'): when framesSinceRebuild >= interval
 * Phase 2 ('downloading' → 'building'): read data, send to worker
 * Phase 3 ('building' → 'idle'): worker result received, upload
 *
 * We simulate the worker completing 1 frame after dispatch (best case).
 * Returns logs of which frames triggered each phase transition.
 */
function simulatePipeline(totalFrames: number, interval: number) {
  let phase: RebuildPhase = 'idle';
  let framesSinceRebuild = 0;
  let workerResultFrame = -1;

  const downloadFrames: number[] = [];
  const dispatchFrames: number[] = [];
  const uploadFrames: number[] = [];

  for (let frame = 0; frame < totalFrames; frame++) {
    // Simulate worker result arriving (1 frame after dispatch)
    let pendingResult = false;
    if (workerResultFrame === frame) {
      pendingResult = true;
      workerResultFrame = -1;
    }

    // Check for worker result first (mirrors loop ordering)
    if (pendingResult) {
      uploadFrames.push(frame);
      phase = 'idle';
      framesSinceRebuild = 0;
    }

    // Phase 2: downloading → building
    if (phase === 'downloading') {
      dispatchFrames.push(frame);
      phase = 'building';
      workerResultFrame = frame + 1; // Worker finishes next frame
    }

    // Count and check for new download
    framesSinceRebuild++;
    if (phase === 'idle' && framesSinceRebuild >= interval) {
      downloadFrames.push(frame);
      phase = 'downloading';
    }
  }

  return { downloadFrames, dispatchFrames, uploadFrames };
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
    it('first download triggers after interval frames', () => {
      const { downloadFrames } = simulatePipeline(20, 4);
      expect(downloadFrames[0]).toBe(3);
    });

    it('worker dispatch always follows download on the next frame', () => {
      const { downloadFrames, dispatchFrames } = simulatePipeline(40, 4);
      for (let i = 0; i < Math.min(downloadFrames.length, dispatchFrames.length); i++) {
        expect(dispatchFrames[i]).toBe(downloadFrames[i] + 1);
      }
    });

    it('upload follows dispatch by 1 frame (worker latency)', () => {
      const { dispatchFrames, uploadFrames } = simulatePipeline(40, 4);
      for (let i = 0; i < Math.min(dispatchFrames.length, uploadFrames.length); i++) {
        expect(uploadFrames[i]).toBe(dispatchFrames[i] + 1);
      }
    });

    it('no phase should overlap on the same frame', () => {
      const { downloadFrames, dispatchFrames, uploadFrames } = simulatePipeline(100, 4);
      // download and dispatch should never be on the same frame
      const dlSet = new Set(downloadFrames);
      for (const f of dispatchFrames) {
        expect(dlSet.has(f)).toBe(false);
      }
      // dispatch and upload should never be on the same frame
      const dispSet = new Set(dispatchFrames);
      for (const f of uploadFrames) {
        expect(dispSet.has(f)).toBe(false);
      }
    });

    it('should produce consistent rebuild cadence', () => {
      const { uploadFrames } = simulatePipeline(200, 8);
      for (let i = 1; i < uploadFrames.length; i++) {
        const gap = uploadFrames[i] - uploadFrames[i - 1];
        // Gap should be stable across rebuilds
        expect(gap).toBe(uploadFrames[1] - uploadFrames[0]);
      }
    });

    it('interval=1 should rebuild as fast as the pipeline allows', () => {
      const { uploadFrames } = simulatePipeline(30, 1);
      // Should get multiple rebuilds
      expect(uploadFrames.length).toBeGreaterThan(3);
    });
  });
});
