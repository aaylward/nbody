/**
 * Tests for PerformanceMonitor
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { PerformanceMonitor } from '../performanceMonitor';

describe('PerformanceMonitor', () => {
  let monitor: PerformanceMonitor;

  beforeEach(() => {
    monitor = new PerformanceMonitor();
  });

  describe('recordPhysicsFrame', () => {
    it('should record physics frame timings', () => {
      monitor.recordPhysicsFrame(50);
      monitor.recordPhysicsFrame(60);
      monitor.recordPhysicsFrame(40);

      const stats = monitor.getStats();
      expect(stats.physicsAvg).toBeCloseTo(50);
    });

    it('should calculate correct FPS from frame timings', () => {
      monitor.recordPhysicsFrame(50); // 50ms = 20 FPS

      const stats = monitor.getStats();
      expect(stats.physicsFPS).toBeCloseTo(20, 0);
    });

    it('should limit to maxSamples (60)', () => {
      // Record 100 frames
      for (let i = 0; i < 100; i++) {
        monitor.recordPhysicsFrame(50);
      }

      const stats = monitor.getStats();
      // Should still work correctly (only last 60 samples)
      expect(stats.physicsAvg).toBeCloseTo(50);
    });
  });

  describe('recordRenderFrame', () => {
    it('should record render frame timings', () => {
      monitor.recordRenderFrame(16.67); // ~60 FPS
      monitor.recordRenderFrame(16.67);
      monitor.recordRenderFrame(16.67);

      const stats = monitor.getStats();
      expect(stats.renderAvg).toBeCloseTo(16.67, 1);
    });

    it('should calculate correct FPS from render timings', () => {
      monitor.recordRenderFrame(16.67); // 16.67ms = 60 FPS

      const stats = monitor.getStats();
      expect(stats.renderFPS).toBeCloseTo(60, 0);
    });
  });

  describe('getPhysicsFPS', () => {
    it('should return 0 when no frames recorded', () => {
      const fps = monitor.getPhysicsFPS();
      expect(fps).toBe(0);
    });

    it('should return correct FPS for various timings', () => {
      monitor.recordPhysicsFrame(100); // 100ms = 10 FPS
      expect(monitor.getPhysicsFPS()).toBeCloseTo(10, 0);

      monitor.reset();

      monitor.recordPhysicsFrame(20); // 20ms = 50 FPS
      expect(monitor.getPhysicsFPS()).toBeCloseTo(50, 0);
    });
  });

  describe('getRenderFPS', () => {
    it('should return 0 when no frames recorded', () => {
      const fps = monitor.getRenderFPS();
      expect(fps).toBe(0);
    });

    it('should return correct FPS for various timings', () => {
      monitor.recordRenderFrame(16.67); // ~60 FPS
      expect(monitor.getRenderFPS()).toBeCloseTo(60, 0);

      monitor.reset();

      monitor.recordRenderFrame(33.33); // ~30 FPS
      expect(monitor.getRenderFPS()).toBeCloseTo(30, 0);
    });
  });

  describe('getStats', () => {
    it('should return comprehensive statistics', () => {
      monitor.recordPhysicsFrame(50);
      monitor.recordPhysicsFrame(60);
      monitor.recordPhysicsFrame(40);
      monitor.recordPhysicsFrame(70);
      monitor.recordPhysicsFrame(55);

      monitor.recordRenderFrame(15);
      monitor.recordRenderFrame(16);
      monitor.recordRenderFrame(17);

      const stats = monitor.getStats();

      expect(stats).toHaveProperty('physicsFPS');
      expect(stats).toHaveProperty('renderFPS');
      expect(stats).toHaveProperty('physicsAvg');
      expect(stats).toHaveProperty('physicsP95');
      expect(stats).toHaveProperty('renderAvg');
      expect(stats).toHaveProperty('renderP95');

      expect(stats.physicsAvg).toBeCloseTo(55);
      expect(stats.renderAvg).toBeCloseTo(16);
    });

    it('should calculate P95 percentile correctly', () => {
      // Record values from 10 to 109
      for (let i = 10; i < 110; i++) {
        monitor.recordPhysicsFrame(i);
      }

      const stats = monitor.getStats();
      // P95 of last 60 samples (50-109) should be around 106
      expect(stats.physicsP95).toBeGreaterThan(100);
    });
  });

  describe('reset', () => {
    it('should clear all recorded timings', () => {
      monitor.recordPhysicsFrame(50);
      monitor.recordRenderFrame(16);

      monitor.reset();

      const stats = monitor.getStats();
      expect(stats.physicsFPS).toBe(0);
      expect(stats.renderFPS).toBe(0);
      expect(stats.physicsAvg).toBe(0);
      expect(stats.renderAvg).toBe(0);
    });
  });

  describe('percentile calculation', () => {
    it('should handle edge cases', () => {
      monitor.recordPhysicsFrame(10);

      const stats = monitor.getStats();
      expect(stats.physicsP95).toBe(10);
    });

    it('should calculate percentiles for multiple values', () => {
      const values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
      values.forEach((v) => monitor.recordPhysicsFrame(v));

      const stats = monitor.getStats();
      // P95 of [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] is 100
      expect(stats.physicsP95).toBe(100);
    });
  });
});
