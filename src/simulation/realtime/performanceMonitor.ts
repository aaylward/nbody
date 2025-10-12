/**
 * Performance monitoring for real-time simulation
 * Tracks physics and render FPS, timing statistics
 */

export interface PerformanceStats {
  physicsFPS: number;
  renderFPS: number;
  physicsAvg: number;
  physicsP95: number;
  renderAvg: number;
  renderP95: number;
}

export class PerformanceMonitor {
  private physicsTimings: number[] = [];
  private renderTimings: number[] = [];
  private readonly maxSamples = 60;

  recordPhysicsFrame(duration: number): void {
    this.physicsTimings.push(duration);
    if (this.physicsTimings.length > this.maxSamples) {
      this.physicsTimings.shift();
    }
  }

  recordRenderFrame(duration: number): void {
    this.renderTimings.push(duration);
    if (this.renderTimings.length > this.maxSamples) {
      this.renderTimings.shift();
    }
  }

  getPhysicsFPS(): number {
    const avg = this.average(this.physicsTimings);
    return avg > 0 ? 1000 / avg : 0;
  }

  getRenderFPS(): number {
    const avg = this.average(this.renderTimings);
    return avg > 0 ? 1000 / avg : 0;
  }

  getStats(): PerformanceStats {
    return {
      physicsFPS: this.getPhysicsFPS(),
      renderFPS: this.getRenderFPS(),
      physicsAvg: this.average(this.physicsTimings),
      physicsP95: this.percentile(this.physicsTimings, 0.95),
      renderAvg: this.average(this.renderTimings),
      renderP95: this.percentile(this.renderTimings, 0.95),
    };
  }

  reset(): void {
    this.physicsTimings = [];
    this.renderTimings = [];
  }

  private average(arr: number[]): number {
    if (arr.length === 0) return 0;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private percentile(arr: number[], p: number): number {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const index = Math.floor(sorted.length * p);
    return sorted[index];
  }
}
