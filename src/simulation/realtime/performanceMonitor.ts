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
  private readonly maxSamples = 60;
  private physicsTimings = new Float64Array(this.maxSamples);
  private renderTimings = new Float64Array(this.maxSamples);
  private physicsCount = 0;
  private renderCount = 0;

  recordPhysicsFrame(duration: number): void {
    this.physicsTimings[this.physicsCount % this.maxSamples] = duration;
    this.physicsCount++;
  }

  recordRenderFrame(duration: number): void {
    this.renderTimings[this.renderCount % this.maxSamples] = duration;
    this.renderCount++;
  }

  getPhysicsFPS(): number {
    const avg = this.average(this.physicsTimings, Math.min(this.physicsCount, this.maxSamples));
    return avg > 0 ? 1000 / avg : 0;
  }

  getRenderFPS(): number {
    const avg = this.average(this.renderTimings, Math.min(this.renderCount, this.maxSamples));
    return avg > 0 ? 1000 / avg : 0;
  }

  getStats(): PerformanceStats {
    const pCount = Math.min(this.physicsCount, this.maxSamples);
    const rCount = Math.min(this.renderCount, this.maxSamples);
    return {
      physicsFPS: this.getPhysicsFPS(),
      renderFPS: this.getRenderFPS(),
      physicsAvg: this.average(this.physicsTimings, pCount),
      physicsP95: this.percentile(this.physicsTimings, pCount, 0.95),
      renderAvg: this.average(this.renderTimings, rCount),
      renderP95: this.percentile(this.renderTimings, rCount, 0.95),
    };
  }

  reset(): void {
    this.physicsCount = 0;
    this.renderCount = 0;
  }

  private average(arr: Float64Array, count: number): number {
    if (count === 0) return 0;
    let sum = 0;
    for (let i = 0; i < count; i++) {
        sum += arr[i];
    }
    return sum / count;
  }

  private percentile(arr: Float64Array, count: number, p: number): number {
    if (count === 0) return 0;
    // For small arrays (60 elements), TypedArray sort is fast enough and operates in-place on a slice copy
    const sorted = arr.slice(0, count).sort();
    const index = Math.min(Math.floor(count * p), count - 1);
    return sorted[index];
  }
}
