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

  private physicsIndex = 0;
  private renderIndex = 0;

  recordPhysicsFrame(duration: number): void {
    this.physicsTimings[this.physicsIndex] = duration;
    this.physicsIndex = (this.physicsIndex + 1) % this.maxSamples;
    if (this.physicsCount < this.maxSamples) {
      this.physicsCount++;
    }
  }

  recordRenderFrame(duration: number): void {
    this.renderTimings[this.renderIndex] = duration;
    this.renderIndex = (this.renderIndex + 1) % this.maxSamples;
    if (this.renderCount < this.maxSamples) {
      this.renderCount++;
    }
  }

  getPhysicsFPS(): number {
    const avg = this.average(this.physicsTimings, this.physicsCount);
    return avg > 0 ? 1000 / avg : 0;
  }

  getRenderFPS(): number {
    const avg = this.average(this.renderTimings, this.renderCount);
    return avg > 0 ? 1000 / avg : 0;
  }

  getStats(): PerformanceStats {
    return {
      physicsFPS: this.getPhysicsFPS(),
      renderFPS: this.getRenderFPS(),
      physicsAvg: this.average(this.physicsTimings, this.physicsCount),
      physicsP95: this.percentile(this.physicsTimings, this.physicsCount, 0.95),
      renderAvg: this.average(this.renderTimings, this.renderCount),
      renderP95: this.percentile(this.renderTimings, this.renderCount, 0.95),
    };
  }

  reset(): void {
    this.physicsCount = 0;
    this.renderCount = 0;
    this.physicsIndex = 0;
    this.renderIndex = 0;
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
    // TypedArray.prototype.sort() sorts numerically in-place
    const sorted = arr.slice(0, count).sort();
    const index = Math.max(0, Math.min(Math.floor(count * p), count - 1));
    return sorted[index];
  }
}
