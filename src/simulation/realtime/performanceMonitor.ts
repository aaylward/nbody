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
  // Optimization: use circular buffers with Float64Array to avoid array allocations
  // and garbage collection overhead in continuous tracking systems
  private physicsTimings: Float64Array;
  private renderTimings: Float64Array;
  private physicsIdx = 0;
  private renderIdx = 0;
  private physicsCount = 0;
  private renderCount = 0;
  private readonly maxSamples = 60;

  constructor() {
    this.physicsTimings = new Float64Array(this.maxSamples);
    this.renderTimings = new Float64Array(this.maxSamples);
  }

  recordPhysicsFrame(duration: number): void {
    this.physicsTimings[this.physicsIdx] = duration;
    this.physicsIdx = (this.physicsIdx + 1) % this.maxSamples;
    if (this.physicsCount < this.maxSamples) this.physicsCount++;
  }

  recordRenderFrame(duration: number): void {
    this.renderTimings[this.renderIdx] = duration;
    this.renderIdx = (this.renderIdx + 1) % this.maxSamples;
    if (this.renderCount < this.maxSamples) this.renderCount++;
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
    this.physicsIdx = 0;
    this.renderIdx = 0;
    this.physicsCount = 0;
    this.renderCount = 0;
  }

  private average(arr: Float64Array, count: number): number {
    if (count === 0) return 0;
    let sum = 0;
    // Optimization: Manual loop is faster than reduce
    for (let i = 0; i < count; i++) {
        sum += arr[i];
    }
    return sum / count;
  }

  private percentile(arr: Float64Array, count: number, p: number): number {
    if (count === 0) return 0;

    // Optimization: For fixed small size arrays, TypedArray.prototype.sort()
    // is fast enough and maintains readability. We slice up to count to avoid mutating buffer
    const sorted = arr.slice(0, count).sort();

    const index = Math.floor(sorted.length * p);
    // clamp to valid array bounds
    return sorted[Math.max(0, Math.min(index, count - 1))];
  }
}
