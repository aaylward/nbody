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
    // ⚡ Bolt: Replaced Array.reduce with a standard for-loop to avoid callback allocation
    // and function call overhead per element, achieving faster average computation in V8.
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
    }
    return sum / arr.length;
  }

  private percentile(arr: number[], p: number): number {
    if (arr.length === 0) return 0;
    // Clamp the target index to valid bounds
    const index = Math.max(0, Math.min(Math.floor(arr.length * p), arr.length - 1));
    // ⚡ Bolt: Replaced O(N log N) full array sort ([...arr].sort()) with O(N) QuickSelect.
    // This provides a substantial performance boost for high-frequency percentile calculations.
    return this.quickSelect([...arr], index);
  }

  private quickSelect(arr: number[], k: number): number {
    let left = 0;
    let right = arr.length - 1;

    while (left <= right) {
      if (left === right) return arr[left];

      const pivotIndex = left + Math.floor(Math.random() * (right - left + 1));
      const pivot = arr[pivotIndex];

      let lt = left;
      let gt = right;
      let i = left;

      while (i <= gt) {
        if (arr[i] < pivot) {
          const temp = arr[lt];
          arr[lt] = arr[i];
          arr[i] = temp;
          lt++;
          i++;
        } else if (arr[i] > pivot) {
          const temp = arr[gt];
          arr[gt] = arr[i];
          arr[i] = temp;
          gt--;
        } else {
          i++;
        }
      }

      if (k >= lt && k <= gt) {
        return arr[k];
      } else if (k < lt) {
        right = lt - 1;
      } else {
        left = gt + 1;
      }
    }

    return 0; // safety base case
  }

}
