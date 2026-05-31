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
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      sum += arr[i];
    }
    return sum / arr.length;
  }

  private percentile(arr: number[], p: number): number {
    if (arr.length === 0) return 0;
    const index = Math.max(0, Math.min(Math.floor(arr.length * p), arr.length - 1));
    const copy = [...arr]; // Keep original array pure
    return this.quickSelect(copy, 0, copy.length - 1, index);
  }

  private quickSelect(arr: number[], left: number, right: number, k: number): number {
    if (left > right) return 0; // Should not happen with valid bounds, but prevents infinite recursion
    if (left === right) return arr[left];

    // Random pivot to avoid worst-case O(N^2)
    const pivotIndex = left + Math.floor(Math.random() * (right - left + 1));
    const pivot = arr[pivotIndex];

    let i = left;
    let lt = left;
    let gt = right;

    // Dutch National Flag 3-way partition
    // Handles identical elements gracefully
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

    if (k >= lt && k <= gt) return arr[k];
    if (k < lt) return this.quickSelect(arr, left, lt - 1, k);
    return this.quickSelect(arr, gt + 1, right, k);
  }
}
