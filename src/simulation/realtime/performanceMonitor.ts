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
  private physicsTimings: Float64Array;
  private renderTimings: Float64Array;
  private physicsIndex = 0;
  private renderIndex = 0;
  private physicsCount = 0;
  private renderCount = 0;
  private readonly maxSamples = 60;

  constructor() {
    this.physicsTimings = new Float64Array(this.maxSamples);
    this.renderTimings = new Float64Array(this.maxSamples);
  }

  recordPhysicsFrame(duration: number): void {
    this.physicsTimings[this.physicsIndex] = duration;
    this.physicsIndex = (this.physicsIndex + 1) % this.maxSamples;
    if (this.physicsCount < this.maxSamples) this.physicsCount++;
  }

  recordRenderFrame(duration: number): void {
    this.renderTimings[this.renderIndex] = duration;
    this.renderIndex = (this.renderIndex + 1) % this.maxSamples;
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
    this.physicsIndex = 0;
    this.renderIndex = 0;
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

    // Create a copy of the valid portion for quickselect
    const data = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      data[i] = arr[i];
    }

    const targetIdx = Math.max(0, Math.min(Math.floor(count * p), count - 1));
    return this.quickSelect(data, 0, count - 1, targetIdx);
  }

  private quickSelect(arr: Float64Array, left: number, right: number, k: number): number {
    if (left === right) return arr[left];
    if (left > right) return 0;

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
      return this.quickSelect(arr, left, lt - 1, k);
    } else {
      return this.quickSelect(arr, gt + 1, right, k);
    }
  }
}
