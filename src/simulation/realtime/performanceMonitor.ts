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
  private physicsIndex = 0;
  private physicsCount = 0;

  private renderTimings = new Float64Array(this.maxSamples);
  private renderIndex = 0;
  private renderCount = 0;

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
    this.physicsIndex = 0;
    this.physicsCount = 0;
    this.renderIndex = 0;
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

    // Create a copy of only the valid elements for QuickSelect
    const workingCopy = new Float64Array(arr.buffer, 0, count).slice();

    // Clamp target index
    const targetIndex = Math.max(0, Math.min(Math.floor(count * p), count - 1));

    return this.quickSelect(workingCopy, 0, count - 1, targetIndex);
  }

  private quickSelect(arr: Float64Array, left: number, right: number, k: number): number {
    while (left < right) {
      // Choose random pivot
      // Note: We use Math.random() here for QuickSelect, which is acceptable for
      // array partitioning (doesn't require cryptographically secure randomness)
      const pivotIndex = left + Math.floor(Math.random() * (right - left + 1));
      const pivot = arr[pivotIndex];

      // Dutch National Flag 3-way partition
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

      if (k < lt) {
        right = lt - 1;
      } else if (k > gt) {
        left = gt + 1;
      } else {
        return pivot;
      }
    }

    return arr[left];
  }
}
