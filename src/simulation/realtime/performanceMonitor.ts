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
    const index = Math.floor(arr.length * p);
    // Copy array because QuickSelect mutates it in-place
    const copy = [...arr];
    return this.quickSelect(copy, index);
  }

  private quickSelect(arr: number[], k: number): number {
    let left = 0, right = arr.length - 1;
    while (left < right) {
      const pivotIndex = this.partition(arr, left, right);
      if (pivotIndex === k) return arr[k];
      if (k < pivotIndex) right = pivotIndex - 1;
      else left = pivotIndex + 1;
    }
    return arr[k];
  }

  private partition(arr: number[], left: number, right: number): number {
    // Lomuto partition scheme
    // Use middle element as pivot to handle mostly sorted data well
    const mid = Math.floor((left + right) / 2);
    let temp = arr[mid];
    arr[mid] = arr[right];
    arr[right] = temp;

    const pivot = arr[right];
    let i = left;
    for (let j = left; j < right; j++) {
      if (arr[j] <= pivot) {
        temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        i++;
      }
    }
    temp = arr[i];
    arr[i] = arr[right];
    arr[right] = temp;
    return i;
  }
}
