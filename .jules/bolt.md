## 2024-05-23 - [Optimized CPU-GPU Data Transfer]
**Learning:** Matching CPU data layout to GPU alignment (32 bytes / 8 floats) significantly improved performance (~10% CPU speedup + faster initialization) by removing conversion overhead and improving cache locality.
**Action:** Always check data structure alignment when working with WebGPU or SIMD-heavy code. Avoid conversion loops by designing shared layouts.

## 2025-02-17 - [Optimized WebGPU Readback]
**Learning:** Compacting particle data (excluding padding/mass) into a dense buffer on GPU before readback reduced transfer size by 25%. This optimization is crucial for real-time CPU-GPU interoperability where bandwidth is the bottleneck.
**Action:** When reading back data from GPU, create a dedicated compact buffer and shader write step to transfer only essential visualization data.

## 2025-02-28 - [Optimized Memory Iteration]
**Learning:** In performance-critical loops (e.g., `interpolateSnapshots`), iterating directly by memory offset (`offset += FLOATS_PER_PARTICLE`) instead of recalculating the offset via multiplication per particle (`p * FLOATS_PER_PARTICLE`) yields >30% performance improvements in V8.
**Action:** When iterating over packed structures in TypedArrays, hoist the multiplication (`numParticles * FLOATS_PER_PARTICLE`) to find the total size and iterate using addition (`offset += stride`) rather than recalculating via multiplication within the loop body.

## 2025-03-04 - [Optimized Snapshot Interpolation]
**Learning:** Using `TypedArray.set()` for a fast memory copy of the initial state, followed by in-place linear interpolation `a += (b - a) * t`, reduces mathematical operations from 2 multiplies/1 add to 1 multiply/1 add/1 subtract per component. This avoids updating invariant properties (like mass) inside the loop and significantly speeds up interpolation in V8.
**Action:** When interpolating large packed structures (e.g. simulation frames), initialize the destination array via `.set(sourceArray)` to copy invariant values instantly, then only calculate the delta (`b - a`) multiplied by `t` for changing values.

## 2025-03-05 - [Optimized CPU N-Body Force Calculation]
**Learning:** In the O(N^2) CPU physics simulation loop, merging mathematical operations (`r = Math.sqrt(r2)` followed by `f = (Gim * jm) / (r2 * r)`) into a single operation (`f = (Gim * jm) / (r2 * Math.sqrt(r2))`) and accessing values from TypedArrays directly without intermediate assignments improved execution time by ~5-10% in V8. Interestingly, attempting to precalculate multipliers outside the inner loop (e.g., `Gim / (r2 * r)`) sometimes led to de-optimizations in the JIT, indicating that simplifying the math expression inline was the most effective approach.
**Action:** When working in tight mathematical loops, consider folding expressions (like `r2 * Math.sqrt(r2)`) to avoid intermediate allocations and assignments.
