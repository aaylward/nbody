## 2024-05-23 - [Optimized CPU-GPU Data Transfer]
**Learning:** Matching CPU data layout to GPU alignment (32 bytes / 8 floats) significantly improved performance (~10% CPU speedup + faster initialization) by removing conversion overhead and improving cache locality.
**Action:** Always check data structure alignment when working with WebGPU or SIMD-heavy code. Avoid conversion loops by designing shared layouts.

## 2025-02-17 - [Optimized WebGPU Readback]
**Learning:** Compacting particle data (excluding padding/mass) into a dense buffer on GPU before readback reduced transfer size by 25%. This optimization is crucial for real-time CPU-GPU interoperability where bandwidth is the bottleneck.
**Action:** When reading back data from GPU, create a dedicated compact buffer and shader write step to transfer only essential visualization data.
