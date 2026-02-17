## 2024-05-23 - [Optimized CPU-GPU Data Transfer]
**Learning:** Matching CPU data layout to GPU alignment (32 bytes / 8 floats) significantly improved performance (~10% CPU speedup + faster initialization) by removing conversion overhead and improving cache locality.
**Action:** Always check data structure alignment when working with WebGPU or SIMD-heavy code. Avoid conversion loops by designing shared layouts.
