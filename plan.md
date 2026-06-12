# Execution Plan
1.  **Optimize `extractPositions`:**
    - Replace sequential post-increments (`positions[pIdx++]`) with explicit index assignments (`positions[pIdx]`, `positions[pIdx + 1]`, `positions[pIdx + 2]`).
    - Increment `pIdx += 3` per iteration.

2.  **Optimize `extractVelocities`:**
    - Replace sequential post-increments (`velocities[vIdx++]`) with explicit index assignments (`velocities[vIdx]`, `velocities[vIdx + 1]`, `velocities[vIdx + 2]`).
    - Increment `vIdx += 3` per iteration.
    - Add an optional `out?: Float32Array` parameter to avoid allocations, matching the `extractPositions` signature.

3.  **Optimize `toParticleObjects`:**
    - Add an optional `out?: Array<{...}>` parameter.
    - Synchronize the `length` property of the `out` array to `numParticles` (`out.length = numParticles`).
    - Update elements in-place if they already exist in the `out` array instead of allocating new objects. Create new objects only if they are missing at a given index.
    - Return the updated `out` array or newly created `particles` array.

4.  **Complete pre-commit steps:**
    - Ensure proper testing, verification, review, and reflection are done using `pre_commit_instructions`.

5.  **Submit the change:**
    - Run the vitest test suite.
    - Submit the change using a descriptive commit message with PR format.
