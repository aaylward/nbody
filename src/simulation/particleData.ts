/**
 * Efficient TypedArray-based particle data storage
 *
 * Each particle is stored as 8 consecutive floats (32 bytes for F32, 64 bytes for F64):
 * [x, y, z, pad, vx, vy, vz, mass]
 *
 * The padding ensures 32-byte alignment for better cache locality and SIMD,
 * and matches the GPU layout for direct memory transfer.
 *
 * This reduces memory overhead by ~70% compared to object arrays
 * and enables efficient GPU transfer and processing.
 */

export const FLOATS_PER_PARTICLE = 8;

// Offsets within each particle's data
export const OFFSET_X = 0;
export const OFFSET_Y = 1;
export const OFFSET_Z = 2;
export const OFFSET_VX = 4;
export const OFFSET_VY = 5;
export const OFFSET_VZ = 6;
export const OFFSET_MASS = 7;

/**
 * Create a new particle data array
 */
export function createParticleArray(numParticles: number): Float32Array {
  return new Float32Array(numParticles * FLOATS_PER_PARTICLE);
}

/**
 * Get particle data by index
 */
export function getParticle(data: Float32Array | Float64Array, index: number) {
  const offset = index * FLOATS_PER_PARTICLE;
  return {
    x: data[offset + OFFSET_X],
    y: data[offset + OFFSET_Y],
    z: data[offset + OFFSET_Z],
    vx: data[offset + OFFSET_VX],
    vy: data[offset + OFFSET_VY],
    vz: data[offset + OFFSET_VZ],
    mass: data[offset + OFFSET_MASS],
  };
}

/**
 * Set particle data by index
 */
export function setParticle(
  data: Float32Array | Float64Array,
  index: number,
  particle: {
    x: number;
    y: number;
    z: number;
    vx: number;
    vy: number;
    vz: number;
    mass?: number;
  }
) {
  const offset = index * FLOATS_PER_PARTICLE;
  data[offset + OFFSET_X] = particle.x;
  data[offset + OFFSET_Y] = particle.y;
  data[offset + OFFSET_Z] = particle.z;
  data[offset + OFFSET_VX] = particle.vx;
  data[offset + OFFSET_VY] = particle.vy;
  data[offset + OFFSET_VZ] = particle.vz;
  data[offset + OFFSET_MASS] = particle.mass ?? 1;
}

/**
 * Get position of particle by index
 */
export function getPosition(data: Float32Array | Float64Array, index: number) {
  const offset = index * FLOATS_PER_PARTICLE;
  return {
    x: data[offset + OFFSET_X],
    y: data[offset + OFFSET_Y],
    z: data[offset + OFFSET_Z],
  };
}

/**
 * Get velocity of particle by index
 */
export function getVelocity(data: Float32Array | Float64Array, index: number) {
  const offset = index * FLOATS_PER_PARTICLE;
  return {
    vx: data[offset + OFFSET_VX],
    vy: data[offset + OFFSET_VY],
    vz: data[offset + OFFSET_VZ],
  };
}

/**
 * Get mass of particle by index
 */
export function getMass(data: Float32Array | Float64Array, index: number): number {
  const offset = index * FLOATS_PER_PARTICLE;
  return data[offset + OFFSET_MASS];
}

/**
 * Update position of particle by index
 */
export function updatePosition(
  data: Float32Array | Float64Array,
  index: number,
  dx: number,
  dy: number,
  dz: number
) {
  const offset = index * FLOATS_PER_PARTICLE;
  data[offset + OFFSET_X] += dx;
  data[offset + OFFSET_Y] += dy;
  data[offset + OFFSET_Z] += dz;
}

/**
 * Update velocity of particle by index
 */
export function updateVelocity(
  data: Float32Array | Float64Array,
  index: number,
  dvx: number,
  dvy: number,
  dvz: number
) {
  const offset = index * FLOATS_PER_PARTICLE;
  data[offset + OFFSET_VX] += dvx;
  data[offset + OFFSET_VY] += dvy;
  data[offset + OFFSET_VZ] += dvz;
}

/**
 * Copy particle data from one array to another
 */
export function copyParticle(
  source: Float32Array | Float64Array,
  sourceIndex: number,
  dest: Float32Array | Float64Array,
  destIndex: number
) {
  const srcOffset = sourceIndex * FLOATS_PER_PARTICLE;
  const destOffset = destIndex * FLOATS_PER_PARTICLE;

  for (let i = 0; i < FLOATS_PER_PARTICLE; i++) {
    dest[destOffset + i] = source[srcOffset + i];
  }
}

/**
 * Get number of particles in array
 */
export function getParticleCount(data: Float32Array | Float64Array): number {
  return data.length / FLOATS_PER_PARTICLE;
}

/**
 * Create a deep copy of particle data
 */
export function cloneParticleData(data: Float32Array | Float64Array): Float32Array {
  return new Float32Array(data);
}

/**
 * Convert object-based particle array to TypedArray (for migration)
 */
export function fromParticleObjects(
  particles: Array<{
    x: number;
    y: number;
    z: number;
    vx: number;
    vy: number;
    vz: number;
    mass?: number;
  }>
): Float32Array {
  const data = createParticleArray(particles.length);

  // Optimization: Inline setParticle logic and use manual offset tracking
  // to avoid function call overhead and inner-loop multiplication.
  let offset = 0;
  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    data[offset + OFFSET_X] = p.x;
    data[offset + OFFSET_Y] = p.y;
    data[offset + OFFSET_Z] = p.z;
    data[offset + OFFSET_VX] = p.vx;
    data[offset + OFFSET_VY] = p.vy;
    data[offset + OFFSET_VZ] = p.vz;
    data[offset + OFFSET_MASS] = p.mass ?? 1;
    offset += FLOATS_PER_PARTICLE;
  }

  return data;
}

/**
 * Convert TypedArray to object-based particle array (for compatibility)
 */
export function toParticleObjects(data: Float32Array): Array<{
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  mass: number;
}> {
  const numParticles = getParticleCount(data);
  const particles = new Array(numParticles);

  // Optimization: Inline getParticle logic and use manual offset tracking
  // to avoid function call overhead and inner-loop multiplication.
  let offset = 0;
  for (let i = 0; i < numParticles; i++) {
    particles[i] = {
      x: data[offset + OFFSET_X],
      y: data[offset + OFFSET_Y],
      z: data[offset + OFFSET_Z],
      vx: data[offset + OFFSET_VX],
      vy: data[offset + OFFSET_VY],
      vz: data[offset + OFFSET_VZ],
      mass: data[offset + OFFSET_MASS],
    };
    offset += FLOATS_PER_PARTICLE;
  }

  return particles;
}

/**
 * Extract positions into a format suitable for THREE.js BufferAttribute
 */
export function extractPositions(data: Float32Array, out?: Float32Array): Float32Array {
  const numParticles = getParticleCount(data);
  const positions = out || new Float32Array(numParticles * 3);

  // Optimization: Use manual offset incrementing for input and output
  // to avoid calculating i * FLOATS_PER_PARTICLE and i * 3 inside the loop.
  let inOffset = 0;
  let outOffset = 0;
  for (let i = 0; i < numParticles; i++) {
    positions[outOffset] = data[inOffset + OFFSET_X];
    positions[outOffset + 1] = data[inOffset + OFFSET_Y];
    positions[outOffset + 2] = data[inOffset + OFFSET_Z];
    inOffset += FLOATS_PER_PARTICLE;
    outOffset += 3;
  }

  return positions;
}

/**
 * Extract velocities into a separate array
 */
export function extractVelocities(data: Float32Array): Float32Array {
  const numParticles = getParticleCount(data);
  const velocities = new Float32Array(numParticles * 3);

  // Optimization: Use manual offset incrementing for input and output
  let inOffset = 0;
  let outOffset = 0;
  for (let i = 0; i < numParticles; i++) {
    velocities[outOffset] = data[inOffset + OFFSET_VX];
    velocities[outOffset + 1] = data[inOffset + OFFSET_VY];
    velocities[outOffset + 2] = data[inOffset + OFFSET_VZ];
    inOffset += FLOATS_PER_PARTICLE;
    outOffset += 3;
  }

  return velocities;
}

/**
 * Calculate colors based on velocity magnitudes
 */
export function calculateColors(
  data: Float32Array,
  maxVelocity = 10,
  out?: Float32Array
): Float32Array {
  const numParticles = getParticleCount(data);
  const colors = out || new Float32Array(numParticles * 3);

  // Optimization: Pre-calculate the inverse of maxVelocity to replace division
  // with multiplication inside the loop, saving CPU cycles.
  const invMaxVelocity = 1.0 / maxVelocity;

  // Optimization: Calculate total floats to avoid multiplication in loop
  const numFloats = numParticles * FLOATS_PER_PARTICLE;

  let cIdx = 0;
  // Optimization: Iterate by offset directly rather than calculating per particle
  for (let offset = 0; offset < numFloats; offset += FLOATS_PER_PARTICLE) {
    const vx = data[offset + OFFSET_VX];
    const vy = data[offset + OFFSET_VY];
    const vz = data[offset + OFFSET_VZ];

    const v = Math.sqrt(vx * vx + vy * vy + vz * vz);
    const norm = Math.min(v * invMaxVelocity, 1);

    const halfNorm = norm * 0.5;

    // Optimization: Avoid repeated addition/multiplication by using pre-calculated halfNorm
    colors[cIdx] = 0.5 + halfNorm;     // R
    colors[cIdx + 1] = 0.5;            // G
    colors[cIdx + 2] = 1 - halfNorm;   // B

    cIdx += 3;
  }

  return colors;
}

/**
 * Calculate center of mass for all particles
 */
export function getCenterOfMass(data: Float32Array | Float64Array): {
  x: number;
  y: number;
  z: number;
} {
  const numParticles = getParticleCount(data);
  const numFloats = numParticles * FLOATS_PER_PARTICLE;

  let totalMass = 0;
  let comX = 0;
  let comY = 0;
  let comZ = 0;

  // Optimization: Iterate by offset directly rather than calculating per particle
  for (let offset = 0; offset < numFloats; offset += FLOATS_PER_PARTICLE) {
    const mass = data[offset + OFFSET_MASS];

    comX += data[offset + OFFSET_X] * mass;
    comY += data[offset + OFFSET_Y] * mass;
    comZ += data[offset + OFFSET_Z] * mass;
    totalMass += mass;
  }

  // Optimization: Pre-calculate the inverse mass
  const invTotalMass = 1.0 / totalMass;

  return {
    x: comX * invTotalMass,
    y: comY * invTotalMass,
    z: comZ * invTotalMass,
  };
}

/**
 * Calculate center of mass velocity for all particles
 */
export function getCenterOfMassVelocity(data: Float32Array | Float64Array): {
  vx: number;
  vy: number;
  vz: number;
} {
  const numParticles = getParticleCount(data);
  const numFloats = numParticles * FLOATS_PER_PARTICLE;

  let totalMass = 0;
  let comVx = 0;
  let comVy = 0;
  let comVz = 0;

  // Optimization: Iterate by offset directly rather than calculating per particle
  for (let offset = 0; offset < numFloats; offset += FLOATS_PER_PARTICLE) {
    const mass = data[offset + OFFSET_MASS];

    comVx += data[offset + OFFSET_VX] * mass;
    comVy += data[offset + OFFSET_VY] * mass;
    comVz += data[offset + OFFSET_VZ] * mass;
    totalMass += mass;
  }

  // Optimization: Pre-calculate the inverse mass
  const invTotalMass = 1.0 / totalMass;

  return {
    vx: comVx * invTotalMass,
    vy: comVy * invTotalMass,
    vz: comVz * invTotalMass,
  };
}

/**
 * Remove center of mass velocity (zero net momentum)
 * This ensures the system doesn't drift over time
 */
export function removeCenterOfMassVelocity(data: Float32Array | Float64Array): void {
  const comVel = getCenterOfMassVelocity(data);
  const numParticles = getParticleCount(data);
  const numFloats = numParticles * FLOATS_PER_PARTICLE;

  // Optimization: Iterate by offset directly rather than calculating per particle
  for (let offset = 0; offset < numFloats; offset += FLOATS_PER_PARTICLE) {
    data[offset + OFFSET_VX] -= comVel.vx;
    data[offset + OFFSET_VY] -= comVel.vy;
    data[offset + OFFSET_VZ] -= comVel.vz;
  }
}
