/**
 * Efficient TypedArray-based particle data storage
 *
 * Each particle is stored as 8 consecutive floats (32 bytes):
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
export function getParticle(data: Float32Array, index: number) {
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
  data: Float32Array,
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
export function getPosition(data: Float32Array, index: number) {
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
export function getVelocity(data: Float32Array, index: number) {
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
export function getMass(data: Float32Array, index: number): number {
  const offset = index * FLOATS_PER_PARTICLE;
  return data[offset + OFFSET_MASS];
}

/**
 * Update position of particle by index
 */
export function updatePosition(
  data: Float32Array,
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
  data: Float32Array,
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
  source: Float32Array,
  sourceIndex: number,
  dest: Float32Array,
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
export function getParticleCount(data: Float32Array): number {
  return data.length / FLOATS_PER_PARTICLE;
}

/**
 * Create a deep copy of particle data
 */
export function cloneParticleData(data: Float32Array): Float32Array {
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

  for (let i = 0; i < particles.length; i++) {
    setParticle(data, i, particles[i]);
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

  for (let i = 0; i < numParticles; i++) {
    particles[i] = getParticle(data, i);
  }

  return particles;
}

/**
 * Extract positions into a format suitable for THREE.js BufferAttribute
 */
export function extractPositions(data: Float32Array, out?: Float32Array): Float32Array {
  const numParticles = getParticleCount(data);
  const positions = out || new Float32Array(numParticles * 3);

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    positions[i * 3 + 0] = data[offset + OFFSET_X];
    positions[i * 3 + 1] = data[offset + OFFSET_Y];
    positions[i * 3 + 2] = data[offset + OFFSET_Z];
  }

  return positions;
}

/**
 * Extract velocities into a separate array
 */
export function extractVelocities(data: Float32Array): Float32Array {
  const numParticles = getParticleCount(data);
  const velocities = new Float32Array(numParticles * 3);

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    velocities[i * 3 + 0] = data[offset + OFFSET_VX];
    velocities[i * 3 + 1] = data[offset + OFFSET_VY];
    velocities[i * 3 + 2] = data[offset + OFFSET_VZ];
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

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    const vx = data[offset + OFFSET_VX];
    const vy = data[offset + OFFSET_VY];
    const vz = data[offset + OFFSET_VZ];

    const v = Math.sqrt(vx * vx + vy * vy + vz * vz);
    const norm = Math.min(v / maxVelocity, 1);

    colors[i * 3 + 0] = 0.5 + norm * 0.5; // R
    colors[i * 3 + 1] = 0.5; // G
    colors[i * 3 + 2] = 1 - norm * 0.5; // B
  }

  return colors;
}

/**
 * Calculate center of mass for all particles
 */
export function getCenterOfMass(data: Float32Array): {
  x: number;
  y: number;
  z: number;
} {
  const numParticles = getParticleCount(data);

  let totalMass = 0;
  let comX = 0;
  let comY = 0;
  let comZ = 0;

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    const mass = data[offset + OFFSET_MASS];

    comX += data[offset + OFFSET_X] * mass;
    comY += data[offset + OFFSET_Y] * mass;
    comZ += data[offset + OFFSET_Z] * mass;
    totalMass += mass;
  }

  return {
    x: comX / totalMass,
    y: comY / totalMass,
    z: comZ / totalMass,
  };
}

/**
 * Calculate center of mass velocity for all particles
 */
export function getCenterOfMassVelocity(data: Float32Array): {
  vx: number;
  vy: number;
  vz: number;
} {
  const numParticles = getParticleCount(data);

  let totalMass = 0;
  let comVx = 0;
  let comVy = 0;
  let comVz = 0;

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    const mass = data[offset + OFFSET_MASS];

    comVx += data[offset + OFFSET_VX] * mass;
    comVy += data[offset + OFFSET_VY] * mass;
    comVz += data[offset + OFFSET_VZ] * mass;
    totalMass += mass;
  }

  return {
    vx: comVx / totalMass,
    vy: comVy / totalMass,
    vz: comVz / totalMass,
  };
}

/**
 * Remove center of mass velocity (zero net momentum)
 * This ensures the system doesn't drift over time
 */
export function removeCenterOfMassVelocity(data: Float32Array): void {
  const comVel = getCenterOfMassVelocity(data);
  const numParticles = getParticleCount(data);

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    data[offset + OFFSET_VX] -= comVel.vx;
    data[offset + OFFSET_VY] -= comVel.vy;
    data[offset + OFFSET_VZ] -= comVel.vz;
  }
}
