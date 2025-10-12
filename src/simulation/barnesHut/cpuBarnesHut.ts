/**
 * CPU-based Barnes-Hut N-body simulation
 * Uses O(N log N) algorithm for force calculation
 */

import {
  getParticleCount,
  FLOATS_PER_PARTICLE,
  OFFSET_X,
  OFFSET_Y,
  OFFSET_Z,
  OFFSET_VX,
  OFFSET_VY,
  OFFSET_VZ,
  OFFSET_MASS,
} from '../particleData';
import { Octree } from './octree';

export interface BarnesHutOptions {
  theta?: number; // Opening angle (0.5 = good accuracy, 1.0 = fast)
  G?: number; // Gravitational constant
  softening?: number; // Softening parameter
}

/**
 * Compute forces on all particles using Barnes-Hut algorithm
 */
export function computeForcesBarnesHut(
  particles: Float32Array,
  forces: Float32Array,
  options: BarnesHutOptions = {}
): void {
  const theta = options.theta ?? 0.5;
  const G = options.G ?? 1.0;
  const softening = options.softening ?? 2.0;

  // Build octree
  const buildStart = performance.now();
  const octree = new Octree(particles);
  const buildTime = performance.now() - buildStart;

  // Compute force on each particle
  const forceStart = performance.now();
  const numParticles = getParticleCount(particles);
  for (let i = 0; i < numParticles; i++) {
    const force = octree.computeForce(i, theta, G, softening);

    forces[i * 3 + 0] = force.x;
    forces[i * 3 + 1] = force.y;
    forces[i * 3 + 2] = force.z;
  }
  const forceTime = performance.now() - forceStart;

  // Log timing on first call
  if (!(computeForcesBarnesHut as any)._hasLogged) {
    console.log(`Barnes-Hut profile: build=${buildTime.toFixed(1)}ms, forces=${forceTime.toFixed(1)}ms`);
    (computeForcesBarnesHut as any)._hasLogged = true;
  }
}

/**
 * Integrate particle motion using forces (Leapfrog integration)
 */
export function integrateLeapfrog(
  particles: Float32Array,
  forces: Float32Array,
  dt: number
): void {
  const numParticles = getParticleCount(particles);

  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    const forceOffset = i * 3;

    const mass = particles[offset + OFFSET_MASS];
    const ax = forces[forceOffset + 0] / mass;
    const ay = forces[forceOffset + 1] / mass;
    const az = forces[forceOffset + 2] / mass;

    // Half-step velocity update
    particles[offset + OFFSET_VX] += ax * dt * 0.5;
    particles[offset + OFFSET_VY] += ay * dt * 0.5;
    particles[offset + OFFSET_VZ] += az * dt * 0.5;

    // Full-step position update
    particles[offset + OFFSET_X] += particles[offset + OFFSET_VX] * dt;
    particles[offset + OFFSET_Y] += particles[offset + OFFSET_VY] * dt;
    particles[offset + OFFSET_Z] += particles[offset + OFFSET_VZ] * dt;
  }
}

/**
 * Single timestep using velocity Verlet integration
 * More efficient than Leapfrog for Barnes-Hut (only one octree build per step)
 */
export function stepBarnesHut(
  particles: Float32Array,
  dt: number,
  options: BarnesHutOptions = {}
): void {
  const numParticles = getParticleCount(particles);
  const forces = new Float32Array(numParticles * 3);

  // 1. Compute forces at current positions
  computeForcesBarnesHut(particles, forces, options);

  // 2. Update positions and velocities using velocity Verlet
  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    const forceOffset = i * 3;

    const mass = particles[offset + OFFSET_MASS];
    const ax = forces[forceOffset + 0] / mass;
    const ay = forces[forceOffset + 1] / mass;
    const az = forces[forceOffset + 2] / mass;

    // Update velocity (full step)
    particles[offset + OFFSET_VX] += ax * dt;
    particles[offset + OFFSET_VY] += ay * dt;
    particles[offset + OFFSET_VZ] += az * dt;

    // Update position (full step)
    particles[offset + OFFSET_X] += particles[offset + OFFSET_VX] * dt;
    particles[offset + OFFSET_Y] += particles[offset + OFFSET_VY] * dt;
    particles[offset + OFFSET_Z] += particles[offset + OFFSET_VZ] * dt;
  }
}
