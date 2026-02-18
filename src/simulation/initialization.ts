import {
  createParticleArray,
  setParticle,
  removeCenterOfMassVelocity,
} from './particleData';

// Constants used for initialization
export const G = 1.0;

/**
 * Initialize particles for N-Body simulation
 * Dynamic scaling based on particle count to ensure stable orbits.
 */
export function initializeNBodyParticles(numParticles: number): Float32Array {
  const particles = createParticleArray(numParticles);

  // Central star
  setParticle(particles, 0, {
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    mass: 5000,
  });

  // Scale parameters based on particle count
  // For small counts (<5000), use default scale.
  // For large counts, increase radius to avoid excessive density.
  // Using sqrt scaling keeps surface density roughly constant.
  const scale = Math.max(1, Math.sqrt(numParticles / 5000));

  const minRadius = 20 * scale;
  const maxRadius = 80 * scale;
  const zScale = 5 * scale;

  const centralMass = 5000;
  // Mass of each orbiting particle is 1.0 (hardcoded in original logic)
  const particleMass = 1.0;

  // Orbiting particles
  for (let i = 1; i < numParticles; i++) {
    const r = minRadius + Math.random() * (maxRadius - minRadius);
    const theta = Math.random() * Math.PI * 2;
    const z = (Math.random() - 0.5) * zScale;

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);

    // Calculate enclosed mass (approximate)
    // Assuming uniform distribution in r between minRadius and maxRadius
    // M_enclosed = M_central + M_cloud * (fraction of cloud inside r)
    // Fraction inside r = (r - minRadius) / (maxRadius - minRadius)
    // Total cloud mass = (numParticles - 1) * particleMass
    const cloudMassInside = (numParticles - 1) * particleMass * ((r - minRadius) / (maxRadius - minRadius));
    const M_enclosed = centralMass + cloudMassInside;

    // Circular orbit velocity: v = sqrt(GM/r)
    // We use M_enclosed to account for the mass of the cloud itself
    const v = Math.sqrt(G * M_enclosed / r);

    // Tangential velocity with some random noise
    const vx = -v * Math.sin(theta) + (Math.random() - 0.5) * 0.5;
    const vy = v * Math.cos(theta) + (Math.random() - 0.5) * 0.5;
    const vz = (Math.random() - 0.5) * 0.2;

    setParticle(particles, i, { x, y, z, vx, vy, vz, mass: particleMass });
  }

  // Remove net momentum to prevent drift
  removeCenterOfMassVelocity(particles);

  return particles;
}
