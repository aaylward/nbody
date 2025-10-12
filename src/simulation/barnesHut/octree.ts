/**
 * Octree data structure for Barnes-Hut N-body algorithm
 *
 * This octree subdivides 3D space into octants recursively to enable
 * O(N log N) force calculations instead of O(N²) brute force.
 */

import {
  getParticleCount,
  OFFSET_X,
  OFFSET_Y,
  OFFSET_Z,
  OFFSET_MASS,
  FLOATS_PER_PARTICLE,
} from '../particleData';

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Bounds {
  min: Vec3;
  max: Vec3;
}

export interface OctreeNode {
  // Spatial bounds
  bounds: Bounds;

  // Aggregate data (for force approximation)
  centerOfMass: Vec3;
  totalMass: number;
  particleCount: number;

  // Tree structure
  isLeaf: boolean;
  particleIndices: number[]; // If leaf, which particles are in this node
  children: OctreeNode[] | null; // If internal, 8 octant children
}

export class Octree {
  private root: OctreeNode;
  private particles: Float32Array;
  private readonly maxParticlesPerNode: number;
  private readonly maxDepth: number;

  constructor(particles: Float32Array, maxParticlesPerNode = 1, maxDepth = 30) {
    this.particles = particles;
    this.maxParticlesPerNode = maxParticlesPerNode;
    this.maxDepth = maxDepth;

    // Compute bounding box for all particles
    const bounds = this.computeBounds();

    // Build the octree
    const numParticles = getParticleCount(particles);
    const allIndices = Array.from({ length: numParticles }, (_, i) => i);
    this.root = this.buildNode(bounds, allIndices, 0);
  }

  /**
   * Compute bounding box that contains all particles
   */
  private computeBounds(): Bounds {
    const numParticles = getParticleCount(this.particles);
    if (numParticles === 0) {
      return {
        min: { x: -1, y: -1, z: -1 },
        max: { x: 1, y: 1, z: 1 },
      };
    }

    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (let i = 0; i < numParticles; i++) {
      const offset = i * FLOATS_PER_PARTICLE;
      const x = this.particles[offset + OFFSET_X];
      const y = this.particles[offset + OFFSET_Y];
      const z = this.particles[offset + OFFSET_Z];

      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      maxZ = Math.max(maxZ, z);
    }

    // Add small padding to avoid particles exactly on boundaries
    const padding = 0.01;
    return {
      min: { x: minX - padding, y: minY - padding, z: minZ - padding },
      max: { x: maxX + padding, y: maxY + padding, z: maxZ + padding },
    };
  }

  /**
   * Recursively build octree node
   */
  private buildNode(bounds: Bounds, particleIndices: number[], depth: number): OctreeNode {
    const node: OctreeNode = {
      bounds,
      centerOfMass: { x: 0, y: 0, z: 0 },
      totalMass: 0,
      particleCount: particleIndices.length,
      isLeaf: false,
      particleIndices: [],
      children: null,
    };

    // Base case: leaf node (either max particles reached or max depth reached)
    if (particleIndices.length <= this.maxParticlesPerNode || depth >= this.maxDepth) {
      node.isLeaf = true;
      node.particleIndices = particleIndices;
      this.computeNodeMass(node);
      return node;
    }

    // Recursive case: subdivide into 8 octants
    const octants = this.subdivide(bounds);
    const octantIndices: number[][] = Array.from({ length: 8 }, () => []);

    // Assign particles to octants (optimized to avoid getParticle calls)
    const midX = (bounds.min.x + bounds.max.x) / 2;
    const midY = (bounds.min.y + bounds.max.y) / 2;
    const midZ = (bounds.min.z + bounds.max.z) / 2;

    for (const idx of particleIndices) {
      const offset = idx * FLOATS_PER_PARTICLE;
      const px = this.particles[offset + OFFSET_X];
      const py = this.particles[offset + OFFSET_Y];
      const pz = this.particles[offset + OFFSET_Z];

      let octantIdx = 0;
      if (px >= midX) octantIdx |= 1;
      if (py >= midY) octantIdx |= 2;
      if (pz >= midZ) octantIdx |= 4;

      octantIndices[octantIdx].push(idx);
    }

    // Recursively build children (only for non-empty octants)
    node.children = [];
    for (let i = 0; i < 8; i++) {
      if (octantIndices[i].length > 0) {
        const child = this.buildNode(octants[i], octantIndices[i], depth + 1);
        node.children.push(child);
      }
    }

    // Compute aggregate center of mass from children
    this.computeNodeMass(node);

    return node;
  }

  /**
   * Subdivide bounds into 8 octants
   */
  private subdivide(bounds: Bounds): Bounds[] {
    const { min, max } = bounds;
    const mid = {
      x: (min.x + max.x) / 2,
      y: (min.y + max.y) / 2,
      z: (min.z + max.z) / 2,
    };

    return [
      // Bottom 4 octants (z < mid.z)
      { min: { x: min.x, y: min.y, z: min.z }, max: { x: mid.x, y: mid.y, z: mid.z } }, // 0: ---
      { min: { x: mid.x, y: min.y, z: min.z }, max: { x: max.x, y: mid.y, z: mid.z } }, // 1: +--
      { min: { x: min.x, y: mid.y, z: min.z }, max: { x: mid.x, y: max.y, z: mid.z } }, // 2: -+-
      { min: { x: mid.x, y: mid.y, z: min.z }, max: { x: max.x, y: max.y, z: mid.z } }, // 3: ++-
      // Top 4 octants (z >= mid.z)
      { min: { x: min.x, y: min.y, z: mid.z }, max: { x: mid.x, y: mid.y, z: max.z } }, // 4: --+
      { min: { x: mid.x, y: min.y, z: mid.z }, max: { x: max.x, y: mid.y, z: max.z } }, // 5: +-+
      { min: { x: min.x, y: mid.y, z: mid.z }, max: { x: mid.x, y: max.y, z: max.z } }, // 6: -++
      { min: { x: mid.x, y: mid.y, z: mid.z }, max: { x: max.x, y: max.y, z: max.z } }, // 7: +++
    ];
  }


  /**
   * Compute center of mass and total mass for a node
   */
  private computeNodeMass(node: OctreeNode): void {
    if (node.isLeaf) {
      // Leaf: compute from particles
      let totalMass = 0;
      let comX = 0;
      let comY = 0;
      let comZ = 0;

      for (const idx of node.particleIndices) {
        const offset = idx * FLOATS_PER_PARTICLE;
        const mass = this.particles[offset + OFFSET_MASS];
        const x = this.particles[offset + OFFSET_X];
        const y = this.particles[offset + OFFSET_Y];
        const z = this.particles[offset + OFFSET_Z];

        totalMass += mass;
        comX += x * mass;
        comY += y * mass;
        comZ += z * mass;
      }

      node.totalMass = totalMass;
      if (totalMass > 0) {
        node.centerOfMass = {
          x: comX / totalMass,
          y: comY / totalMass,
          z: comZ / totalMass,
        };
      }
    } else {
      // Internal: compute from children
      let totalMass = 0;
      let comX = 0;
      let comY = 0;
      let comZ = 0;

      for (const child of node.children!) {
        if (child.totalMass > 0) {
          totalMass += child.totalMass;
          comX += child.centerOfMass.x * child.totalMass;
          comY += child.centerOfMass.y * child.totalMass;
          comZ += child.centerOfMass.z * child.totalMass;
        }
      }

      node.totalMass = totalMass;
      if (totalMass > 0) {
        node.centerOfMass = {
          x: comX / totalMass,
          y: comY / totalMass,
          z: comZ / totalMass,
        };
      }
    }
  }

  /**
   * Compute gravitational force on a particle using Barnes-Hut approximation
   * @param particleIndex Index of particle to compute force on
   * @param theta Opening angle parameter (0.5 = good accuracy, 1.0 = fast)
   * @param G Gravitational constant
   * @param softening Softening parameter to avoid singularities
   */
  computeForce(
    particleIndex: number,
    theta: number,
    G = 1.0,
    softening = 2.0
  ): Vec3 {
    const offset = particleIndex * FLOATS_PER_PARTICLE;
    const px = this.particles[offset + OFFSET_X];
    const py = this.particles[offset + OFFSET_Y];
    const pz = this.particles[offset + OFFSET_Z];
    const pmass = this.particles[offset + OFFSET_MASS];

    return this.computeForceRecursive(
      { x: px, y: py, z: pz, mass: pmass },
      particleIndex,
      this.root,
      theta,
      G,
      softening
    );
  }

  /**
   * Recursively traverse octree to compute force
   */
  private computeForceRecursive(
    particle: { x: number; y: number; z: number; mass: number },
    particleIndex: number,
    node: OctreeNode,
    theta: number,
    G: number,
    softening: number
  ): Vec3 {
    // Empty node contributes no force
    if (node.totalMass === 0 || node.particleCount === 0) {
      return { x: 0, y: 0, z: 0 };
    }

    const dx = node.centerOfMass.x - particle.x;
    const dy = node.centerOfMass.y - particle.y;
    const dz = node.centerOfMass.z - particle.z;
    const r2 = dx * dx + dy * dy + dz * dz;
    const r = Math.sqrt(r2);

    // Get cell width
    const cellWidth = Math.max(
      node.bounds.max.x - node.bounds.min.x,
      node.bounds.max.y - node.bounds.min.y,
      node.bounds.max.z - node.bounds.min.z
    );

    // Barnes-Hut criterion: use approximation if cell is far enough
    const useApproximation = node.isLeaf || cellWidth / r < theta;

    if (useApproximation) {
      // Leaf node or far enough: compute force as if node is a single particle
      // Skip self-interaction for leaf nodes
      if (node.isLeaf && node.particleIndices.includes(particleIndex)) {
        // For leaf nodes, compute exact forces to all OTHER particles
        let fx = 0,
          fy = 0,
          fz = 0;
        for (const idx of node.particleIndices) {
          if (idx === particleIndex) continue; // Skip self

          const offset = idx * FLOATS_PER_PARTICLE;
          const ox = this.particles[offset + OFFSET_X];
          const oy = this.particles[offset + OFFSET_Y];
          const oz = this.particles[offset + OFFSET_Z];
          const omass = this.particles[offset + OFFSET_MASS];

          const dx = ox - particle.x;
          const dy = oy - particle.y;
          const dz = oz - particle.z;
          const r2 = dx * dx + dy * dy + dz * dz + softening * softening;
          const r = Math.sqrt(r2);
          const invR3 = 1 / (r * r2);
          const f = G * particle.mass * omass * invR3;

          fx += f * dx;
          fy += f * dy;
          fz += f * dz;
        }
        return { x: fx, y: fy, z: fz };
      }

      // Use center of mass approximation
      const r2Soft = r2 + softening * softening;
      const rSoft = Math.sqrt(r2Soft);
      const invR3 = 1 / (rSoft * r2Soft);
      const f = G * particle.mass * node.totalMass * invR3;

      return {
        x: f * dx,
        y: f * dy,
        z: f * dz,
      };
    } else {
      // Too close: recurse into children
      let fx = 0,
        fy = 0,
        fz = 0;

      // Only recurse into non-empty children
      if (node.children) {
        for (const child of node.children) {
          if (child.particleCount > 0) {
            const force = this.computeForceRecursive(
              particle,
              particleIndex,
              child,
              theta,
              G,
              softening
            );
            fx += force.x;
            fy += force.y;
            fz += force.z;
          }
        }
      }

      return { x: fx, y: fy, z: fz };
    }
  }

  /**
   * Get the root node (useful for testing)
   */
  getRoot(): OctreeNode {
    return this.root;
  }

  /**
   * Count total nodes in tree (useful for debugging)
   */
  countNodes(): number {
    return this.countNodesRecursive(this.root);
  }

  private countNodesRecursive(node: OctreeNode): number {
    if (node.isLeaf) return 1;
    let count = 1;
    for (const child of node.children!) {
      count += this.countNodesRecursive(child);
    }
    return count;
  }

  /**
   * Get maximum depth of tree (useful for debugging)
   */
  getMaxDepth(): number {
    return this.getMaxDepthRecursive(this.root, 0);
  }

  private getMaxDepthRecursive(node: OctreeNode, depth: number): number {
    if (node.isLeaf) return depth;
    let maxDepth = depth;
    for (const child of node.children!) {
      maxDepth = Math.max(maxDepth, this.getMaxDepthRecursive(child, depth + 1));
    }
    return maxDepth;
  }
}
