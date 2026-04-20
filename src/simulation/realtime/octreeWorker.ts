/**
 * Web Worker that builds the Barnes-Hut octree off the main thread and
 * returns a GPU-ready buffer.
 *
 * Uses the flat-array builder which writes nodes directly in GPU format —
 * no intermediate object tree, no separate serialization pass.
 */

import { buildFlatOctree, BYTES_PER_NODE } from '../barnesHut/flatOctree';

let nodeScratch: ArrayBuffer | null = null;
let indexScratch: Int32Array | null = null;
let scratchMaxNodes = 0;

function ensureScratch(maxNodes: number, maxParticles: number) {
  if (!nodeScratch || scratchMaxNodes < maxNodes) {
    nodeScratch = new ArrayBuffer(maxNodes * BYTES_PER_NODE);
    scratchMaxNodes = maxNodes;
  }
  if (!indexScratch || indexScratch.length < maxParticles) {
    indexScratch = new Int32Array(maxParticles);
  }
}

self.onmessage = (e: MessageEvent) => {
  const { particleData, maxNodes } = e.data as {
    particleData: ArrayBuffer;
    maxNodes: number;
  };

  const particles = new Float32Array(particleData);
  const numParticles = particles.length / 8;
  ensureScratch(maxNodes, numParticles);

  const { buffer, nodeCount } = buildFlatOctree(particles, {
    nodeScratch: nodeScratch!,
    indexScratch: indexScratch!,
    maxNodes,
  });

  self.postMessage(
    { buffer, nodeCount, particleData },
    { transfer: [buffer, particleData] } as StructuredSerializeOptions
  );
};
