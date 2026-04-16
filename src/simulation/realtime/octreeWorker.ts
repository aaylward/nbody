/**
 * Web Worker that builds and serializes the Barnes-Hut octree off the
 * main thread. Receives particle positions, returns a GPU-ready buffer.
 */

import { Octree, OctreeNode } from '../barnesHut/octree';

const BYTES_PER_NODE = 32; // 5 floats + 3 u32s = 8 × 4 bytes

// Reusable scratch — allocated on first message, reused across rebuilds.
let scratchBuffer: ArrayBuffer | null = null;
let floatView: Float32Array;
let intView: Uint32Array;
let bfsQueue: (OctreeNode | null)[];
let scratchMaxNodes = 0;

function ensureScratch(maxNodes: number) {
  if (maxNodes <= scratchMaxNodes) return;
  scratchMaxNodes = maxNodes;
  scratchBuffer = new ArrayBuffer(maxNodes * BYTES_PER_NODE);
  floatView = new Float32Array(scratchBuffer);
  intView = new Uint32Array(scratchBuffer);
  bfsQueue = new Array(maxNodes).fill(null);
}

function serializeOctree(octree: Octree, maxNodes: number): { usedBytes: number; nodeCount: number } {
  ensureScratch(maxNodes);

  const queue = bfsQueue;
  queue[0] = octree.getRoot();
  let head = 0;
  let tail = 1;
  let nextIndex = 1;

  while (head < tail) {
    const node = queue[head] as OctreeNode;
    queue[head] = null;
    const wordOffset = head * 8;
    head++;

    const { min, max } = node.bounds;
    const cellWidth = Math.max(max.x - min.x, max.y - min.y, max.z - min.z);

    floatView[wordOffset + 0] = node.centerOfMass.x;
    floatView[wordOffset + 1] = node.centerOfMass.y;
    floatView[wordOffset + 2] = node.centerOfMass.z;
    floatView[wordOffset + 3] = node.totalMass;
    floatView[wordOffset + 4] = cellWidth;

    const children = node.children;
    const childCount = children ? children.length : 0;

    intView[wordOffset + 5] = childCount > 0 ? nextIndex : 0;
    intView[wordOffset + 6] = childCount;
    intView[wordOffset + 7] = node.particleCount;

    if (childCount > 0) {
      if (tail + childCount > scratchMaxNodes) {
        throw new Error(`Octree exceeded scratch capacity (${scratchMaxNodes})`);
      }
      for (let i = 0; i < childCount; i++) {
        queue[tail++] = children![i];
        nextIndex++;
      }
    }
  }

  const nodeCount = tail;
  return { usedBytes: nodeCount * BYTES_PER_NODE, nodeCount };
}

self.onmessage = (e: MessageEvent) => {
  const { particleData, maxNodes } = e.data as {
    particleData: ArrayBuffer;
    maxNodes: number;
  };

  const particles = new Float32Array(particleData);
  const octree = new Octree(particles);
  const { usedBytes, nodeCount } = serializeOctree(octree, maxNodes);

  // Copy the used portion into a transferable buffer so the main thread
  // can upload it directly to the GPU without another copy.
  const result = new ArrayBuffer(usedBytes);
  new Uint8Array(result).set(new Uint8Array(scratchBuffer!, 0, usedBytes));

  self.postMessage({ buffer: result, nodeCount }, { transfer: [result] } as StructuredSerializeOptions);
};
