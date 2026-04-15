/**
 * Real-time N-body simulation using GPU Barnes-Hut algorithm
 * Phase 3: Hybrid approach - CPU builds octree, GPU traverses for forces
 */

import {
  createParticleArray,
  setParticle,
  removeCenterOfMassVelocity,
} from '../particleData';
import { Octree, OctreeNode } from '../barnesHut/octree';
import { packParticlesForGPU, packVelocitiesForGPU, unpackParticlesFromGPU } from './barnesHutPacking';
import { PerformanceMonitor } from './performanceMonitor';

export interface RealtimeSimulationGPUBarnesHutOptions {
  device: GPUDevice;
  numParticles: number;
  deltaT?: number;
  targetPhysicsFPS?: number;
  theta?: number;
  octreeRebuildInterval?: number;
}

// Octree node format for GPU (flatten for buffer)
// Each node: [centerX, centerY, centerZ, totalMass, cellWidth, childStart, childCount, particleCount]
// NOTE: childStart, childCount, particleCount must be stored as integers, not floats!
const FLOATS_PER_NODE = 5; // 5 floats: centerX, centerY, centerZ, totalMass, cellWidth
const INTS_PER_NODE = 3;   // 3 ints: childStart, childCount, particleCount
const BYTES_PER_NODE = FLOATS_PER_NODE * 4 + INTS_PER_NODE * 4; // 32 bytes total

export class RealtimeNBodySimulationGPUBarnesHut {
  private device: GPUDevice;
  private numParticles: number;
  private deltaT: number;
  private running = false;
  private physicsFrameCount = 0;
  public lastPhysicsTime = 0; // Used by getPhysicsProgress in future

  // GPU resources
  private particleBuffer: GPUBuffer;
  private renderPositionBuffer: GPUBuffer;
  private velocityBuffer: GPUBuffer;
  private forcesBuffer: GPUBuffer;
  private octreeBuffer: GPUBuffer;
  // Separate uniform buffers per pipeline — the two shaders declare
  // different Uniforms struct layouts at binding 3, so they cannot share
  // a single buffer without one pass misreading the other's fields.
  private forcesUniformsBuffer: GPUBuffer;
  private integrateUniformsBuffer: GPUBuffer;
  private stagingBuffer: GPUBuffer; // Reuse staging buffer for downloads
  private forcesPipeline: GPUComputePipeline;
  private integratePipeline: GPUComputePipeline;
  private forcesBindGroup: GPUBindGroup | null = null;
  private integrateBindGroup: GPUBindGroup | null = null;

  // CPU octree
  private particlesCPU: Float32Array;
  private theta: number;
  private octreeRebuildInterval: number;
  private rebuildPhase: 'idle' | 'downloading' = 'idle';
  private downloadPromise: Promise<void> | null = null;
  private framesSinceRebuild = 0;
  public targetPhysicsFPS: number;
  public monitor: PerformanceMonitor;

  // Reusable scratch storage for octree serialization (avoids per-frame allocation).
  // The backing ArrayBuffer is sized once to the worst-case node count and then
  // overwritten in place every physics tick.
  private octreeScratch: ArrayBuffer;
  private octreeFloatView: Float32Array;
  private octreeIntView: Uint32Array;
  private bfsNodeQueue: (OctreeNode | null)[];

  constructor(options: RealtimeSimulationGPUBarnesHutOptions) {
    this.device = options.device;
    this.numParticles = options.numParticles;
    this.deltaT = options.deltaT ?? 0.01;
    this.targetPhysicsFPS = options.targetPhysicsFPS ?? 20;
    this.theta = options.theta ?? 0.8;
    this.octreeRebuildInterval = options.octreeRebuildInterval ?? 4;

    // Initialize CPU particle array for octree building
    this.particlesCPU = createParticleArray(this.numParticles);
    this.initializeParticles();

    // Initialize performance monitor
    this.monitor = new PerformanceMonitor();

    // Create GPU buffers.
    // NOTE: velocity and forces are declared as array<vec3f> in the shaders.
    // In WGSL, vec3<f32> has size 12 but alignment 16, so an array element
    // stride is 16 bytes (the trailing 4 bytes are padding). The buffer
    // allocation and the CPU-side packing must both honor the 16-byte stride,
    // otherwise the shader reads misaligned garbage and runs out of bounds.
    const particleBufferSize = this.numParticles * 4 * 4; // vec3f pos + f32 mass = 16 bytes
    const velocityBufferSize = this.numParticles * 4 * 4; // vec3f stride = 16 bytes
    const forcesBufferSize = this.numParticles * 4 * 4;   // vec3f stride = 16 bytes
    const maxOctreeNodes = this.numParticles * 8; // Worst case: many internal nodes
    const octreeBufferSize = maxOctreeNodes * BYTES_PER_NODE;

    this.particleBuffer = this.device.createBuffer({
      size: particleBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.renderPositionBuffer = this.device.createBuffer({
      size: particleBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.velocityBuffer = this.device.createBuffer({
      size: velocityBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.forcesBuffer = this.device.createBuffer({
      size: forcesBufferSize,
      usage: GPUBufferUsage.STORAGE,
    });

    this.octreeBuffer = this.device.createBuffer({
      size: octreeBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Allocate reusable host-side scratch that matches the GPU buffer. Each node
    // occupies BYTES_PER_NODE bytes, laid out as 5 floats + 3 u32s in the same
    // 32-bit slots — the Float32 and Uint32 views alias the same memory.
    this.octreeScratch = new ArrayBuffer(octreeBufferSize);
    this.octreeFloatView = new Float32Array(this.octreeScratch);
    this.octreeIntView = new Uint32Array(this.octreeScratch);
    this.bfsNodeQueue = new Array(maxOctreeNodes).fill(null);

    this.forcesUniformsBuffer = this.device.createBuffer({
      size: 16, // numParticles (u32), theta, G, softening
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.integrateUniformsBuffer = this.device.createBuffer({
      size: 16, // numParticles (u32), deltaT, 8 bytes padding
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.stagingBuffer = this.device.createBuffer({
      size: particleBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Upload initial data
    this.uploadParticlesToGPU();
    this.uploadVelocitiesToGPU();

    // The integrate pipeline's uniforms never change at runtime in the
    // current code path — write them once here and leave the buffer alone.
    const integrateUniformsBuf = new ArrayBuffer(16);
    new Uint32Array(integrateUniformsBuf, 0, 1)[0] = this.numParticles;
    new Float32Array(integrateUniformsBuf)[1] = this.deltaT;
    this.device.queue.writeBuffer(this.integrateUniformsBuffer, 0, integrateUniformsBuf);

    // Create compute pipelines
    this.forcesPipeline = this.createForcesPipeline();
    this.integratePipeline = this.createIntegratePipeline();
  }

  private initializeParticles(): void {
    // Central massive object
    setParticle(this.particlesCPU, 0, {
      x: 0, y: 0, z: 0,
      vx: 0, vy: 0, vz: 0,
      mass: 5000,
    });

    // Orbiting particles
    for (let i = 1; i < this.numParticles; i++) {
      const r = 20 + Math.random() * 60;
      const theta = Math.random() * Math.PI * 2;
      const z = (Math.random() - 0.5) * 5;

      const x = r * Math.cos(theta);
      const y = r * Math.sin(theta);

      const v = Math.sqrt(5000 / r);
      const vx = -v * Math.sin(theta) + (Math.random() - 0.5) * 0.5;
      const vy = v * Math.cos(theta) + (Math.random() - 0.5) * 0.5;
      const vz = (Math.random() - 0.5) * 0.2;

      setParticle(this.particlesCPU, i, { x, y, z, vx, vy, vz, mass: 1 });
    }

    removeCenterOfMassVelocity(this.particlesCPU);
  }

  private uploadParticlesToGPU(): void {
    const gpuData = packParticlesForGPU(this.particlesCPU, this.numParticles);
    this.device.queue.writeBuffer(this.particleBuffer, 0, gpuData);
  }

  private uploadVelocitiesToGPU(): void {
    const velocities = packVelocitiesForGPU(this.particlesCPU, this.numParticles);
    this.device.queue.writeBuffer(this.velocityBuffer, 0, velocities);
  }

  private serializeOctree(
    octree: Octree
  ): { view: Uint8Array; nodeCount: number } {
    // Flatten octree into breadth-first order and write straight into the
    // reusable scratch buffer. The queue is a preallocated array with head/tail
    // pointers — no Array.shift() (which is O(n) per call), no per-node
    // allocations, and no second copy pass.
    const floats = this.octreeFloatView;
    const ints = this.octreeIntView;
    const queue = this.bfsNodeQueue;
    const capacity = queue.length;

    queue[0] = octree.getRoot();
    let head = 0;
    let tail = 1;
    let nextIndex = 1;

    while (head < tail) {
      const node = queue[head] as OctreeNode;
      queue[head] = null; // release reference so the tree can be GC'd after this frame
      const index = head;
      head++;

      // Each record is 32 bytes = 8 × 32-bit words. Float and int views alias
      // the same buffer, so `wordOffset + 0..4` are floats and `+5..7` are u32s.
      const wordOffset = index * 8;

      const { min, max } = node.bounds;
      const cellWidth = Math.max(max.x - min.x, max.y - min.y, max.z - min.z);

      floats[wordOffset + 0] = node.centerOfMass.x;
      floats[wordOffset + 1] = node.centerOfMass.y;
      floats[wordOffset + 2] = node.centerOfMass.z;
      floats[wordOffset + 3] = node.totalMass;
      floats[wordOffset + 4] = cellWidth;

      const children = node.children;
      const childCount = children ? children.length : 0;

      ints[wordOffset + 5] = childCount > 0 ? nextIndex : 0; // childStart
      ints[wordOffset + 6] = childCount;                     // childCount
      ints[wordOffset + 7] = node.particleCount;             // particleCount

      if (childCount > 0) {
        // Guard against the (should-never-happen) case of blowing past the
        // worst-case node count used to size GPU + scratch buffers.
        if (tail + childCount > capacity) {
          throw new Error(
            `Octree node count exceeded scratch capacity (${capacity}); ` +
              `increase maxOctreeNodes.`
          );
        }
        for (let i = 0; i < childCount; i++) {
          queue[tail++] = children![i];
          nextIndex++;
        }
      }
    }

    // Return a zero-copy view over just the written bytes. The caller feeds
    // this directly to GPUQueue.writeBuffer, which copies it into the GPU
    // buffer without any intermediate host-side allocation.
    const nodeCount = tail;
    const usedBytes = nodeCount * BYTES_PER_NODE;
    return {
      view: new Uint8Array(this.octreeScratch, 0, usedBytes),
      nodeCount,
    };
  }

  private createForcesPipeline(): GPUComputePipeline {
    const shaderModule = this.device.createShaderModule({
      label: 'Barnes-Hut Forces Shader',
      code: `
        struct Particle {
          pos: vec3f,
          mass: f32,
        }

        struct OctreeNode {
          centerOfMass: vec3f,
          totalMass: f32,
          cellWidth: f32,
          childStart: u32,
          childCount: u32,
          particleCount: u32,
        }

        struct Uniforms {
          numParticles: u32,
          theta: f32,
          G: f32,
          softening: f32,
        }

        @group(0) @binding(0) var<storage, read> particles: array<Particle>;
        @group(0) @binding(1) var<storage, read> octree: array<OctreeNode>;
        @group(0) @binding(2) var<storage, read_write> forces: array<vec3f>;
        @group(0) @binding(3) var<uniform> uniforms: Uniforms;

        // Stack for iterative tree traversal
        const MAX_STACK_SIZE = 64u;
        const MAX_ITERATIONS = 10000u; // Prevent infinite loops

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3u) {
          let particleIdx = id.x;
          if (particleIdx >= uniforms.numParticles) { return; }

          let p = particles[particleIdx];

          // Iterative tree traversal using stack (no recursion!)
          var stack: array<u32, MAX_STACK_SIZE>;
          var stackPtr = 0u;
          stack[0] = 0u; // Start with root node
          stackPtr = 1u;

          var totalForce = vec3f(0.0);
          var iterations = 0u;

          // Get octree size for bounds checking
          let octreeSize = arrayLength(&octree);

          // Iteratively process nodes from stack
          while (stackPtr > 0u && iterations < MAX_ITERATIONS) {
            iterations += 1u;
            stackPtr -= 1u;
            let nodeIdx = stack[stackPtr];

            // Bounds check: ensure nodeIdx is valid
            if (nodeIdx >= octreeSize) {
              continue; // Skip invalid node index
            }

            let node = octree[nodeIdx];

            // Skip empty nodes
            if (node.particleCount == 0u) {
              continue;
            }

            let r = node.centerOfMass - p.pos;
            let dist = length(r);

            // Avoid division by zero
            if (dist < 0.0001) {
              continue;
            }

            // Barnes-Hut criterion: use approximation if far enough
            let useApproximation = (node.childCount == 0u) || (node.cellWidth / dist < uniforms.theta);

            if (useApproximation) {
              // Use center of mass approximation
              let r2 = dot(r, r) + uniforms.softening * uniforms.softening;
              let rSoft = sqrt(r2);
              let invR3 = 1.0 / (rSoft * r2);
              let f = uniforms.G * p.mass * node.totalMass * invR3;
              totalForce += f * r;
            } else {
              // Too close: push children onto stack (with bounds and overflow checks)
              for (var childOffset = 0u; childOffset < node.childCount; childOffset++) {
                let childIdx = node.childStart + childOffset;

                // Bounds check: ensure child index is valid
                if (childIdx >= octreeSize) {
                  continue; // Skip invalid child
                }

                // Stack overflow check: only push if space available
                if (stackPtr < MAX_STACK_SIZE) {
                  stack[stackPtr] = childIdx;
                  stackPtr += 1u;
                } else {
                  // Stack overflow: treat remaining children as approximations
                  let childNode = octree[childIdx];
                  if (childNode.particleCount > 0u) {
                    let r_child = childNode.centerOfMass - p.pos;
                    let dist_child = length(r_child);
                    if (dist_child > 0.0001) {
                      let r2 = dot(r_child, r_child) + uniforms.softening * uniforms.softening;
                      let rSoft = sqrt(r2);
                      let invR3 = 1.0 / (rSoft * r2);
                      let f = uniforms.G * p.mass * childNode.totalMass * invR3;
                      totalForce += f * r_child;
                    }
                  }
                }
              }
            }
          }

          forces[particleIdx] = totalForce;
        }
      `,
    });

    return this.device.createComputePipeline({
      label: 'Barnes-Hut Forces Pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  private createIntegratePipeline(): GPUComputePipeline {
    const shaderModule = this.device.createShaderModule({
      label: 'Integration Shader',
      code: `
        struct Particle {
          pos: vec3f,
          mass: f32,
        }

        struct Uniforms {
          numParticles: u32,
          deltaT: f32,
        }

        @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
        @group(0) @binding(1) var<storage, read> forces: array<vec3f>;
        @group(0) @binding(2) var<storage, read_write> velocities: array<vec3f>;
        @group(0) @binding(3) var<uniform> uniforms: Uniforms;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) id: vec3u) {
          let i = id.x;
          if (i >= uniforms.numParticles) { return; }

          let p = particles[i];
          let f = forces[i];
          let a = f / p.mass;

          // Update velocity
          velocities[i] += a * uniforms.deltaT;

          // Update position
          particles[i].pos += velocities[i] * uniforms.deltaT;
        }
      `,
    });

    return this.device.createComputePipeline({
      label: 'Integration Pipeline',
      layout: 'auto',
      compute: {
        module: shaderModule,
        entryPoint: 'main',
      },
    });
  }

  async start(): Promise<void> {
    console.log(`Starting GPU Barnes-Hut simulation with ${this.numParticles} particles...`);
    this.running = true;
    this.lastPhysicsTime = performance.now();
    this.physicsLoop();
  }

  stop(): void {
    this.running = false;
  }

  private async physicsLoop(): Promise<void> {
    let frameCount = 0;

    // Build the initial octree from CPU-side particle data (already matches
    // GPU state from the constructor) so the first frame has valid forces.
    this.uploadOctree(new Octree(this.particlesCPU));

    while (this.running) {
      await new Promise((resolve) => setTimeout(resolve, 0));

      const startTime = performance.now();

      // --- Fast path: always runs first so the render buffer updates ---
      // --- promptly even if a rebuild blocks the thread afterward.    ---

      // Update forces uniforms (theta may change at runtime via setTheta).
      // numParticles must be written as u32 (not f32) because the shader
      // declares it as u32 — the raw bits are reinterpreted, not converted.
      const forcesUniformsBuf = new ArrayBuffer(16);
      new Uint32Array(forcesUniformsBuf, 0, 1)[0] = this.numParticles;
      const forcesUniformsF32 = new Float32Array(forcesUniformsBuf);
      forcesUniformsF32[1] = this.theta;
      forcesUniformsF32[2] = 1.0; // G
      forcesUniformsF32[3] = 2.0; // softening
      this.device.queue.writeBuffer(this.forcesUniformsBuffer, 0, forcesUniformsBuf);

      // Create bind groups (one-time).
      if (!this.forcesBindGroup) {
        this.forcesBindGroup = this.device.createBindGroup({
          layout: this.forcesPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.particleBuffer } },
            { binding: 1, resource: { buffer: this.octreeBuffer } },
            { binding: 2, resource: { buffer: this.forcesBuffer } },
            { binding: 3, resource: { buffer: this.forcesUniformsBuffer } },
          ],
        });
      }

      if (!this.integrateBindGroup) {
        this.integrateBindGroup = this.device.createBindGroup({
          layout: this.integratePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.particleBuffer } },
            { binding: 1, resource: { buffer: this.forcesBuffer } },
            { binding: 2, resource: { buffer: this.velocityBuffer } },
            { binding: 3, resource: { buffer: this.integrateUniformsBuffer } },
          ],
        });
      }

      // GPU forces + integrate + render copy
      const gpuStart = performance.now();
      const commandEncoder = this.device.createCommandEncoder();

      const forcesPass = commandEncoder.beginComputePass();
      forcesPass.setPipeline(this.forcesPipeline);
      forcesPass.setBindGroup(0, this.forcesBindGroup);
      const workgroupCount = Math.ceil(this.numParticles / 256);
      forcesPass.dispatchWorkgroups(workgroupCount);
      forcesPass.end();

      const integratePass = commandEncoder.beginComputePass();
      integratePass.setPipeline(this.integratePipeline);
      integratePass.setBindGroup(0, this.integrateBindGroup);
      integratePass.dispatchWorkgroups(workgroupCount);
      integratePass.end();

      commandEncoder.copyBufferToBuffer(
        this.particleBuffer,
        0,
        this.renderPositionBuffer,
        0,
        this.numParticles * 4 * 4
      );

      this.device.queue.submit([commandEncoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const gpuTime = performance.now() - gpuStart;

      // --- Octree rebuild pipeline (after render buffer is updated) ---
      // Pipelined across two frames to avoid a single large spike:
      //   Frame N:   kick off GPU→CPU copy, start async mapAsync
      //   Frame N+1: await map, read data, build octree, upload
      // The GPU→CPU DMA transfer overlaps with frame N+1's GPU compute.

      if (this.rebuildPhase === 'downloading') {
        const rebuildStart = performance.now();
        await this.downloadPromise!;
        const gpuData = new Float32Array(this.stagingBuffer.getMappedRange());
        unpackParticlesFromGPU(gpuData, this.particlesCPU, this.numParticles);
        this.stagingBuffer.unmap();

        const octree = new Octree(this.particlesCPU);
        this.uploadOctree(octree);

        this.rebuildPhase = 'idle';
        this.framesSinceRebuild = 0;

        if (frameCount <= 2) {
          console.log(`  Rebuild: ${(performance.now() - rebuildStart).toFixed(1)}ms`);
        }
      }

      this.framesSinceRebuild++;
      if (this.rebuildPhase === 'idle' && this.framesSinceRebuild >= this.octreeRebuildInterval) {
        // Kick off the async download — resolves while the next frame's
        // GPU compute is running, so the DMA cost is mostly hidden.
        const dlEncoder = this.device.createCommandEncoder();
        dlEncoder.copyBufferToBuffer(
          this.particleBuffer, 0,
          this.stagingBuffer, 0,
          this.numParticles * 4 * 4
        );
        this.device.queue.submit([dlEncoder.finish()]);
        this.downloadPromise = this.stagingBuffer.mapAsync(GPUMapMode.READ);
        this.rebuildPhase = 'downloading';
      }

      if (frameCount === 0) {
        console.log(`GPU Barnes-Hut profile (rebuild every ${this.octreeRebuildInterval} frames):`);
        console.log(`  GPU compute: ${gpuTime.toFixed(1)}ms`);
        console.log(`  Total: ${(performance.now() - startTime).toFixed(1)}ms`);
      }
      frameCount++;

      this.physicsFrameCount++;
      const elapsed = performance.now() - startTime;
      this.monitor.recordPhysicsFrame(elapsed);

      const targetFrameTime = 1000 / this.targetPhysicsFPS;
      const sleepTime = Math.max(0, targetFrameTime - elapsed);

      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime));
      }

      this.lastPhysicsTime = performance.now();
    }
  }

  private uploadOctree(octree: Octree): void {
    const { view } = this.serializeOctree(octree);
    this.device.queue.writeBuffer(
      this.octreeBuffer, 0,
      view.buffer, view.byteOffset, view.byteLength
    );
  }

  getDevice(): GPUDevice {
    return this.device;
  }

  getRenderPositionBuffer(): GPUBuffer {
    return this.renderPositionBuffer;
  }

  getParticleCount(): number {
    return this.numParticles;
  }

  getPhysicsFrameCount(): number {
    return this.physicsFrameCount;
  }

  setTargetPhysicsFPS(fps: number): void {
    this.targetPhysicsFPS = Math.max(1, Math.min(60, fps));
  }

  setTheta(theta: number): void {
    this.theta = Math.max(0.1, Math.min(1.5, theta));
  }

  getTheta(): number {
    return this.theta;
  }

  setOctreeRebuildInterval(interval: number): void {
    this.octreeRebuildInterval = Math.max(1, Math.min(16, interval));
  }

  getOctreeRebuildInterval(): number {
    return this.octreeRebuildInterval;
  }

  getPhysicsProgress(): number {
    // Calculate progress to next physics frame (0.0 to 1.0)
    const frameTime = 1000 / this.targetPhysicsFPS;
    const elapsed = performance.now() - this.lastPhysicsTime;
    return Math.min(1.0, elapsed / frameTime);
  }

  destroy(): void {
    this.running = false;
    this.particleBuffer.destroy();
    this.renderPositionBuffer.destroy();
    this.velocityBuffer.destroy();
    this.forcesBuffer.destroy();
    this.octreeBuffer.destroy();
    this.forcesUniformsBuffer.destroy();
    this.integrateUniformsBuffer.destroy();
    this.stagingBuffer.destroy();
  }
}
