/**
 * Real-time N-body simulation using GPU Barnes-Hut algorithm
 * Phase 3: Hybrid approach - CPU builds octree, GPU traverses for forces
 */

import { initializeNBodyParticles } from '../initialization';
import { buildFlatOctree, BYTES_PER_NODE } from '../barnesHut/flatOctree';
import { packParticlesForGPU, packVelocitiesForGPU, unpackParticlesFromGPU } from './barnesHutPacking';
import { PerformanceMonitor } from './performanceMonitor';
import OctreeWorker from './octreeWorker?worker';

export interface RealtimeSimulationGPUBarnesHutOptions {
  device: GPUDevice;
  numParticles: number;
  deltaT?: number;
  targetPhysicsFPS?: number;
  theta?: number;
  octreeRebuildInterval?: number;
}

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

  // Hoisted uniform data to avoid per-frame allocations
  private forcesUniformsBuf = new ArrayBuffer(16);
  private forcesUniformsU32 = new Uint32Array(this.forcesUniformsBuf);
  private forcesUniformsF32 = new Float32Array(this.forcesUniformsBuf);
  private rebuildPhase: 'idle' | 'downloading' | 'building' = 'idle';
  private downloadPromise: Promise<void> | null = null;
  private pendingOctreeResult: { buffer: ArrayBuffer; nodeCount: number; particleData: ArrayBuffer } | null = null;
  private framesSinceRebuild = 0;
  public targetPhysicsFPS: number;
  public monitor: PerformanceMonitor;

  // Web Worker for off-thread octree construction
  private octreeWorker: Worker;
  private maxOctreeNodes: number;

  constructor(options: RealtimeSimulationGPUBarnesHutOptions) {
    this.device = options.device;
    this.numParticles = options.numParticles;
    this.deltaT = options.deltaT ?? 0.01;
    this.targetPhysicsFPS = options.targetPhysicsFPS ?? 20;
    this.theta = options.theta ?? 0.8;
    this.octreeRebuildInterval = options.octreeRebuildInterval ?? 4;

    // Initialize CPU particle array for octree building
    this.particlesCPU = initializeNBodyParticles(this.numParticles);

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
    this.maxOctreeNodes = this.numParticles * 8; // Worst case: many internal nodes
    const octreeBufferSize = this.maxOctreeNodes * BYTES_PER_NODE;

    // Spawn worker for off-thread octree construction
    this.octreeWorker = new OctreeWorker();
    this.octreeWorker.onmessage = (e: MessageEvent) => {
      this.pendingOctreeResult = e.data as { buffer: ArrayBuffer; nodeCount: number; particleData: ArrayBuffer };
    };

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

  private uploadParticlesToGPU(): void {
    const gpuData = packParticlesForGPU(this.particlesCPU, this.numParticles);
    this.device.queue.writeBuffer(this.particleBuffer, 0, gpuData);
  }

  private uploadVelocitiesToGPU(): void {
    const velocities = packVelocitiesForGPU(this.particlesCPU, this.numParticles);
    this.device.queue.writeBuffer(this.velocityBuffer, 0, velocities);
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
    this.uploadInitialOctree();

    while (this.running) {
      await new Promise((resolve) => setTimeout(resolve, 0));

      const startTime = performance.now();

      // --- Fast path: always runs first so the render buffer updates ---
      // --- promptly even if a rebuild blocks the thread afterward.    ---

      // Update forces uniforms (theta may change at runtime via setTheta).
      // numParticles must be written as u32 (not f32) because the shader
      // declares it as u32 — the raw bits are reinterpreted, not converted.
      this.forcesUniformsU32[0] = this.numParticles;
      this.forcesUniformsF32[1] = this.theta;
      this.forcesUniformsF32[2] = 1.0; // G
      this.forcesUniformsF32[3] = 2.0; // softening
      this.device.queue.writeBuffer(this.forcesUniformsBuffer, 0, this.forcesUniformsBuf);

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

      // --- Octree rebuild pipeline (fully non-blocking) ---
      // Three phases, each on a separate frame, so no phase blocks the
      // main thread long enough to stall requestAnimationFrame:
      //   Frame N:   kick off GPU→CPU copy + async mapAsync
      //   Frame N+1: await map (fast — DMA already finished), read data,
      //              send to Web Worker for octree build
      //   Frame N+2+: worker posts back serialized buffer, upload to GPU
      //
      // The worker result may arrive mid-frame via onmessage; we check
      // for it at the top of the pipeline section each frame.

      if (this.pendingOctreeResult) {
        this.device.queue.writeBuffer(
          this.octreeBuffer, 0, this.pendingOctreeResult.buffer
        );
        if (frameCount <= 2) {
          console.log(`  Octree uploaded (${this.pendingOctreeResult.nodeCount} nodes)`);
        }

        // Restore ownership of the CPU particle buffer from the worker
        // so we can reuse it for the next download phase without reallocating.
        this.particlesCPU = new Float32Array(this.pendingOctreeResult.particleData);

        this.pendingOctreeResult = null;
        this.rebuildPhase = 'idle';
        this.framesSinceRebuild = 0;
      }

      if (this.rebuildPhase === 'downloading') {
        await this.downloadPromise!;
        const gpuData = new Float32Array(this.stagingBuffer.getMappedRange());
        unpackParticlesFromGPU(gpuData, this.particlesCPU, this.numParticles);
        this.stagingBuffer.unmap();

        // Hand off to worker — octree build + serialize happens off-thread.
        // Transfer the buffer directly (0-copy) to the worker, avoiding
        // allocating and slicing a new buffer every rebuild.
        const workerData = this.particlesCPU.buffer;
        this.octreeWorker.postMessage(
          { particleData: workerData, maxNodes: this.maxOctreeNodes },
          [workerData]
        );
        this.rebuildPhase = 'building';
      }

      this.framesSinceRebuild++;
      if (this.rebuildPhase === 'idle' && this.framesSinceRebuild >= this.octreeRebuildInterval) {
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

  private uploadInitialOctree(): void {
    const { buffer } = buildFlatOctree(this.particlesCPU, { maxNodes: this.maxOctreeNodes });
    this.device.queue.writeBuffer(this.octreeBuffer, 0, buffer);
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
    this.octreeWorker.terminate();
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
