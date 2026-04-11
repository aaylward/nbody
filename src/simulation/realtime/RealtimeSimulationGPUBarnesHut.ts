/**
 * Real-time N-body simulation using GPU Barnes-Hut algorithm
 * Phase 3: Hybrid approach - CPU builds octree, GPU traverses for forces
 */

import {
  createParticleArray,
  setParticle,
  removeCenterOfMassVelocity,
} from '../particleData';
import { Octree } from '../barnesHut/octree';
import { PerformanceMonitor } from './performanceMonitor';

export interface RealtimeSimulationGPUBarnesHutOptions {
  device: GPUDevice;
  numParticles: number;
  deltaT?: number;
  targetPhysicsFPS?: number;
  theta?: number;
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
  private uniformsBuffer: GPUBuffer;
  private stagingBuffer: GPUBuffer; // Reuse staging buffer for downloads
  private forcesPipeline: GPUComputePipeline;
  private integratePipeline: GPUComputePipeline;
  private forcesBindGroup: GPUBindGroup | null = null;
  private integrateBindGroup: GPUBindGroup | null = null;

  // CPU octree
  private particlesCPU: Float32Array;
  private theta: number;
  public targetPhysicsFPS: number;
  public monitor: PerformanceMonitor;

  constructor(options: RealtimeSimulationGPUBarnesHutOptions) {
    this.device = options.device;
    this.numParticles = options.numParticles;
    this.deltaT = options.deltaT ?? 0.01;
    this.targetPhysicsFPS = options.targetPhysicsFPS ?? 20;
    this.theta = options.theta ?? 0.8;

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

    this.uniformsBuffer = this.device.createBuffer({
      size: 16, // 4 floats: numParticles, theta, G, softening (or deltaT)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.stagingBuffer = this.device.createBuffer({
      size: particleBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Upload initial data
    this.uploadParticlesToGPU();
    this.uploadVelocitiesToGPU();

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
    // Pack particles for GPU: [x, y, z, mass] per particle
    const gpuData = new Float32Array(this.numParticles * 4);
    for (let i = 0; i < this.numParticles; i++) {
      const offset = i * 7; // CPU format
      const gpuOffset = i * 4;
      gpuData[gpuOffset + 0] = this.particlesCPU[offset + 0]; // x
      gpuData[gpuOffset + 1] = this.particlesCPU[offset + 1]; // y
      gpuData[gpuOffset + 2] = this.particlesCPU[offset + 2]; // z
      gpuData[gpuOffset + 3] = this.particlesCPU[offset + 6]; // mass
    }

    this.device.queue.writeBuffer(this.particleBuffer, 0, gpuData);
  }

  private uploadVelocitiesToGPU(): void {
    // Pack velocities for GPU: [vx, vy, vz, _pad] per particle.
    // The shader sees this as array<vec3f>, which has a 16-byte stride, so
    // each velocity needs a 4-float slot on the host side too.
    const velocities = new Float32Array(this.numParticles * 4);
    for (let i = 0; i < this.numParticles; i++) {
      const offset = i * 7; // CPU format
      const vOffset = i * 4;
      velocities[vOffset + 0] = this.particlesCPU[offset + 3]; // vx
      velocities[vOffset + 1] = this.particlesCPU[offset + 4]; // vy
      velocities[vOffset + 2] = this.particlesCPU[offset + 5]; // vz
      // velocities[vOffset + 3] left as 0 (padding)
    }

    this.device.queue.writeBuffer(this.velocityBuffer, 0, velocities);
  }

  private serializeOctree(octree: Octree): { buffer: ArrayBuffer; nodeCount: number } {
    // Flatten octree into breadth-first array for GPU
    // Each node: 5 floats (center, mass, width) + 3 u32s (childStart, childCount, particleCount)
    const nodes: Array<{
      floats: Float32Array;
      ints: Uint32Array;
    }> = [];
    const queue = [{ node: octree.getRoot(), index: 0 }];
    let nextIndex = 1;

    while (queue.length > 0) {
      const { node, index } = queue.shift()!;

      // Serialize this node
      const cellWidth = Math.max(
        node.bounds.max.x - node.bounds.min.x,
        node.bounds.max.y - node.bounds.min.y,
        node.bounds.max.z - node.bounds.min.z
      );

      const floats = new Float32Array(FLOATS_PER_NODE);
      floats[0] = node.centerOfMass.x;
      floats[1] = node.centerOfMass.y;
      floats[2] = node.centerOfMass.z;
      floats[3] = node.totalMass;
      floats[4] = cellWidth;

      const ints = new Uint32Array(INTS_PER_NODE);
      ints[0] = node.children && node.children.length > 0 ? nextIndex : 0; // childStart
      ints[1] = node.children ? node.children.length : 0; // childCount
      ints[2] = node.particleCount;

      nodes[index] = { floats, ints };

      // Enqueue children
      if (node.children && node.children.length > 0) {
        for (const child of node.children) {
          queue.push({ node: child, index: nextIndex });
          nextIndex++;
        }
      }
    }

    // Pack all nodes into a single buffer with correct memory layout
    const totalBytes = nodes.length * BYTES_PER_NODE;
    const buffer = new ArrayBuffer(totalBytes);
    const floatView = new Float32Array(buffer);
    const intView = new Uint32Array(buffer);

    for (let i = 0; i < nodes.length; i++) {
      const nodeOffset = (i * BYTES_PER_NODE) / 4; // offset in 32-bit words

      // Write floats (5 floats = 20 bytes)
      floatView[nodeOffset + 0] = nodes[i].floats[0];
      floatView[nodeOffset + 1] = nodes[i].floats[1];
      floatView[nodeOffset + 2] = nodes[i].floats[2];
      floatView[nodeOffset + 3] = nodes[i].floats[3];
      floatView[nodeOffset + 4] = nodes[i].floats[4];

      // Write ints (3 u32s = 12 bytes), immediately after floats
      intView[nodeOffset + 5] = nodes[i].ints[0];
      intView[nodeOffset + 6] = nodes[i].ints[1];
      intView[nodeOffset + 7] = nodes[i].ints[2];
    }

    return { buffer, nodeCount: nodes.length };
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
    while (this.running) {
      await new Promise((resolve) => setTimeout(resolve, 0));

      const startTime = performance.now();

      // 1. Download particles from GPU to CPU for octree building
      const downloadStart = performance.now();
      await this.downloadParticlesFromGPU();
      const downloadTime = performance.now() - downloadStart;

      // 2. Build octree on CPU
      const buildStart = performance.now();
      const octree = new Octree(this.particlesCPU);
      const buildTime = performance.now() - buildStart;

      // 3. Serialize and upload octree to GPU
      const serializeStart = performance.now();
      const { buffer: octreeData, nodeCount } = this.serializeOctree(octree);
      this.device.queue.writeBuffer(this.octreeBuffer, 0, octreeData);
      const serializeTime = performance.now() - serializeStart;

      // 4. Update uniforms
      const uniforms = new Float32Array([
        this.numParticles,
        this.theta,
        1.0, // G
        2.0, // softening
      ]);
      this.device.queue.writeBuffer(this.uniformsBuffer, 0, uniforms);

      // 5. Create bind groups (if not already created or if octree changed)
      if (!this.forcesBindGroup) {
        this.forcesBindGroup = this.device.createBindGroup({
          layout: this.forcesPipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.particleBuffer } },
            { binding: 1, resource: { buffer: this.octreeBuffer } },
            { binding: 2, resource: { buffer: this.forcesBuffer } },
            { binding: 3, resource: { buffer: this.uniformsBuffer } },
          ],
        });
      }

      if (!this.integrateBindGroup) {
        const integrateUniforms = new Float32Array([this.numParticles, this.deltaT]);
        this.device.queue.writeBuffer(this.uniformsBuffer, 0, integrateUniforms);

        this.integrateBindGroup = this.device.createBindGroup({
          layout: this.integratePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.particleBuffer } },
            { binding: 1, resource: { buffer: this.forcesBuffer } },
            { binding: 2, resource: { buffer: this.velocityBuffer } },
            { binding: 3, resource: { buffer: this.uniformsBuffer } },
          ],
        });
      }

      // 6. Run force computation on GPU
      const gpuStart = performance.now();
      const commandEncoder = this.device.createCommandEncoder();

      const forcesPass = commandEncoder.beginComputePass();
      forcesPass.setPipeline(this.forcesPipeline);
      forcesPass.setBindGroup(0, this.forcesBindGroup);
      const workgroupCount = Math.ceil(this.numParticles / 256);
      forcesPass.dispatchWorkgroups(workgroupCount);
      forcesPass.end();

      // 7. Run integration on GPU
      const integratePass = commandEncoder.beginComputePass();
      integratePass.setPipeline(this.integratePipeline);
      integratePass.setBindGroup(0, this.integrateBindGroup);
      integratePass.dispatchWorkgroups(workgroupCount);
      integratePass.end();

      // 8. Copy to render buffer
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

      if (frameCount === 0) {
        console.log(`GPU Barnes-Hut profile:`);
        console.log(`  Download: ${downloadTime.toFixed(1)}ms`);
        console.log(`  Build octree: ${buildTime.toFixed(1)}ms (${nodeCount} nodes)`);
        console.log(`  Serialize: ${serializeTime.toFixed(1)}ms`);
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

  private async downloadParticlesFromGPU(): Promise<void> {
    // Read back particles from GPU to CPU for octree building
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.particleBuffer,
      0,
      this.stagingBuffer,
      0,
      this.numParticles * 4 * 4
    );
    this.device.queue.submit([commandEncoder.finish()]);

    // Wait for copy to complete and map
    await this.stagingBuffer.mapAsync(GPUMapMode.READ);
    const gpuData = new Float32Array(this.stagingBuffer.getMappedRange());

    // Unpack back to CPU format
    for (let i = 0; i < this.numParticles; i++) {
      const offset = i * 7;
      const gpuOffset = i * 4;
      this.particlesCPU[offset + 0] = gpuData[gpuOffset + 0]; // x
      this.particlesCPU[offset + 1] = gpuData[gpuOffset + 1]; // y
      this.particlesCPU[offset + 2] = gpuData[gpuOffset + 2]; // z
      this.particlesCPU[offset + 6] = gpuData[gpuOffset + 3]; // mass
    }

    this.stagingBuffer.unmap();
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
    this.uniformsBuffer.destroy();
    this.stagingBuffer.destroy();
  }
}
