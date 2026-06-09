/**
 * Real-time N-body simulation using GPU compute
 * Phase 1: Brute-force O(N²) GPU physics with temporal interpolation
 */

import {
  FLOATS_PER_PARTICLE,
  OFFSET_X,
  OFFSET_Y,
  OFFSET_Z,
  OFFSET_VX,
  OFFSET_VY,
  OFFSET_VZ,
  OFFSET_MASS,
} from '../particleData';
import { initializeNBodyParticles } from '../initialization';
import { PerformanceMonitor } from './performanceMonitor';

export interface RealtimeSimulationOptions {
  numParticles: number;
  deltaT?: number;
  targetPhysicsFPS?: number;
}

export class RealtimeNBodySimulation {
  // Particle data (CPU-side, only for initialization)
  private initialParticles: Float32Array;

  // GPU resources - double buffered particle data
  private device: GPUDevice;
  private particleBufferCurrent!: GPUBuffer;
  private particleBufferNext!: GPUBuffer;
  private forceBuffer!: GPUBuffer;
  private uniformBuffer!: GPUBuffer;
  private interpolationUniformBuffer!: GPUBuffer;

  // Compute pipelines
  private forcePipeline!: GPUComputePipeline;
  private kickDriftPipeline!: GPUComputePipeline;
  private kickPipeline!: GPUComputePipeline;
  private interpolatePipeline!: GPUComputePipeline;

  // Bind groups
  private forceBindGroup!: GPUBindGroup;
  private kickDriftBindGroup!: GPUBindGroup;
  private kickBindGroup!: GPUBindGroup;
  private interpolateBindGroup!: GPUBindGroup;

  // Render buffer (interpolated positions for rendering)
  private renderPositionBuffer!: GPUBuffer;

  // GPU buffer layout constants (matches existing nbody.ts)
  private readonly GPU_FLOATS_PER_PARTICLE = 12;
  private readonly GPU_POS_X = 0;
  private readonly GPU_POS_Y = 1;
  private readonly GPU_POS_Z = 2;
  private readonly GPU_POS_PAD = 3;
  private readonly GPU_VEL_X = 4;
  private readonly GPU_VEL_Y = 5;
  private readonly GPU_VEL_Z = 6;
  private readonly GPU_MASS = 7;
  private readonly GPU_MASS_PAD = 8;

  // Simulation state
  private numParticles: number;
  private deltaT: number;
  private running = false;
  private physicsFrameCount = 0;
  private lastPhysicsTime = 0;

  // Performance
  public monitor: PerformanceMonitor;
  public targetPhysicsFPS: number;

  constructor(device: GPUDevice, options: RealtimeSimulationOptions) {
    this.device = device;
    this.numParticles = options.numParticles;
    this.deltaT = options.deltaT ?? 0.01;
    this.targetPhysicsFPS = options.targetPhysicsFPS ?? 20;

    // Initialize performance monitor
    this.monitor = new PerformanceMonitor();

    // Initialize particles using shared logic (scales radius with N)
    this.initialParticles = initializeNBodyParticles(this.numParticles);

    // Set up GPU resources
    this.setupGPU();
  }

  private setupGPU(): void {
    // Particle shader structure
    const computeForceShader = `
      struct Particle {
          pos: vec3f,
          vel: vec3f,
          mass: f32,
          _pad: f32,
      }

      @group(0) @binding(0) var<storage, read> particles: array<Particle>;
      @group(0) @binding(1) var<storage, read_write> forces: array<vec3f>;

      const G: f32 = 1.0;
      const SOFTENING: f32 = 2.0;

      @compute @workgroup_size(256)
      fn computeForces(@builtin(global_invocation_id) id: vec3u) {
          let i = id.x;
          if (i >= arrayLength(&particles)) { return; }

          var force = vec3f(0.0, 0.0, 0.0);
          let pi = particles[i].pos;
          let mi = particles[i].mass;

          for (var j = 0u; j < arrayLength(&particles); j++) {
              if (i == j) { continue; }

              let r = particles[j].pos - pi;
              let r2 = dot(r, r) + SOFTENING * SOFTENING;
              let invR = 1.0 / sqrt(r2);
              let invR3 = invR * invR * invR;
              let f = G * mi * particles[j].mass * invR3;

              force += f * r;
          }

          forces[i] = force;
      }
    `;

    const interpolateShader = `
      struct Particle {
          pos: vec3f,
          vel: vec3f,
          mass: f32,
          _pad: f32,
      }

      struct Uniforms {
          alpha: f32,
      }

      @group(0) @binding(0) var<storage, read> particlesCurrent: array<Particle>;
      @group(0) @binding(1) var<storage, read> particlesNext: array<Particle>;
      @group(0) @binding(2) var<storage, read_write> positions: array<vec3f>;
      @group(0) @binding(3) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(256)
      fn interpolate(@builtin(global_invocation_id) id: vec3u) {
          let i = id.x;
          if (i >= arrayLength(&particlesCurrent)) { return; }

          let pos0 = particlesCurrent[i].pos;
          let pos1 = particlesNext[i].pos;
          positions[i] = mix(pos0, pos1, uniforms.alpha);
      }
    `;

    const kickDriftShader = `
      struct Particle {
          pos: vec3f,
          vel: vec3f,
          mass: f32,
          _pad: f32,
      }

      struct Uniforms {
          dt: f32,
      }

      @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
      @group(0) @binding(1) var<storage, read> forces: array<vec3f>;
      @group(0) @binding(2) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(256)
      fn kickDrift(@builtin(global_invocation_id) id: vec3u) {
          let i = id.x;
          if (i >= arrayLength(&particles)) { return; }

          let mass = particles[i].mass;
          let accel = forces[i] / mass;

          // Half-step velocity update (kick)
          particles[i].vel += accel * uniforms.dt * 0.5;

          // Full-step position update (drift)
          particles[i].pos += particles[i].vel * uniforms.dt;
      }
    `;

    const kickShader = `
      struct Particle {
          pos: vec3f,
          vel: vec3f,
          mass: f32,
          _pad: f32,
      }

      struct Uniforms {
          dt: f32,
      }

      @group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
      @group(0) @binding(1) var<storage, read> forces: array<vec3f>;
      @group(0) @binding(2) var<uniform> uniforms: Uniforms;

      @compute @workgroup_size(256)
      fn kick(@builtin(global_invocation_id) id: vec3u) {
          let i = id.x;
          if (i >= arrayLength(&particles)) { return; }

          let mass = particles[i].mass;
          let accel = forces[i] / mass;

          // Half-step velocity update
          particles[i].vel += accel * uniforms.dt * 0.5;
      }
    `;

    // Convert to GPU format
    const gpuParticleData = this.convertToGPUFormat(this.initialParticles);

    // Create double-buffered particle buffers
    this.particleBufferCurrent = this.device.createBuffer({
      size: gpuParticleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.particleBufferCurrent.getMappedRange()).set(gpuParticleData);
    this.particleBufferCurrent.unmap();

    this.particleBufferNext = this.device.createBuffer({
      size: gpuParticleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.particleBufferNext.getMappedRange()).set(gpuParticleData);
    this.particleBufferNext.unmap();

    this.forceBuffer = this.device.createBuffer({
      size: this.numParticles * 4 * 4, // vec3f requires 16-byte alignment
      usage: GPUBufferUsage.STORAGE,
    });

    this.uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.uniformBuffer.getMappedRange()).set([this.deltaT, 0, 0, 0]);
    this.uniformBuffer.unmap();

    // Interpolation uniform buffer (alpha value)
    this.interpolationUniformBuffer = this.device.createBuffer({
      size: 16, // Padding for alignment
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Render position buffer (vec3f per particle, with alignment)
    this.renderPositionBuffer = this.device.createBuffer({
      size: this.numParticles * 4 * 4, // vec3f requires 16-byte alignment
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX,
    });

    // Create pipelines
    const forceModule = this.device.createShaderModule({ code: computeForceShader });
    const kickDriftModule = this.device.createShaderModule({ code: kickDriftShader });
    const kickModule = this.device.createShaderModule({ code: kickShader });
    const interpolateModule = this.device.createShaderModule({ code: interpolateShader });

    this.forcePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: forceModule, entryPoint: 'computeForces' },
    });

    this.kickDriftPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: kickDriftModule, entryPoint: 'kickDrift' },
    });

    this.kickPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: kickModule, entryPoint: 'kick' },
    });

    this.interpolatePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: interpolateModule, entryPoint: 'interpolate' },
    });

    // Create bind groups (will be updated in computePhysicsStep to swap buffers)
    this.updateBindGroups();
  }

  private updateBindGroups(): void {
    this.forceBindGroup = this.device.createBindGroup({
      layout: this.forcePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBufferCurrent } },
        { binding: 1, resource: { buffer: this.forceBuffer } },
      ],
    });

    this.kickDriftBindGroup = this.device.createBindGroup({
      layout: this.kickDriftPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBufferCurrent } },
        { binding: 1, resource: { buffer: this.forceBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    });

    this.kickBindGroup = this.device.createBindGroup({
      layout: this.kickPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBufferCurrent } },
        { binding: 1, resource: { buffer: this.forceBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    });

    this.interpolateBindGroup = this.device.createBindGroup({
      layout: this.interpolatePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBufferCurrent } },
        { binding: 1, resource: { buffer: this.particleBufferNext } },
        { binding: 2, resource: { buffer: this.renderPositionBuffer } },
        { binding: 3, resource: { buffer: this.interpolationUniformBuffer } },
      ],
    });
  }

  private convertToGPUFormat(particles: Float32Array): Float32Array {
    const gpuData = new Float32Array(this.numParticles * this.GPU_FLOATS_PER_PARTICLE);

    for (let i = 0; i < this.numParticles; i++) {
      const srcOffset = i * FLOATS_PER_PARTICLE;
      const dstOffset = i * this.GPU_FLOATS_PER_PARTICLE;

      gpuData[dstOffset + this.GPU_POS_X] = particles[srcOffset + OFFSET_X];
      gpuData[dstOffset + this.GPU_POS_Y] = particles[srcOffset + OFFSET_Y];
      gpuData[dstOffset + this.GPU_POS_Z] = particles[srcOffset + OFFSET_Z];
      gpuData[dstOffset + this.GPU_POS_PAD] = 0;
      gpuData[dstOffset + this.GPU_VEL_X] = particles[srcOffset + OFFSET_VX];
      gpuData[dstOffset + this.GPU_VEL_Y] = particles[srcOffset + OFFSET_VY];
      gpuData[dstOffset + this.GPU_VEL_Z] = particles[srcOffset + OFFSET_VZ];
      gpuData[dstOffset + this.GPU_MASS] = particles[srcOffset + OFFSET_MASS];
      gpuData[dstOffset + this.GPU_MASS_PAD] = 0;
    }

    return gpuData;
  }


  async start(): Promise<void> {
    this.running = true;
    this.lastPhysicsTime = performance.now();
    this.physicsLoop();
  }

  stop(): void {
    this.running = false;
  }

  private async physicsLoop(): Promise<void> {
    while (this.running) {
      const startTime = performance.now();

      // Compute next physics step
      await this.computePhysicsStep();

      // Swap GPU buffers (double buffering)
      [this.particleBufferCurrent, this.particleBufferNext] =
        [this.particleBufferNext, this.particleBufferCurrent];

      // Update bind groups to point to swapped buffers
      this.updateBindGroups();

      this.physicsFrameCount++;
      const elapsed = performance.now() - startTime;
      this.monitor.recordPhysicsFrame(elapsed);

      // Maintain target framerate
      const targetFrameTime = 1000 / this.targetPhysicsFPS;
      const sleepTime = Math.max(0, targetFrameTime - elapsed);

      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime));
      }

      this.lastPhysicsTime = performance.now();
    }
  }

  private async computePhysicsStep(): Promise<void> {
    const workgroupCount = Math.ceil(this.numParticles / 256);

    // Leapfrog integration: kick-drift-kick
    const commandEncoder = this.device.createCommandEncoder();

    // 1. Compute forces
    const forcePass1 = commandEncoder.beginComputePass();
    forcePass1.setPipeline(this.forcePipeline);
    forcePass1.setBindGroup(0, this.forceBindGroup);
    forcePass1.dispatchWorkgroups(workgroupCount);
    forcePass1.end();

    // 2. Kick + drift
    const kickDriftPass = commandEncoder.beginComputePass();
    kickDriftPass.setPipeline(this.kickDriftPipeline);
    kickDriftPass.setBindGroup(0, this.kickDriftBindGroup);
    kickDriftPass.dispatchWorkgroups(workgroupCount);
    kickDriftPass.end();

    // 3. Recompute forces
    const forcePass2 = commandEncoder.beginComputePass();
    forcePass2.setPipeline(this.forcePipeline);
    forcePass2.setBindGroup(0, this.forceBindGroup);
    forcePass2.dispatchWorkgroups(workgroupCount);
    forcePass2.end();

    // 4. Second kick
    const kickPass = commandEncoder.beginComputePass();
    kickPass.setPipeline(this.kickPipeline);
    kickPass.setBindGroup(0, this.kickBindGroup);
    kickPass.dispatchWorkgroups(workgroupCount);
    kickPass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Wait for GPU to finish (non-blocking, just ensures queue flush)
    await this.device.queue.onSubmittedWorkDone();
  }

  /**
   * Perform GPU interpolation and return the render position buffer
   * This should be called from the render loop at 60 FPS
   */
  getRenderPositionBuffer(): GPUBuffer {
    const alpha = this.getPhysicsProgress();

    // Update interpolation uniform with current alpha
    this.device.queue.writeBuffer(
      this.interpolationUniformBuffer,
      0,
      new Float32Array([alpha, 0, 0, 0])
    );

    // Run interpolation compute shader
    const commandEncoder = this.device.createCommandEncoder();
    const interpolatePass = commandEncoder.beginComputePass();
    interpolatePass.setPipeline(this.interpolatePipeline);
    interpolatePass.setBindGroup(0, this.interpolateBindGroup);
    interpolatePass.dispatchWorkgroups(Math.ceil(this.numParticles / 256));
    interpolatePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    return this.renderPositionBuffer;
  }

  getPhysicsProgress(): number {
    // Calculate progress to next physics frame (0.0 to 1.0)
    const frameTime = 1000 / this.targetPhysicsFPS;
    const elapsed = performance.now() - this.lastPhysicsTime;
    return Math.min(1.0, elapsed / frameTime);
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

  getDevice(): GPUDevice {
    return this.device;
  }

  // Clean up GPU resources
  destroy(): void {
    this.running = false;
    this.particleBufferCurrent?.destroy();
    this.particleBufferNext?.destroy();
    this.forceBuffer?.destroy();
    this.uniformBuffer?.destroy();
    this.interpolationUniformBuffer?.destroy();
    this.renderPositionBuffer?.destroy();
  }
}
