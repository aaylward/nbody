import {
  createParticleArray,
  setParticle,
  cloneParticleData,
  removeCenterOfMassVelocity,
  FLOATS_PER_PARTICLE,
  OFFSET_X,
  OFFSET_Y,
  OFFSET_Z,
  OFFSET_VX,
  OFFSET_VY,
  OFFSET_VZ,
  OFFSET_MASS,
} from './particleData';

export interface NBodySimulationOptions {
  numParticles: number;
  numSnapshots: number;
  deltaT: number;
  onProgress?: (progress: number, message: string) => void;
}

let gpuDevice: GPUDevice | null = null;

// Constants for GPU layout (Optimized 32 bytes - 8 floats)
export const GPU_FLOATS_PER_PARTICLE = 8;
export const GPU_POS_X = 0;
export const GPU_POS_Y = 1;
export const GPU_POS_Z = 2;
export const GPU_POS_PAD = 3;
export const GPU_VEL_X = 4;
export const GPU_VEL_Y = 5;
export const GPU_VEL_Z = 6;
export const GPU_MASS = 7; // Stored in vel.w

export async function initGPU(): Promise<boolean> {
  if (!navigator.gpu) {
    console.log('WebGPU not available - using CPU fallback');
    return false;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log('WebGPU adapter not available');
      return false;
    }

    gpuDevice = await adapter.requestDevice();
    console.log('WebGPU initialized successfully! 🚀');
    return true;
  } catch (e) {
    console.error('WebGPU initialization failed:', e);
    return false;
  }
}

export class NBodyGPU {
  device: GPUDevice;
  numParticles: number;

  particleBuffer!: GPUBuffer;
  forceBuffer!: GPUBuffer;
  uniformBuffer!: GPUBuffer;

  // Pipelines
  forcePipeline!: GPUComputePipeline;
  kickDriftPipeline!: GPUComputePipeline;
  kickPipeline!: GPUComputePipeline;

  // Bind Groups
  forceBindGroup!: GPUBindGroup;
  kickDriftBindGroup!: GPUBindGroup;
  kickBindGroup!: GPUBindGroup;

  // Readback
  readBuffer!: GPUBuffer;
  stagingBuffer!: GPUBuffer; // Used for copying from particle buffer before mapping

  workgroupCount: number;

  // Shaders
  private computeForceShader = `
    struct Particle {
        pos: vec4f,
        vel: vec4f,
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
        let pi = particles[i].pos.xyz;
        let mi = particles[i].vel.w;

        for (var j = 0u; j < arrayLength(&particles); j++) {
            if (i == j) { continue; }

            let r = particles[j].pos.xyz - pi;
            let r2 = dot(r, r) + SOFTENING * SOFTENING;
            // Optimization: Use fast inverse square root intrinsic
            let invR = inverseSqrt(r2);
            let invR3 = invR * invR * invR;
            let f = G * mi * particles[j].vel.w * invR3;

            force += f * r;
        }

        forces[i] = force;
    }
  `;

  private kickDriftShader = `
    struct Particle {
        pos: vec4f,
        vel: vec4f,
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

        let mass = particles[i].vel.w;
        let accel = forces[i] / mass;

        // Half-step velocity update (kick)
        particles[i].vel.x += accel.x * uniforms.dt * 0.5;
        particles[i].vel.y += accel.y * uniforms.dt * 0.5;
        particles[i].vel.z += accel.z * uniforms.dt * 0.5;

        // Full-step position update (drift)
        particles[i].pos.x += particles[i].vel.x * uniforms.dt;
        particles[i].pos.y += particles[i].vel.y * uniforms.dt;
        particles[i].pos.z += particles[i].vel.z * uniforms.dt;
    }
  `;

  private kickShader = `
    struct Particle {
        pos: vec4f,
        vel: vec4f,
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

        let mass = particles[i].vel.w;
        let accel = forces[i] / mass;

        // Half-step velocity update
        particles[i].vel.x += accel.x * uniforms.dt * 0.5;
        particles[i].vel.y += accel.y * uniforms.dt * 0.5;
        particles[i].vel.z += accel.z * uniforms.dt * 0.5;
    }
  `;

  constructor(device: GPUDevice, numParticles: number) {
    this.device = device;
    this.numParticles = numParticles;
    this.workgroupCount = Math.ceil(numParticles / 256);
  }

  async init(deltaT: number) {
    // Initialize particles using TypedArray (CPU side first)
    const particles = createParticleArray(this.numParticles);

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

    // Orbiting particles
    for (let i = 1; i < this.numParticles; i++) {
      const r = 20 + Math.random() * 60;
      const theta = Math.random() * Math.PI * 2;
      const z = (Math.random() - 0.5) * 5;

      const x = r * Math.cos(theta);
      const y = r * Math.sin(theta);

      // Circular orbit velocity: v = sqrt(GM/r) where G=1.0, M=5000
      const v = Math.sqrt(5000 / r);
      const vx = -v * Math.sin(theta) + (Math.random() - 0.5) * 0.5;
      const vy = v * Math.cos(theta) + (Math.random() - 0.5) * 0.5;
      const vz = (Math.random() - 0.5) * 0.2;

      setParticle(particles, i, { x, y, z, vx, vy, vz, mass: 1 });
    }

    // Remove net momentum to prevent drift
    removeCenterOfMassVelocity(particles);

    // Convert from compact format (7 floats) to GPU format (8 floats - 32 bytes)
    const gpuParticleData = new Float32Array(this.numParticles * GPU_FLOATS_PER_PARTICLE);
    for (let i = 0; i < this.numParticles; i++) {
      const srcOffset = i * FLOATS_PER_PARTICLE;
      const dstOffset = i * GPU_FLOATS_PER_PARTICLE;

      // pos.xyz
      gpuParticleData[dstOffset + GPU_POS_X] = particles[srcOffset + OFFSET_X];
      gpuParticleData[dstOffset + GPU_POS_Y] = particles[srcOffset + OFFSET_Y];
      gpuParticleData[dstOffset + GPU_POS_Z] = particles[srcOffset + OFFSET_Z];
      gpuParticleData[dstOffset + GPU_POS_PAD] = 0; // Padding (pos.w)

      // vel.xyz + mass (vel.w)
      gpuParticleData[dstOffset + GPU_VEL_X] = particles[srcOffset + OFFSET_VX];
      gpuParticleData[dstOffset + GPU_VEL_Y] = particles[srcOffset + OFFSET_VY];
      gpuParticleData[dstOffset + GPU_VEL_Z] = particles[srcOffset + OFFSET_VZ];
      gpuParticleData[dstOffset + GPU_MASS] = particles[srcOffset + OFFSET_MASS];
    }

    // Create Buffers
    this.particleBuffer = this.device.createBuffer({
      size: gpuParticleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(this.particleBuffer.getMappedRange()).set(gpuParticleData);
    this.particleBuffer.unmap();

    this.forceBuffer = this.device.createBuffer({
      size: this.numParticles * 4 * 4, // vec3f in storage requires 16-byte alignment
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.uniformBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(this.uniformBuffer.getMappedRange()).set([deltaT, 0, 0, 0]);
    this.uniformBuffer.unmap();

    this.stagingBuffer = this.device.createBuffer({
      size: gpuParticleData.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // Create Pipelines
    const forceModule = this.device.createShaderModule({ code: this.computeForceShader });
    const kickDriftModule = this.device.createShaderModule({ code: this.kickDriftShader });
    const kickModule = this.device.createShaderModule({ code: this.kickShader });

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

    // Bind Groups
    this.forceBindGroup = this.device.createBindGroup({
      layout: this.forcePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.forceBuffer } },
      ],
    });

    this.kickDriftBindGroup = this.device.createBindGroup({
      layout: this.kickDriftPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.forceBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    });

    this.kickBindGroup = this.device.createBindGroup({
      layout: this.kickPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.particleBuffer } },
        { binding: 1, resource: { buffer: this.forceBuffer } },
        { binding: 2, resource: { buffer: this.uniformBuffer } },
      ],
    });

    // Compute initial forces so step() can skip the first calculation
    this.computeForces();
  }

  computeForces() {
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.forcePipeline);
    pass.setBindGroup(0, this.forceBindGroup);
    pass.dispatchWorkgroups(this.workgroupCount);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  step(_dt: number) {
    // We could update deltaT in uniform buffer here if it changes,
    // but for now we assume it's constant or update it sparingly.
    // If needed:
    // this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([dt]));

    const commandEncoder = this.device.createCommandEncoder();

    // Optimization: We skip the first force calculation because forces are
    // already computed at the end of the previous step (or in init).
    // This reduces the number of expensive O(N^2) compute passes by 50%.

    // 1. First kick + drift
    const kickDriftPass = commandEncoder.beginComputePass();
    kickDriftPass.setPipeline(this.kickDriftPipeline);
    kickDriftPass.setBindGroup(0, this.kickDriftBindGroup);
    kickDriftPass.dispatchWorkgroups(this.workgroupCount);
    kickDriftPass.end();

    // 3. Recompute forces at new positions
    const forcePass2 = commandEncoder.beginComputePass();
    forcePass2.setPipeline(this.forcePipeline);
    forcePass2.setBindGroup(0, this.forceBindGroup);
    forcePass2.dispatchWorkgroups(this.workgroupCount);
    forcePass2.end();

    // 4. Second kick
    const kickPass = commandEncoder.beginComputePass();
    kickPass.setPipeline(this.kickPipeline);
    kickPass.setBindGroup(0, this.kickBindGroup);
    kickPass.dispatchWorkgroups(this.workgroupCount);
    kickPass.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }

  async getParticleData(outData?: Float32Array): Promise<Float32Array> {
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      this.particleBuffer,
      0,
      this.stagingBuffer,
      0,
      this.stagingBuffer.size
    );
    this.device.queue.submit([commandEncoder.finish()]);

    await this.stagingBuffer.mapAsync(GPUMapMode.READ);
    const mappedRange = this.stagingBuffer.getMappedRange();

    if (outData) {
      outData.set(new Float32Array(mappedRange));
      this.stagingBuffer.unmap();
      return outData;
    }

    const data = new Float32Array(mappedRange.slice(0));
    this.stagingBuffer.unmap();

    return data;
  }
}

export let activeSimulation: NBodyGPU | null = null;
export function setActiveSimulation(sim: NBodyGPU | null) {
  activeSimulation = sim;
}

export async function initRealTimeSimulation(
  options: NBodySimulationOptions
): Promise<boolean> {
  const { numParticles, deltaT, onProgress } = options;
  const gpuAvailable = gpuDevice || (await initGPU());

  if (!gpuAvailable) {
    onProgress?.(100, 'WebGPU not available');
    return false;
  }

  if (!gpuDevice) throw new Error('GPU device not initialized');

  activeSimulation = new NBodyGPU(gpuDevice, numParticles);
  await activeSimulation.init(deltaT);

  onProgress?.(100, 'Simulation Ready');
  return true;
}

export async function generateNBodyDemo(
  options: NBodySimulationOptions
): Promise<Float32Array[]> {
  const { numParticles, numSnapshots, deltaT, onProgress } = options;

  const gpuAvailable = gpuDevice || (await initGPU());

  if (gpuAvailable) {
    return generateNBodyGPU({ numParticles, numSnapshots, deltaT, onProgress });
  } else {
    return generateNBodyCPU({ numParticles, numSnapshots, deltaT, onProgress });
  }
}

async function generateNBodyGPU(
  options: NBodySimulationOptions
): Promise<Float32Array[]> {
  const { numParticles, numSnapshots, deltaT, onProgress } = options;
  const snapshots: Float32Array[] = [];

  if (!gpuDevice) throw new Error('GPU device not initialized');

  console.log(`Starting GPU simulation: ${numParticles} particles, ${numSnapshots} snapshots`);
  onProgress?.(0, 'Initializing particles...');

  const sim = new NBodyGPU(gpuDevice, numParticles);
  await sim.init(deltaT);

  const saveInterval = 5;
  const transferInterval = 10;

  // Collect initial state
  const initialData = await sim.getParticleData();
  snapshots.push(convertGPUDataToCompact(initialData, numParticles));

  for (let step = 0; step < numSnapshots; step++) {
    sim.step(deltaT);

    if (step % saveInterval === 0) {
      // For the demo generation, we just read back everything.
      // Optimization: We could read back in chunks or use staging buffers more smartly,
      // but this reuses the NBodyGPU class which is cleaner.
      // However, reading back every frame is slow, so we only do it on saveInterval.

      const gpuData = await sim.getParticleData();

      // Check for NaNs (only check the first few to save time?)
      const x = gpuData[GPU_POS_X];
      if (isNaN(x)) {
         console.error(`NaN detected in GPU simulation at step ${step}`);
         onProgress?.(100, `Simulation stopped - NaN detected`);
         return snapshots;
      }

      snapshots.push(convertGPUDataToCompact(gpuData, numParticles));
    }

    if (step % transferInterval === 0) {
      const progress = ((step + 1) / numSnapshots) * 100;
      onProgress?.(
        progress,
        `Computing on GPU... ${progress.toFixed(1)}% (${step + 1}/${numSnapshots})`
      );
      // Allow UI to update
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  console.log(`GPU simulation collected ${snapshots.length} snapshots`);

  // Interpolate frames if we have enough snapshots
  if (snapshots.length < 2) {
    console.error(`Not enough snapshots for interpolation: ${snapshots.length}`);
    onProgress?.(100, `Simulation incomplete - only ${snapshots.length} snapshots`);
    return snapshots;
  }

  onProgress?.(95, 'Interpolating frames...');
  const fullSnapshots = interpolateSnapshots(snapshots, saveInterval, numParticles);

  console.log(`GPU simulation complete: ${fullSnapshots.length} total frames`);
  onProgress?.(100, 'Simulation complete!');
  return fullSnapshots;
}

function convertGPUDataToCompact(gpuData: Float32Array, numParticles: number): Float32Array {
  const snapshot = createParticleArray(numParticles);
  for (let i = 0; i < numParticles; i++) {
    const gpuIdx = i * GPU_FLOATS_PER_PARTICLE;
    const compactOffset = i * FLOATS_PER_PARTICLE;

    snapshot[compactOffset + OFFSET_X] = gpuData[gpuIdx + GPU_POS_X];
    snapshot[compactOffset + OFFSET_Y] = gpuData[gpuIdx + GPU_POS_Y];
    snapshot[compactOffset + OFFSET_Z] = gpuData[gpuIdx + GPU_POS_Z];
    snapshot[compactOffset + OFFSET_VX] = gpuData[gpuIdx + GPU_VEL_X];
    snapshot[compactOffset + OFFSET_VY] = gpuData[gpuIdx + GPU_VEL_Y];
    snapshot[compactOffset + OFFSET_VZ] = gpuData[gpuIdx + GPU_VEL_Z];
    snapshot[compactOffset + OFFSET_MASS] = gpuData[gpuIdx + GPU_MASS];
  }
  return snapshot;
}

function computeForcesCPU(particles: Float32Array, forces: Float32Array, numParticles: number) {
  const G = 1.0;
  const softening = 2.0;
  const softeningSq = softening * softening;

  forces.fill(0);

  for (let i = 0; i < numParticles; i++) {
    const iOffset = i * FLOATS_PER_PARTICLE;
    const ix = particles[iOffset + OFFSET_X];
    const iy = particles[iOffset + OFFSET_Y];
    const iz = particles[iOffset + OFFSET_Z];
    const im = particles[iOffset + OFFSET_MASS];
    const Gim = G * im;

    // Cache force indices for i
    const iFxIndex = i * 3;
    const iFyIndex = iFxIndex + 1;
    const iFzIndex = iFxIndex + 2;

    // Accumulate forces in local variables
    let fx_i = 0;
    let fy_i = 0;
    let fz_i = 0;

    // Pre-calculate starting offset for j
    let jOffset = (i + 1) * FLOATS_PER_PARTICLE;
    let jFIndex = (i + 1) * 3;

    for (let j = i + 1; j < numParticles; j++) {
      const jx = particles[jOffset + OFFSET_X];
      const jy = particles[jOffset + OFFSET_Y];
      const jz = particles[jOffset + OFFSET_Z];
      const jm = particles[jOffset + OFFSET_MASS];

      const dx = jx - ix;
      const dy = jy - iy;
      const dz = jz - iz;

      const r2 = dx * dx + dy * dy + dz * dz + softeningSq;
      const r = Math.sqrt(r2);
      const f = (Gim * jm) / (r2 * r);

      const fx = f * dx;
      const fy = f * dy;
      const fz = f * dz;

      // Accumulate for i (local vars)
      fx_i += fx;
      fy_i += fy;
      fz_i += fz;

      // Update for j (direct array access)
      forces[jFIndex] -= fx;
      forces[jFIndex + 1] -= fy;
      forces[jFIndex + 2] -= fz;

      // Increment offsets manually
      jOffset += FLOATS_PER_PARTICLE;
      jFIndex += 3;
    }

    // Write back accumulated forces for i
    forces[iFxIndex] += fx_i;
    forces[iFyIndex] += fy_i;
    forces[iFzIndex] += fz_i;
  }
}

async function generateNBodyCPU(
  options: NBodySimulationOptions
): Promise<Float32Array[]> {
  const { numParticles, numSnapshots, deltaT, onProgress } = options;
  const snapshots: Float32Array[] = [];

  // Initialize particles using TypedArray
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

  // Orbiting particles
  for (let i = 1; i < numParticles; i++) {
    const r = 20 + Math.random() * 60;
    const theta = Math.random() * Math.PI * 2;
    const z = (Math.random() - 0.5) * 5;

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);

    // Circular orbit velocity: v = sqrt(GM/r) where G=1.0, M=5000
    const v = Math.sqrt(5000 / r);
    const vx = -v * Math.sin(theta) + (Math.random() - 0.5) * 0.5;
    const vy = v * Math.cos(theta) + (Math.random() - 0.5) * 0.5;
    const vz = (Math.random() - 0.5) * 0.2;

    setParticle(particles, i, { x, y, z, vx, vy, vz, mass: 1 });
  }

  // Remove net momentum to prevent drift
  removeCenterOfMassVelocity(particles);

  const chunkSize = 50;

  // Pre-allocate forces array
  const forces = new Float32Array(numParticles * 3);

  // Compute initial forces (O(N²)) - Pre-loop optimization
  // Forces calculated here are used for the first kick
  computeForcesCPU(particles, forces, numParticles);

  for (let chunkStart = 0; chunkStart < numSnapshots; chunkStart += chunkSize) {
    const chunkEnd = Math.min(chunkStart + chunkSize, numSnapshots);

    for (let step = chunkStart; step < chunkEnd; step++) {
      // Clone snapshot
      snapshots.push(cloneParticleData(particles));

      // Optimization: Use forces calculated at the end of the previous step (or init)
      // We do NOT recalculate forces here.

      // Leapfrog integration (kick-drift-kick)
      // Optimization: Fuse Kick1 and Drift loops for better cache locality (1 pass instead of 2)
      for (let i = 0; i < numParticles; i++) {
        const offset = i * FLOATS_PER_PARTICLE;
        const mass = particles[offset + OFFSET_MASS];

        const ax = forces[i * 3 + 0] / mass;
        const ay = forces[i * 3 + 1] / mass;
        const az = forces[i * 3 + 2] / mass;

        // Kick 1
        particles[offset + OFFSET_VX] += ax * deltaT * 0.5;
        particles[offset + OFFSET_VY] += ay * deltaT * 0.5;
        particles[offset + OFFSET_VZ] += az * deltaT * 0.5;

        // Drift (uses updated velocity)
        particles[offset + OFFSET_X] += particles[offset + OFFSET_VX] * deltaT;
        particles[offset + OFFSET_Y] += particles[offset + OFFSET_VY] * deltaT;
        particles[offset + OFFSET_Z] += particles[offset + OFFSET_VZ] * deltaT;
      }

      // Recompute forces at new positions
      computeForcesCPU(particles, forces, numParticles);

      // Half-step velocity update (kick)
      for (let i = 0; i < numParticles; i++) {
        const offset = i * FLOATS_PER_PARTICLE;
        const mass = particles[offset + OFFSET_MASS];

        const ax = forces[i * 3 + 0] / mass;
        const ay = forces[i * 3 + 1] / mass;
        const az = forces[i * 3 + 2] / mass;

        particles[offset + OFFSET_VX] += ax * deltaT * 0.5;
        particles[offset + OFFSET_VY] += ay * deltaT * 0.5;
        particles[offset + OFFSET_VZ] += az * deltaT * 0.5;
      }

      // Correct momentum drift every 10 steps
      if (step % 10 === 0) {
        removeCenterOfMassVelocity(particles);
      }
    }

    const progress = (chunkEnd / numSnapshots) * 100;
    onProgress?.(progress, `${progress.toFixed(1)}% (${chunkEnd}/${numSnapshots} timesteps)`);

    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  return snapshots;
}

function interpolateSnapshots(
  snapshots: Float32Array[],
  saveInterval: number,
  numParticles: number
): Float32Array[] {
  const totalFrames = (snapshots.length - 1) * saveInterval + 1;
  const fullSnapshots: Float32Array[] = new Array(totalFrames);

  let frameIndex = 0;

  for (let i = 0; i < snapshots.length - 1; i++) {
    const snap1 = snapshots[i];
    const snap2 = snapshots[i + 1];

    fullSnapshots[frameIndex++] = snap1;

    for (let j = 1; j < saveInterval; j++) {
      const t = j / saveInterval;
      const oneMinusT = 1 - t;

      const interpolated = createParticleArray(numParticles);

      for (let p = 0; p < numParticles; p++) {
        const offset1 = p * FLOATS_PER_PARTICLE;
        const offset2 = p * FLOATS_PER_PARTICLE;

        // Interpolate each component
        interpolated[offset1 + OFFSET_X] = snap1[offset1 + OFFSET_X] * oneMinusT + snap2[offset2 + OFFSET_X] * t;
        interpolated[offset1 + OFFSET_Y] = snap1[offset1 + OFFSET_Y] * oneMinusT + snap2[offset2 + OFFSET_Y] * t;
        interpolated[offset1 + OFFSET_Z] = snap1[offset1 + OFFSET_Z] * oneMinusT + snap2[offset2 + OFFSET_Z] * t;
        interpolated[offset1 + OFFSET_VX] = snap1[offset1 + OFFSET_VX] * oneMinusT + snap2[offset2 + OFFSET_VX] * t;
        interpolated[offset1 + OFFSET_VY] = snap1[offset1 + OFFSET_VY] * oneMinusT + snap2[offset2 + OFFSET_VY] * t;
        interpolated[offset1 + OFFSET_VZ] = snap1[offset1 + OFFSET_VZ] * oneMinusT + snap2[offset2 + OFFSET_VZ] * t;
        interpolated[offset1 + OFFSET_MASS] = snap1[offset1 + OFFSET_MASS]; // Mass doesn't change
      }

      fullSnapshots[frameIndex++] = interpolated;
    }
  }

  fullSnapshots[frameIndex] = snapshots[snapshots.length - 1];

  return fullSnapshots;
}
