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

  // WebGPU shaders
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

  // Leapfrog integration - first kick + drift (combined for efficiency)
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

  // Leapfrog integration - second kick (half-step velocity update)
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

  onProgress?.(5, 'Setting up GPU buffers...');

  // Create GPU buffers
  // Particle struct is 48 bytes (12 floats) due to alignment:
  // struct Particle {
  //   pos: vec3f,   // offset 0-11 (bytes), padded to offset 16
  //   vel: vec3f,   // offset 16-27 (bytes)
  //   mass: f32,    // offset 28-31 (bytes) ← immediately after vel!
  //   _pad: f32,    // offset 32-35 (bytes)
  // }                // total 36 bytes, padded to 48 for 16-byte struct alignment

  // GPU buffer layout (12 floats per particle with padding)
  const GPU_FLOATS_PER_PARTICLE = 12;
  const GPU_POS_X = 0;
  const GPU_POS_Y = 1;
  const GPU_POS_Z = 2;
  const GPU_POS_PAD = 3;
  const GPU_VEL_X = 4;
  const GPU_VEL_Y = 5;
  const GPU_VEL_Z = 6;
  const GPU_MASS = 7;
  const GPU_MASS_PAD = 8;
  const GPU_STRUCT_PAD_0 = 9;
  const GPU_STRUCT_PAD_1 = 10;
  const GPU_STRUCT_PAD_2 = 11;

  // Convert from our compact format (7 floats) to GPU format (12 floats)
  const gpuParticleData = new Float32Array(numParticles * GPU_FLOATS_PER_PARTICLE);
  for (let i = 0; i < numParticles; i++) {
    const srcOffset = i * FLOATS_PER_PARTICLE;
    const dstOffset = i * GPU_FLOATS_PER_PARTICLE;

    gpuParticleData[dstOffset + GPU_POS_X] = particles[srcOffset + OFFSET_X];
    gpuParticleData[dstOffset + GPU_POS_Y] = particles[srcOffset + OFFSET_Y];
    gpuParticleData[dstOffset + GPU_POS_Z] = particles[srcOffset + OFFSET_Z];
    gpuParticleData[dstOffset + GPU_POS_PAD] = 0;
    gpuParticleData[dstOffset + GPU_VEL_X] = particles[srcOffset + OFFSET_VX];
    gpuParticleData[dstOffset + GPU_VEL_Y] = particles[srcOffset + OFFSET_VY];
    gpuParticleData[dstOffset + GPU_VEL_Z] = particles[srcOffset + OFFSET_VZ];
    gpuParticleData[dstOffset + GPU_MASS] = particles[srcOffset + OFFSET_MASS];
    gpuParticleData[dstOffset + GPU_MASS_PAD] = 0;
    gpuParticleData[dstOffset + GPU_STRUCT_PAD_0] = 0;
    gpuParticleData[dstOffset + GPU_STRUCT_PAD_1] = 0;
    gpuParticleData[dstOffset + GPU_STRUCT_PAD_2] = 0;
  }

  const particleBuffer = gpuDevice.createBuffer({
    size: gpuParticleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Float32Array(particleBuffer.getMappedRange()).set(gpuParticleData);
  particleBuffer.unmap();

  const forceBuffer = gpuDevice.createBuffer({
    size: numParticles * 4 * 4, // vec3f in storage requires 16-byte alignment (4 floats)
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const uniformBuffer = gpuDevice.createBuffer({
    size: 16, // Uniform buffers require 16-byte minimum alignment
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(uniformBuffer.getMappedRange()).set([deltaT, 0, 0, 0]);
  uniformBuffer.unmap();

  const stagingBuffer = gpuDevice.createBuffer({
    size: gpuParticleData.byteLength * 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const snapshotReadBuffer = gpuDevice.createBuffer({
    size: gpuParticleData.byteLength * 2,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Create compute pipelines
  const forceModule = gpuDevice.createShaderModule({ code: computeForceShader });
  const kickDriftModule = gpuDevice.createShaderModule({ code: kickDriftShader });
  const kickModule = gpuDevice.createShaderModule({ code: kickShader });

  const forcePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: forceModule, entryPoint: 'computeForces' },
  });

  const kickDriftPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: kickDriftModule, entryPoint: 'kickDrift' },
  });

  const kickPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: kickModule, entryPoint: 'kick' },
  });

  const forceBindGroup = gpuDevice.createBindGroup({
    layout: forcePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: forceBuffer } },
    ],
  });

  const kickDriftBindGroup = gpuDevice.createBindGroup({
    layout: kickDriftPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: forceBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });

  const kickBindGroup = gpuDevice.createBindGroup({
    layout: kickPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: forceBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });

  const workgroupCount = Math.ceil(numParticles / 256);

  const saveInterval = 5;
  const transferInterval = 10;
  const snapshotsPerTransfer = transferInterval / saveInterval;

  let stagingIndex = 0;

  for (let step = 0; step < numSnapshots; step++) {
    // Transfer to CPU if staging buffer is full
    if (step > 0 && step % transferInterval === 0) {
      const commandEncoder = gpuDevice.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(
        stagingBuffer,
        0,
        snapshotReadBuffer,
        0,
        gpuParticleData.byteLength * snapshotsPerTransfer
      );
      gpuDevice.queue.submit([commandEncoder.finish()]);

      await snapshotReadBuffer.mapAsync(GPUMapMode.READ);
      const gpuData = new Float32Array(snapshotReadBuffer.getMappedRange());

      for (let s = 0; s < snapshotsPerTransfer; s++) {
        const snapshot = createParticleArray(numParticles);
        const gpuOffset = s * numParticles * GPU_FLOATS_PER_PARTICLE;

        let hasNaN = false;
        for (let i = 0; i < numParticles; i++) {
          const gpuIdx = gpuOffset + i * GPU_FLOATS_PER_PARTICLE;
          const x = gpuData[gpuIdx + GPU_POS_X];
          const y = gpuData[gpuIdx + GPU_POS_Y];
          const z = gpuData[gpuIdx + GPU_POS_Z];

          if (isNaN(x) || isNaN(y) || isNaN(z)) {
            console.error(`NaN detected in GPU transfer at snapshot ${snapshots.length}, particle ${i}: (${x}, ${y}, ${z})`);
            hasNaN = true;
            break;
          }

          // Convert from GPU format (12 floats) to compact format (7 floats)
          const compactOffset = i * FLOATS_PER_PARTICLE;
          snapshot[compactOffset + OFFSET_X] = x;
          snapshot[compactOffset + OFFSET_Y] = y;
          snapshot[compactOffset + OFFSET_Z] = z;
          snapshot[compactOffset + OFFSET_VX] = gpuData[gpuIdx + GPU_VEL_X];
          snapshot[compactOffset + OFFSET_VY] = gpuData[gpuIdx + GPU_VEL_Y];
          snapshot[compactOffset + OFFSET_VZ] = gpuData[gpuIdx + GPU_VEL_Z];
          snapshot[compactOffset + OFFSET_MASS] = gpuData[gpuIdx + GPU_MASS];
        }

        if (hasNaN) {
          console.error(`Stopping GPU simulation due to NaN at ${snapshots.length} snapshots`);
          onProgress?.(100, `Simulation stopped - NaN detected`);
          return snapshots;
        }

        snapshots.push(snapshot);
      }

      snapshotReadBuffer.unmap();
      stagingIndex = 0;

      const progress = ((step + 1) / numSnapshots) * 100;
      onProgress?.(
        progress,
        `Computing on GPU... ${progress.toFixed(1)}% (${step + 1}/${numSnapshots})`
      );

      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    // Save snapshot to staging buffer
    if (step % saveInterval === 0) {
      const commandEncoder = gpuDevice.createCommandEncoder();
      const offset = stagingIndex * gpuParticleData.byteLength;
      commandEncoder.copyBufferToBuffer(
        particleBuffer,
        0,
        stagingBuffer,
        offset,
        gpuParticleData.byteLength
      );
      gpuDevice.queue.submit([commandEncoder.finish()]);
      stagingIndex++;
    }

    // Leapfrog integration: 4-pass optimized kick-drift-kick
    const commandEncoder = gpuDevice.createCommandEncoder();

    // 1. Compute forces at current positions
    const forcePass1 = commandEncoder.beginComputePass();
    forcePass1.setPipeline(forcePipeline);
    forcePass1.setBindGroup(0, forceBindGroup);
    forcePass1.dispatchWorkgroups(workgroupCount);
    forcePass1.end();

    // 2. First kick + drift (combined for efficiency)
    const kickDriftPass = commandEncoder.beginComputePass();
    kickDriftPass.setPipeline(kickDriftPipeline);
    kickDriftPass.setBindGroup(0, kickDriftBindGroup);
    kickDriftPass.dispatchWorkgroups(workgroupCount);
    kickDriftPass.end();

    // 3. Recompute forces at new positions
    const forcePass2 = commandEncoder.beginComputePass();
    forcePass2.setPipeline(forcePipeline);
    forcePass2.setBindGroup(0, forceBindGroup);
    forcePass2.dispatchWorkgroups(workgroupCount);
    forcePass2.end();

    // 4. Second kick: half-step velocity update
    const kickPass = commandEncoder.beginComputePass();
    kickPass.setPipeline(kickPipeline);
    kickPass.setBindGroup(0, kickBindGroup);
    kickPass.dispatchWorkgroups(workgroupCount);
    kickPass.end();

    gpuDevice.queue.submit([commandEncoder.finish()]);
  }

  // Handle remaining snapshots
  if (stagingIndex > 0) {
    const commandEncoder = gpuDevice.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(
      stagingBuffer,
      0,
      snapshotReadBuffer,
      0,
      gpuParticleData.byteLength * stagingIndex
    );
    gpuDevice.queue.submit([commandEncoder.finish()]);

    await snapshotReadBuffer.mapAsync(GPUMapMode.READ);
    const gpuData = new Float32Array(snapshotReadBuffer.getMappedRange());

    for (let s = 0; s < stagingIndex; s++) {
      const snapshot = createParticleArray(numParticles);
      const gpuOffset = s * numParticles * GPU_FLOATS_PER_PARTICLE;

      for (let i = 0; i < numParticles; i++) {
        const gpuIdx = gpuOffset + i * GPU_FLOATS_PER_PARTICLE;
        const compactOffset = i * FLOATS_PER_PARTICLE;

        // Convert from GPU format (12 floats) to compact format (7 floats)
        snapshot[compactOffset + OFFSET_X] = gpuData[gpuIdx + GPU_POS_X];
        snapshot[compactOffset + OFFSET_Y] = gpuData[gpuIdx + GPU_POS_Y];
        snapshot[compactOffset + OFFSET_Z] = gpuData[gpuIdx + GPU_POS_Z];
        snapshot[compactOffset + OFFSET_VX] = gpuData[gpuIdx + GPU_VEL_X];
        snapshot[compactOffset + OFFSET_VY] = gpuData[gpuIdx + GPU_VEL_Y];
        snapshot[compactOffset + OFFSET_VZ] = gpuData[gpuIdx + GPU_VEL_Z];
        snapshot[compactOffset + OFFSET_MASS] = gpuData[gpuIdx + GPU_MASS];
      }
      snapshots.push(snapshot);
    }

    snapshotReadBuffer.unmap();
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

  const G = 1.0;
  const softening = 2.0;
  const chunkSize = 50;

  // Pre-allocate forces array
  const forces = new Float32Array(numParticles * 3);

  for (let chunkStart = 0; chunkStart < numSnapshots; chunkStart += chunkSize) {
    const chunkEnd = Math.min(chunkStart + chunkSize, numSnapshots);

    for (let step = chunkStart; step < chunkEnd; step++) {
      // Clone snapshot
      snapshots.push(cloneParticleData(particles));

      // Reset forces
      forces.fill(0);

      // Compute forces (O(N²) all-pairs)
      for (let i = 0; i < numParticles; i++) {
        const iOffset = i * FLOATS_PER_PARTICLE;
        const ix = particles[iOffset + OFFSET_X];
        const iy = particles[iOffset + OFFSET_Y];
        const iz = particles[iOffset + OFFSET_Z];
        const im = particles[iOffset + OFFSET_MASS];

        for (let j = i + 1; j < numParticles; j++) {
          const jOffset = j * FLOATS_PER_PARTICLE;
          const jx = particles[jOffset + OFFSET_X];
          const jy = particles[jOffset + OFFSET_Y];
          const jz = particles[jOffset + OFFSET_Z];
          const jm = particles[jOffset + OFFSET_MASS];

          const dx = jx - ix;
          const dy = jy - iy;
          const dz = jz - iz;

          const r2 = dx * dx + dy * dy + dz * dz + softening * softening;
          const r = Math.sqrt(r2);
          const f = (G * im * jm) / r2;

          const fx = (f * dx) / r;
          const fy = (f * dy) / r;
          const fz = (f * dz) / r;

          forces[i * 3 + 0] += fx;
          forces[i * 3 + 1] += fy;
          forces[i * 3 + 2] += fz;

          forces[j * 3 + 0] -= fx;
          forces[j * 3 + 1] -= fy;
          forces[j * 3 + 2] -= fz;
        }
      }

      // Leapfrog integration (kick-drift-kick)
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

      // Full-step position update (drift)
      for (let i = 0; i < numParticles; i++) {
        const offset = i * FLOATS_PER_PARTICLE;

        particles[offset + OFFSET_X] += particles[offset + OFFSET_VX] * deltaT;
        particles[offset + OFFSET_Y] += particles[offset + OFFSET_VY] * deltaT;
        particles[offset + OFFSET_Z] += particles[offset + OFFSET_VZ] * deltaT;
      }

      // Recompute forces at new positions
      forces.fill(0);
      for (let i = 0; i < numParticles; i++) {
        const iOffset = i * FLOATS_PER_PARTICLE;
        const ix = particles[iOffset + OFFSET_X];
        const iy = particles[iOffset + OFFSET_Y];
        const iz = particles[iOffset + OFFSET_Z];
        const im = particles[iOffset + OFFSET_MASS];

        for (let j = i + 1; j < numParticles; j++) {
          const jOffset = j * FLOATS_PER_PARTICLE;
          const jx = particles[jOffset + OFFSET_X];
          const jy = particles[jOffset + OFFSET_Y];
          const jz = particles[jOffset + OFFSET_Z];
          const jm = particles[jOffset + OFFSET_MASS];

          const dx = jx - ix;
          const dy = jy - iy;
          const dz = jz - iz;

          const r2 = dx * dx + dy * dy + dz * dz + softening * softening;
          const r = Math.sqrt(r2);
          const f = (G * im * jm) / r2;

          const fx = (f * dx) / r;
          const fy = (f * dy) / r;
          const fz = (f * dz) / r;

          forces[i * 3 + 0] += fx;
          forces[i * 3 + 1] += fy;
          forces[i * 3 + 2] += fz;

          forces[j * 3 + 0] -= fx;
          forces[j * 3 + 1] -= fy;
          forces[j * 3 + 2] -= fz;
        }
      }

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
