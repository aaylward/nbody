import { Particle } from '../types';

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
): Promise<Particle[][]> {
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
): Promise<Particle[][]> {
  const { numParticles, numSnapshots, deltaT, onProgress } = options;
  const snapshots: Particle[][] = [];

  if (!gpuDevice) throw new Error('GPU device not initialized');

  console.log(`Starting GPU simulation: ${numParticles} particles, ${numSnapshots} snapshots`);
  onProgress?.(0, 'Initializing particles...');

  // Initialize particles
  const particles: Particle[] = [];

  particles.push({
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    mass: 5000,
  });

  for (let i = 1; i < numParticles; i++) {
    const r = 20 + Math.random() * 60;
    const theta = Math.random() * Math.PI * 2;
    const z = (Math.random() - 0.5) * 5;

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);

    const v = Math.sqrt(5000 / r) * 0.8;
    const vx = -v * Math.sin(theta) + (Math.random() - 0.5) * 0.5;
    const vy = v * Math.cos(theta) + (Math.random() - 0.5) * 0.5;
    const vz = (Math.random() - 0.5) * 0.2;

    particles.push({ x, y, z, vx, vy, vz, mass: 1 });
  }

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

  const updateParticlesShader = `
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
    fn updateParticles(@builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i >= arrayLength(&particles)) { return; }

        let mass = particles[i].mass;
        let accel = forces[i] / mass;

        particles[i].vel += accel * uniforms.dt;
        particles[i].pos += particles[i].vel * uniforms.dt;
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

  // Float offsets for particle data layout
  const POS_X = 0;
  const POS_Y = 1;
  const POS_Z = 2;
  const POS_PAD = 3;
  const VEL_X = 4;
  const VEL_Y = 5;
  const VEL_Z = 6;
  const MASS = 7;
  const MASS_PAD = 8;
  const STRUCT_PAD_0 = 9;
  const STRUCT_PAD_1 = 10;
  const STRUCT_PAD_2 = 11;
  const FLOATS_PER_PARTICLE = 12;

  const particleData = new Float32Array(numParticles * FLOATS_PER_PARTICLE);
  for (let i = 0; i < numParticles; i++) {
    const offset = i * FLOATS_PER_PARTICLE;
    particleData[offset + POS_X] = particles[i].x;
    particleData[offset + POS_Y] = particles[i].y;
    particleData[offset + POS_Z] = particles[i].z;
    particleData[offset + POS_PAD] = 0;
    particleData[offset + VEL_X] = particles[i].vx;
    particleData[offset + VEL_Y] = particles[i].vy;
    particleData[offset + VEL_Z] = particles[i].vz;
    particleData[offset + MASS] = particles[i].mass ?? 1;
    particleData[offset + MASS_PAD] = 0;
    particleData[offset + STRUCT_PAD_0] = 0;
    particleData[offset + STRUCT_PAD_1] = 0;
    particleData[offset + STRUCT_PAD_2] = 0;
  }

  const particleBuffer = gpuDevice.createBuffer({
    size: particleData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });
  new Float32Array(particleBuffer.getMappedRange()).set(particleData);
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
    size: particleData.byteLength * 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const snapshotReadBuffer = gpuDevice.createBuffer({
    size: particleData.byteLength * 2,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Create compute pipelines
  const forceModule = gpuDevice.createShaderModule({ code: computeForceShader });
  const updateModule = gpuDevice.createShaderModule({ code: updateParticlesShader });

  const forcePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: forceModule, entryPoint: 'computeForces' },
  });

  const updatePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: updateModule, entryPoint: 'updateParticles' },
  });

  const forceBindGroup = gpuDevice.createBindGroup({
    layout: forcePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particleBuffer } },
      { binding: 1, resource: { buffer: forceBuffer } },
    ],
  });

  const updateBindGroup = gpuDevice.createBindGroup({
    layout: updatePipeline.getBindGroupLayout(0),
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
        particleData.byteLength * snapshotsPerTransfer
      );
      gpuDevice.queue.submit([commandEncoder.finish()]);

      await snapshotReadBuffer.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(snapshotReadBuffer.getMappedRange());

      for (let s = 0; s < snapshotsPerTransfer; s++) {
        const snapshot: Particle[] = [];
        const offset = s * numParticles * FLOATS_PER_PARTICLE;

        let hasNaN = false;
        for (let i = 0; i < numParticles; i++) {
          const idx = offset + i * FLOATS_PER_PARTICLE;
          const x = data[idx + POS_X];
          const y = data[idx + POS_Y];
          const z = data[idx + POS_Z];

          if (isNaN(x) || isNaN(y) || isNaN(z)) {
            console.error(`NaN detected in GPU transfer at snapshot ${snapshots.length}, particle ${i}: (${x}, ${y}, ${z})`);
            hasNaN = true;
            break;
          }

          snapshot.push({
            x,
            y,
            z,
            vx: data[idx + VEL_X],
            vy: data[idx + VEL_Y],
            vz: data[idx + VEL_Z],
          });
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
      const offset = stagingIndex * particleData.byteLength;
      commandEncoder.copyBufferToBuffer(
        particleBuffer,
        0,
        stagingBuffer,
        offset,
        particleData.byteLength
      );
      gpuDevice.queue.submit([commandEncoder.finish()]);
      stagingIndex++;
    }

    // Compute forces and update
    const commandEncoder = gpuDevice.createCommandEncoder();

    const forcePass = commandEncoder.beginComputePass();
    forcePass.setPipeline(forcePipeline);
    forcePass.setBindGroup(0, forceBindGroup);
    forcePass.dispatchWorkgroups(workgroupCount);
    forcePass.end();

    const updatePass = commandEncoder.beginComputePass();
    updatePass.setPipeline(updatePipeline);
    updatePass.setBindGroup(0, updateBindGroup);
    updatePass.dispatchWorkgroups(workgroupCount);
    updatePass.end();

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
      particleData.byteLength * stagingIndex
    );
    gpuDevice.queue.submit([commandEncoder.finish()]);

    await snapshotReadBuffer.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(snapshotReadBuffer.getMappedRange());

    for (let s = 0; s < stagingIndex; s++) {
      const snapshot: Particle[] = [];
      const offset = s * numParticles * FLOATS_PER_PARTICLE;

      for (let i = 0; i < numParticles; i++) {
        const idx = offset + i * FLOATS_PER_PARTICLE;
        snapshot.push({
          x: data[idx + POS_X],
          y: data[idx + POS_Y],
          z: data[idx + POS_Z],
          vx: data[idx + VEL_X],
          vy: data[idx + VEL_Y],
          vz: data[idx + VEL_Z],
        });
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
): Promise<Particle[][]> {
  const { numParticles, numSnapshots, deltaT, onProgress } = options;
  const snapshots: Particle[][] = [];

  const particles: Particle[] = [];

  particles.push({
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    mass: 5000,
  });

  for (let i = 1; i < numParticles; i++) {
    const r = 20 + Math.random() * 60;
    const theta = Math.random() * Math.PI * 2;
    const z = (Math.random() - 0.5) * 5;

    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);

    const v = Math.sqrt(5000 / r) * 0.8;
    const vx = -v * Math.sin(theta) + (Math.random() - 0.5) * 0.5;
    const vy = v * Math.cos(theta) + (Math.random() - 0.5) * 0.5;
    const vz = (Math.random() - 0.5) * 0.2;

    particles.push({ x, y, z, vx, vy, vz, mass: 1 });
  }

  const G = 1.0;
  const softening = 2.0;
  const chunkSize = 50;

  for (let chunkStart = 0; chunkStart < numSnapshots; chunkStart += chunkSize) {
    const chunkEnd = Math.min(chunkStart + chunkSize, numSnapshots);

    for (let step = chunkStart; step < chunkEnd; step++) {
      snapshots.push(JSON.parse(JSON.stringify(particles)));

      const forces = particles.map(() => ({ fx: 0, fy: 0, fz: 0 }));

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[j].x - particles[i].x;
          const dy = particles[j].y - particles[i].y;
          const dz = particles[j].z - particles[i].z;

          const r2 = dx * dx + dy * dy + dz * dz + softening * softening;
          const r = Math.sqrt(r2);
          const f = (G * (particles[i].mass ?? 1) * (particles[j].mass ?? 1)) / r2;

          const fx = (f * dx) / r;
          const fy = (f * dy) / r;
          const fz = (f * dz) / r;

          forces[i].fx += fx;
          forces[i].fy += fy;
          forces[i].fz += fz;

          forces[j].fx -= fx;
          forces[j].fy -= fy;
          forces[j].fz -= fz;
        }
      }

      for (let i = 0; i < particles.length; i++) {
        const ax = forces[i].fx / (particles[i].mass ?? 1);
        const ay = forces[i].fy / (particles[i].mass ?? 1);
        const az = forces[i].fz / (particles[i].mass ?? 1);

        particles[i].vx += ax * deltaT;
        particles[i].vy += ay * deltaT;
        particles[i].vz += az * deltaT;

        particles[i].x += particles[i].vx * deltaT;
        particles[i].y += particles[i].vy * deltaT;
        particles[i].z += particles[i].vz * deltaT;
      }
    }

    const progress = (chunkEnd / numSnapshots) * 100;
    onProgress?.(progress, `${progress.toFixed(1)}% (${chunkEnd}/${numSnapshots} timesteps)`);

    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  return snapshots;
}

function interpolateSnapshots(
  snapshots: Particle[][],
  saveInterval: number,
  numParticles: number
): Particle[][] {
  const totalFrames = (snapshots.length - 1) * saveInterval + 1;
  const fullSnapshots: Particle[][] = new Array(totalFrames);

  let frameIndex = 0;

  for (let i = 0; i < snapshots.length - 1; i++) {
    const snap1 = snapshots[i];
    const snap2 = snapshots[i + 1];

    fullSnapshots[frameIndex++] = snap1;

    for (let j = 1; j < saveInterval; j++) {
      const t = j / saveInterval;
      const oneMinusT = 1 - t;

      const interpolated: Particle[] = new Array(numParticles);

      for (let p = 0; p < numParticles; p++) {
        const p1 = snap1[p];
        const p2 = snap2[p];

        interpolated[p] = {
          x: p1.x * oneMinusT + p2.x * t,
          y: p1.y * oneMinusT + p2.y * t,
          z: p1.z * oneMinusT + p2.z * t,
          vx: p1.vx * oneMinusT + p2.vx * t,
          vy: p1.vy * oneMinusT + p2.vy * t,
          vz: p1.vz * oneMinusT + p2.vz * t,
        };
      }

      fullSnapshots[frameIndex++] = interpolated;
    }
  }

  fullSnapshots[frameIndex] = snapshots[snapshots.length - 1];

  return fullSnapshots;
}
