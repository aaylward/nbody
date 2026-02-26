import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NBodyGPU, initGPU, GPU_FLOATS_PER_PARTICLE, RENDER_FLOATS_PER_PARTICLE } from '../nbody';

// Mock WebGPU globals
const mockQueue = {
  submit: vi.fn(),
  writeBuffer: vi.fn(),
};

const mockComputePass = {
  setPipeline: vi.fn(),
  setBindGroup: vi.fn(),
  dispatchWorkgroups: vi.fn(),
  end: vi.fn(),
};

const mockCommandEncoder = {
  beginComputePass: vi.fn(() => mockComputePass),
  copyBufferToBuffer: vi.fn(),
  finish: vi.fn(),
};

const mockBuffer = {
  getMappedRange: vi.fn(() => new Float32Array(20000).buffer), // Enough for particles
  unmap: vi.fn(),
  mapAsync: vi.fn().mockResolvedValue(undefined),
  size: 20000,
};

const mockDevice = {
  createBuffer: vi.fn(() => mockBuffer),
  createShaderModule: vi.fn(),
  createComputePipeline: vi.fn(() => ({
    getBindGroupLayout: vi.fn(),
  })),
  createBindGroup: vi.fn(),
  createCommandEncoder: vi.fn(() => mockCommandEncoder),
  queue: mockQueue,
};

const mockGPUDevice = mockDevice as unknown as GPUDevice;

// Polyfill globals needed by nbody.ts
global.navigator = {
  gpu: {
    requestAdapter: vi.fn().mockResolvedValue({
      requestDevice: vi.fn().mockResolvedValue(mockDevice),
    }),
  },
// eslint-disable-next-line @typescript-eslint/no-explicit-any
} as any;

// @ts-expect-error - Partial mock
global.GPUBufferUsage = {
  STORAGE: 1,
  COPY_DST: 2,
  COPY_SRC: 4,
  UNIFORM: 8,
  MAP_READ: 16,
};

// @ts-expect-error - Partial mock
global.GPUMapMode = {
  READ: 1,
};

describe('NBodyGPU', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should initialize successfully', async () => {
    await initGPU();
    const numParticles = 100;
    const sim = new NBodyGPU(mockGPUDevice, numParticles);
    await sim.init(0.01);

    // Verify buffer creation
    expect(mockDevice.createBuffer).toHaveBeenCalledTimes(5); // particle, force, uniform, compact, staging

    // Verify pipeline creation (3 pipelines)
    expect(mockDevice.createComputePipeline).toHaveBeenCalledTimes(3);

    // Verify bind group creation (3 bind groups)
    expect(mockDevice.createBindGroup).toHaveBeenCalledTimes(3);
  });

  it('should step simulation correctly (4 passes)', async () => {
    const numParticles = 100;
    const sim = new NBodyGPU(mockGPUDevice, numParticles);
    await sim.init(0.01);

    sim.step(0.01);

    expect(mockDevice.createCommandEncoder).toHaveBeenCalled();
    // 4 compute passes: force, kickDrift, force, kick
    expect(mockCommandEncoder.beginComputePass).toHaveBeenCalledTimes(4);
    expect(mockComputePass.dispatchWorkgroups).toHaveBeenCalledTimes(4);
    expect(mockQueue.submit).toHaveBeenCalled();
  });

  it('should read back particle data', async () => {
    const numParticles = 100;
    const sim = new NBodyGPU(mockGPUDevice, numParticles);
    await sim.init(0.01);

    await sim.getParticleData();

    expect(mockCommandEncoder.copyBufferToBuffer).toHaveBeenCalled();
    expect(mockBuffer.mapAsync).toHaveBeenCalled();
    expect(mockBuffer.getMappedRange).toHaveBeenCalled();
    expect(mockBuffer.unmap).toHaveBeenCalled();
  });

  it('should read back particle data into provided buffer', async () => {
    const numParticles = 100;
    const sim = new NBodyGPU(mockGPUDevice, numParticles);
    await sim.init(0.01);

    const expectedSize = numParticles * RENDER_FLOATS_PER_PARTICLE;
    // Mock getMappedRange to return exact size buffer for this test
    mockBuffer.getMappedRange.mockReturnValueOnce(new Float32Array(expectedSize).buffer);

    const outData = new Float32Array(expectedSize);
    const result = await sim.getParticleData(outData);

    expect(result).toBe(outData); // Should return the same buffer
    expect(mockCommandEncoder.copyBufferToBuffer).toHaveBeenCalled();
    expect(mockBuffer.mapAsync).toHaveBeenCalled();
    expect(mockBuffer.getMappedRange).toHaveBeenCalled();
    expect(mockBuffer.unmap).toHaveBeenCalled();
  });

  it('should handle large number of particles (1M)', async () => {
    const numParticles = 1000000;
    const sim = new NBodyGPU(mockGPUDevice, numParticles);

    // Mock getMappedRange to return enough buffer for 1M particles
    // 1M * 8 floats * 4 bytes = 32MB
    const largeBuffer = new Float32Array(numParticles * GPU_FLOATS_PER_PARTICLE).buffer;
    mockBuffer.getMappedRange.mockReturnValue(largeBuffer);

    await sim.init(0.01);

    expect(mockDevice.createBuffer).toHaveBeenCalled();
    // Should calculate correct workgroup count: ceil(1000000 / 256) = 3907
    expect(sim.workgroupCount).toBe(Math.ceil(1000000 / 256));

    sim.step(0.01);
    expect(mockComputePass.dispatchWorkgroups).toHaveBeenCalledWith(3907);
  });
});
