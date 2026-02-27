import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NBodyGPU } from '../nbody';

// Mock WebGPU interfaces
const createMockDevice = () => {
  return {
    createBuffer: vi.fn(() => ({
      getMappedRange: vi.fn(() => {
        // Return a new ArrayBuffer each time so it can be viewed as typed array
        // and doesn't conflict
        return new ArrayBuffer(256 * 8 * 4); // 256 particles * 8 floats * 4 bytes
      }),
      unmap: vi.fn(),
      mapAsync: vi.fn(),
      size: 256 * 8 * 4,
    })),
    createShaderModule: vi.fn(),
    createComputePipeline: vi.fn(() => ({
      getBindGroupLayout: vi.fn((index) => `layout-${index}`),
    })),
    createBindGroup: vi.fn(),
    createCommandEncoder: vi.fn(() => ({
      beginComputePass: vi.fn(() => ({
        setPipeline: vi.fn(),
        setBindGroup: vi.fn(),
        dispatchWorkgroups: vi.fn(),
        end: vi.fn(),
      })),
      finish: vi.fn(),
      copyBufferToBuffer: vi.fn(),
    })),
    queue: {
      submit: vi.fn(),
      writeBuffer: vi.fn(),
    },
  } as unknown as GPUDevice;
};

describe('NBodyGPU BindGroup Configuration', () => {
  let mockDevice: GPUDevice;

  beforeEach(() => {
    mockDevice = createMockDevice();
    // Mock global GPUBufferUsage
    global.GPUBufferUsage = {
      STORAGE: 1,
      COPY_DST: 2,
      COPY_SRC: 4,
      UNIFORM: 8,
      MAP_READ: 16,
      MAP_WRITE: 32,
    } as any;

    // Mock global GPUMapMode
    global.GPUMapMode = {
      READ: 1,
      WRITE: 2,
    } as any;
  });

  it('should create kickDriftBindGroup with correct bindings (0, 1, 2)', async () => {
    const numParticles = 256;
    const sim = new NBodyGPU(mockDevice, numParticles);

    // We need to spy on createBindGroup to verify arguments
    const createBindGroupSpy = vi.spyOn(mockDevice, 'createBindGroup');

    await sim.init(0.01);

    // Find the call for kickDriftBindGroup
    const bindGroupCalls = createBindGroupSpy.mock.calls;

    // We expect 3 bind groups: force, kickDrift, kick.
    expect(bindGroupCalls.length).toBe(3);

    // In nbody.ts, the order of creation is:
    // 1. forceBindGroup
    // 2. kickDriftBindGroup
    // 3. kickBindGroup

    const kickDriftCall = bindGroupCalls[1][0] as GPUBindGroupDescriptor;
    const kickCall = bindGroupCalls[2][0] as GPUBindGroupDescriptor;

    // Asserting the EXPECTED state (Fix applied):
    // kickDrift should have 3 entries (0, 1, 2).
    // kick should have 4 entries (0, 1, 2, 3).

    // With current BUGgy code:
    // kickDrift has 4 entries.
    // kick has 3 entries.

    // This test confirms the bug by checking for the WRONG state first if we wanted to be strict,
    // but the goal is to make it PASS after the fix.

    // To confirm it fails NOW (before fix):

    // Check kickDriftBindGroup: Should NOT have binding 3
    const kickDriftHasBinding3 = kickDriftCall.entries.some(e => e.binding === 3);
    // Expect this to be false in correct code, but it is true in buggy code.
    expect(kickDriftHasBinding3).toBe(false);

    // Check kickBindGroup: SHOULD have binding 3
    const kickHasBinding3 = kickCall.entries.some(e => e.binding === 3);
    // Expect this to be true in correct code, but it is false in buggy code.
    expect(kickHasBinding3).toBe(true);
  });
});
