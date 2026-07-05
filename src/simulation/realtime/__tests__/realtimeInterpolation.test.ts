/**
 * Tests for GPU render interpolation in the Barnes-Hut backend.
 *
 * The physics loop runs at ~20 FPS, but rendering samples positions at
 * display rate (~60 FPS). To keep motion smooth, each physics step
 * snapshots pre-integration positions into prevPositionBuffer, and
 * getRenderPositionBuffer() dispatches a compute pass that blends
 * prev → current by alpha = progress toward the next physics frame.
 *
 * These tests protect:
 *   - prev buffer seeding (first frames blend two identical states)
 *   - the snapshot copy ordering (after forces, before integrate)
 *   - the interpolation dispatch contract (alpha in [0,1], bind group
 *     wiring, workgroup count, returned buffer)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// The real worker import ('?worker') can't be constructed under jsdom.
vi.mock('../octreeWorker?worker', () => ({
  default: class MockWorker {
    onmessage: ((e: MessageEvent) => void) | null = null;
    postMessage(): void {}
    terminate(): void {}
  },
}));

import { RealtimeNBodySimulationGPUBarnesHut } from '../RealtimeSimulationGPUBarnesHut';

interface MockBuffer {
  label: number;
  size: number;
  usage: number;
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
  mapAsync: () => Promise<void>;
  destroy: () => void;
}

type EncoderOp =
  | { kind: 'computePass'; pipeline: unknown; workgroups: number }
  | { kind: 'copy'; src: MockBuffer; dst: MockBuffer; size: number };

interface MockDeviceHarness {
  device: GPUDevice;
  buffers: MockBuffer[];
  encoders: EncoderOp[][];
  writeBufferCalls: Array<{ buffer: MockBuffer; data: ArrayBufferView | ArrayBuffer }>;
  bindGroupCalls: GPUBindGroupDescriptor[];
  pipelines: Array<{ label: string }>;
}

function createMockDevice(): MockDeviceHarness {
  const buffers: MockBuffer[] = [];
  const encoders: EncoderOp[][] = [];
  const writeBufferCalls: MockDeviceHarness['writeBufferCalls'] = [];
  const bindGroupCalls: GPUBindGroupDescriptor[] = [];
  const pipelines: Array<{ label: string }> = [];

  const device = {
    createBuffer: vi.fn((desc: { size: number; usage: number }) => {
      const buf: MockBuffer = {
        label: buffers.length,
        size: desc.size,
        usage: desc.usage,
        getMappedRange: () => new ArrayBuffer(desc.size),
        unmap: () => {},
        mapAsync: () => Promise.resolve(),
        destroy: () => {},
      };
      buffers.push(buf);
      return buf;
    }),
    createShaderModule: vi.fn((desc: { label?: string }) => ({ label: desc.label })),
    createComputePipeline: vi.fn((desc: { label?: string }) => {
      const pipeline = {
        label: desc.label ?? '',
        getBindGroupLayout: vi.fn((index: number) => `layout-${desc.label}-${index}`),
      };
      pipelines.push(pipeline);
      return pipeline;
    }),
    createBindGroup: vi.fn((desc: GPUBindGroupDescriptor) => {
      bindGroupCalls.push(desc);
      return { layout: desc.layout };
    }),
    createCommandEncoder: vi.fn(() => {
      const ops: EncoderOp[] = [];
      encoders.push(ops);
      return {
        beginComputePass: vi.fn(() => {
          const op: EncoderOp = { kind: 'computePass', pipeline: null, workgroups: 0 };
          ops.push(op);
          return {
            setPipeline: vi.fn((p: unknown) => { op.pipeline = p; }),
            setBindGroup: vi.fn(),
            dispatchWorkgroups: vi.fn((n: number) => { op.workgroups = n; }),
            end: vi.fn(),
          };
        }),
        copyBufferToBuffer: vi.fn((src: MockBuffer, _so: number, dst: MockBuffer, _do: number, size: number) => {
          ops.push({ kind: 'copy', src, dst, size });
        }),
        finish: vi.fn(),
      };
    }),
    queue: {
      submit: vi.fn(),
      writeBuffer: vi.fn((buffer: MockBuffer, _offset: number, data: ArrayBufferView | ArrayBuffer) => {
        writeBufferCalls.push({ buffer, data });
      }),
      onSubmittedWorkDone: vi.fn(() => Promise.resolve()),
    },
  } as unknown as GPUDevice;

  return { device, buffers, encoders, writeBufferCalls, bindGroupCalls, pipelines };
}

const NUM_PARTICLES = 512;

describe('Barnes-Hut render interpolation', () => {
  let harness: MockDeviceHarness;

  beforeEach(() => {
    harness = createMockDevice();
    global.GPUBufferUsage = {
      STORAGE: 1,
      COPY_DST: 2,
      COPY_SRC: 4,
      UNIFORM: 8,
      MAP_READ: 16,
      MAP_WRITE: 32,
    } as unknown as typeof GPUBufferUsage;
    global.GPUMapMode = { READ: 1, WRITE: 2 } as unknown as typeof GPUMapMode;
  });

  function createSim() {
    return new RealtimeNBodySimulationGPUBarnesHut({
      device: harness.device,
      numParticles: NUM_PARTICLES,
    });
  }

  it('seeds the previous-state buffer with the same initial data as the particle buffer', () => {
    createSim();

    // uploadParticlesToGPU writes the packed particle data to the particle
    // buffer and then to the prev-position buffer, so the first interpolation
    // blends two identical states.
    const [first, second] = harness.writeBufferCalls;
    expect(first.data).toBe(second.data);
    expect(first.buffer).not.toBe(second.buffer);
    expect(second.buffer.size).toBe(NUM_PARTICLES * 4 * 4);
  });

  it('dispatches the interpolation pass and returns the render buffer', () => {
    const sim = createSim();
    const encodersBefore = harness.encoders.length;

    const renderBuffer = sim.getRenderPositionBuffer() as unknown as MockBuffer;

    // A new encoder ran exactly one compute pass on the interpolation pipeline.
    expect(harness.encoders.length).toBe(encodersBefore + 1);
    const ops = harness.encoders[harness.encoders.length - 1];
    expect(ops).toHaveLength(1);
    expect(ops[0].kind).toBe('computePass');
    const pass = ops[0] as Extract<EncoderOp, { kind: 'computePass' }>;
    expect((pass.pipeline as { label: string }).label).toBe('Render Interpolation Pipeline');
    expect(pass.workgroups).toBe(Math.ceil(NUM_PARTICLES / 256));

    // The bind group wires prev (0), current (1), output (2), uniforms (3),
    // and the returned buffer is the interpolation output.
    const bindGroup = harness.bindGroupCalls[harness.bindGroupCalls.length - 1];
    const entries = Array.from(bindGroup.entries) as unknown as Array<{ binding: number; resource: { buffer: MockBuffer } }>;
    expect(entries.map((e) => e.binding).sort()).toEqual([0, 1, 2, 3]);
    const output = entries.find((e) => e.binding === 2)!.resource.buffer;
    expect(renderBuffer).toBe(output);

    // Alpha uniform was written and is a valid blend factor.
    const alphaWrite = harness.writeBufferCalls[harness.writeBufferCalls.length - 1];
    const alpha = new Float32Array(
      (alphaWrite.data as Float32Array).buffer ?? (alphaWrite.data as ArrayBuffer)
    )[0];
    expect(alpha).toBeGreaterThanOrEqual(0);
    expect(alpha).toBeLessThanOrEqual(1);
  });

  it('reuses the interpolation bind group across render frames', () => {
    const sim = createSim();

    sim.getRenderPositionBuffer();
    const bindGroupsAfterFirst = harness.bindGroupCalls.length;
    sim.getRenderPositionBuffer();
    sim.getRenderPositionBuffer();

    expect(harness.bindGroupCalls.length).toBe(bindGroupsAfterFirst);
  });

  it('snapshots pre-integration positions between the forces and integrate passes', async () => {
    const sim = createSim();
    sim.setTargetPhysicsFPS(60);

    await sim.start();
    await new Promise((resolve) => setTimeout(resolve, 25));
    sim.stop();
    await new Promise((resolve) => setTimeout(resolve, 25));

    // Find a physics-step encoder: forces pass, snapshot copy, integrate pass.
    const physicsEncoders = harness.encoders.filter(
      (ops) => ops.length === 3 && ops[0].kind === 'computePass' && ops[2].kind === 'computePass'
    );
    expect(physicsEncoders.length).toBeGreaterThan(0);

    for (const ops of physicsEncoders) {
      const copy = ops[1];
      expect(copy.kind).toBe('copy');
      const { src, dst, size } = copy as Extract<EncoderOp, { kind: 'copy' }>;
      // Source is the live particle buffer (also bound as "current" in the
      // interpolation bind group); destination is the prev-state snapshot.
      expect(size).toBe(NUM_PARTICLES * 4 * 4);
      expect(src).not.toBe(dst);
      expect(dst.usage & GPUBufferUsage.COPY_DST).toBeTruthy();

      // The snapshot copy must land in prevPositionBuffer, which is the
      // buffer the constructor seeded second with the initial particle data.
      expect(dst).toBe(harness.writeBufferCalls[1].buffer);
      expect(src).toBe(harness.writeBufferCalls[0].buffer);
    }
  });
});
