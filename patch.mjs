import fs from 'fs';

const filePath = 'src/simulation/realtime/RealtimeSimulation.ts';
let code = fs.readFileSync(filePath, 'utf8');

// Replace scalar buffers with an array
code = code.replace(
  /private particleBufferCurrent!: GPUBuffer;\n\s*private particleBufferNext!: GPUBuffer;/,
  'private particleBuffers!: [GPUBuffer, GPUBuffer];\n  private currentBufferIndex = 0;'
);

// Replace scalar bind groups with arrays
code = code.replace(
  /private forceBindGroup!: GPUBindGroup;\n\s*private kickDriftBindGroup!: GPUBindGroup;\n\s*private kickBindGroup!: GPUBindGroup;\n\s*private interpolateBindGroup!: GPUBindGroup;/,
  'private forceBindGroups!: [GPUBindGroup, GPUBindGroup];\n  private kickDriftBindGroups!: [GPUBindGroup, GPUBindGroup];\n  private kickBindGroups!: [GPUBindGroup, GPUBindGroup];\n  private interpolateBindGroups!: [GPUBindGroup, GPUBindGroup];'
);

// Update init logic
code = code.replace(
  /this\.particleBufferCurrent = this\.device\.createBuffer\(\{[\s\S]*?\}\);\n\s*new Float32Array\(this\.particleBufferCurrent\.getMappedRange\(\)\)\.set\(gpuParticleData\);\n\s*this\.particleBufferCurrent\.unmap\(\);\n\n\s*this\.particleBufferNext = this\.device\.createBuffer\(\{[\s\S]*?\}\);\n\s*new Float32Array\(this\.particleBufferNext\.getMappedRange\(\)\)\.set\(gpuParticleData\);\n\s*this\.particleBufferNext\.unmap\(\);/,
  `const buffer0 = this.device.createBuffer({
      size: gpuParticleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer0.getMappedRange()).set(gpuParticleData);
    buffer0.unmap();

    const buffer1 = this.device.createBuffer({
      size: gpuParticleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(buffer1.getMappedRange()).set(gpuParticleData);
    buffer1.unmap();

    this.particleBuffers = [buffer0, buffer1];`
);

// Replace updateBindGroups logic
code = code.replace(
  /private updateBindGroups\(\): void \{[\s\S]*?\}\n\n\s*private convertToGPUFormat/m,
  `private createBindGroups(): void {
    this.forceBindGroups = [
      this.device.createBindGroup({
        layout: this.forcePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[0] } },
          { binding: 1, resource: { buffer: this.forceBuffer } },
        ],
      }),
      this.device.createBindGroup({
        layout: this.forcePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[1] } },
          { binding: 1, resource: { buffer: this.forceBuffer } },
        ],
      })
    ];

    this.kickDriftBindGroups = [
      this.device.createBindGroup({
        layout: this.kickDriftPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[0] } },
          { binding: 1, resource: { buffer: this.forceBuffer } },
          { binding: 2, resource: { buffer: this.uniformBuffer } },
        ],
      }),
      this.device.createBindGroup({
        layout: this.kickDriftPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[1] } },
          { binding: 1, resource: { buffer: this.forceBuffer } },
          { binding: 2, resource: { buffer: this.uniformBuffer } },
        ],
      })
    ];

    this.kickBindGroups = [
      this.device.createBindGroup({
        layout: this.kickPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[0] } },
          { binding: 1, resource: { buffer: this.forceBuffer } },
          { binding: 2, resource: { buffer: this.uniformBuffer } },
        ],
      }),
      this.device.createBindGroup({
        layout: this.kickPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[1] } },
          { binding: 1, resource: { buffer: this.forceBuffer } },
          { binding: 2, resource: { buffer: this.uniformBuffer } },
        ],
      })
    ];

    this.interpolateBindGroups = [
      this.device.createBindGroup({
        layout: this.interpolatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[0] } },
          { binding: 1, resource: { buffer: this.particleBuffers[1] } },
          { binding: 2, resource: { buffer: this.renderPositionBuffer } },
          { binding: 3, resource: { buffer: this.interpolationUniformBuffer } },
        ],
      }),
      this.device.createBindGroup({
        layout: this.interpolatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.particleBuffers[1] } },
          { binding: 1, resource: { buffer: this.particleBuffers[0] } }, // Swapped for interpolation
          { binding: 2, resource: { buffer: this.renderPositionBuffer } },
          { binding: 3, resource: { buffer: this.interpolationUniformBuffer } },
        ],
      })
    ];
  }

  private convertToGPUFormat`
);

// Replace initial updateBindGroups call
code = code.replace(/this\.updateBindGroups\(\);/g, 'this.createBindGroups();');

// Replace physics loop buffer swap
code = code.replace(
  /\/\/ Swap GPU buffers \(double buffering\)\n\s*\[this\.particleBufferCurrent, this\.particleBufferNext\] =\n\s*\[this\.particleBufferNext, this\.particleBufferCurrent\];\n\n\s*\/\/ Update bind groups to point to swapped buffers\n\s*this\.createBindGroups\(\);/,
  '// Swap GPU buffers (double buffering) by toggling index\n      this.currentBufferIndex = 1 - this.currentBufferIndex;'
);

// Update usages in computePhysicsStep
code = code.replace(/this\.forceBindGroup/g, 'this.forceBindGroups[this.currentBufferIndex]');
code = code.replace(/this\.kickDriftBindGroup/g, 'this.kickDriftBindGroups[this.currentBufferIndex]');
code = code.replace(/this\.kickBindGroup/g, 'this.kickBindGroups[this.currentBufferIndex]');

// Update usages in getRenderPositionBuffer
code = code.replace(/this\.interpolateBindGroup/g, 'this.interpolateBindGroups[this.currentBufferIndex]');

// Update usages in destroy
code = code.replace(
  /this\.particleBufferCurrent\?\.destroy\(\);\n\s*this\.particleBufferNext\?\.destroy\(\);/,
  'this.particleBuffers?.[0]?.destroy();\n    this.particleBuffers?.[1]?.destroy();'
);

fs.writeFileSync(filePath, code);
