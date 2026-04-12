# N-Body Simulation Viewer

Interactive gravitational N-body simulation running in the browser. Simulates thousands of particles orbiting a central mass, with real-time 3D visualization.

## Features

- **WebGPU-accelerated physics** — compute shaders handle force calculations on the GPU, with automatic CPU fallback
- **Barnes-Hut algorithm** — O(N log N) force approximation via octree spatial decomposition for the real-time GPU mode
- **Symplectic integration** — kick-drift-kick leapfrog integrator preserves energy over long simulation runs
- **Two simulation modes:**
  - **Pre-computed** — generates all timesteps up front, then plays back with interpolated frames for smooth animation
  - **Real-time** — steps physics live each frame on the GPU with adaptive performance monitoring
- **3D orbit controls** — pan, zoom, and rotate the camera around the simulation

## Getting Started

```bash
npm install
npm run dev
```

Open http://localhost:5173 in a browser with [WebGPU support](https://caniuse.com/webgpu) for GPU acceleration. The simulation falls back to CPU if WebGPU is unavailable.

## Usage

Use the top bar to switch between simulation modes. The control panel lets you adjust:

- Particle count
- Timestep size (delta T)
- Number of snapshots (pre-computed mode)
- Particle render size
- Playback speed

## How It Works

Particles are initialized in a disk around a heavy central body, with velocities set for approximately circular orbits. The simulation computes gravitational forces between all particle pairs (or uses Barnes-Hut approximation) and integrates their trajectories using the leapfrog method.

The particle data is stored in flat `Float32Array` buffers with a struct-of-arrays layout (8 floats per particle: position xyz, padding, velocity xyz, mass) that maps directly to GPU storage buffers.

## Tech Stack

- React + TypeScript
- Three.js via React Three Fiber
- WebGPU (WGSL compute shaders)
- Zustand for state management
- Vite + Vitest

## Deployment

Static site deployed via Cloudflare Pages (`wrangler.jsonc`).

```bash
npm run build
```

## Tests

```bash
npm test
```
