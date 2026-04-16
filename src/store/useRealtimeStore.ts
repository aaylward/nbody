/**
 * Zustand store for real-time N-body simulation
 * GPU-only: brute-force O(N²) and Barnes-Hut O(N log N) backends
 */

import { create } from 'zustand';
import { RealtimeNBodySimulation } from '../simulation/realtime/RealtimeSimulation';
import { RealtimeNBodySimulationGPUBarnesHut } from '../simulation/realtime/RealtimeSimulationGPUBarnesHut';
import { PerformanceStats } from '../simulation/realtime/performanceMonitor';

export type SimulationBackend = 'gpu' | 'gpu-barnes-hut';

// Union type for simulation
export type SimulationInstance =
  | RealtimeNBodySimulation
  | RealtimeNBodySimulationGPUBarnesHut;

export interface RealtimeState {
  simulation: SimulationInstance | null;
  backend: SimulationBackend;
  isRunning: boolean;
  physicsFrameCount: number;
  stats: PerformanceStats;
  error: string | null;
  theta: number; // Barnes-Hut opening angle
  octreeRebuildInterval: number; // Rebuild octree every N physics frames
}

export interface RealtimeActions {
  startSimulation: (numParticles: number, device: GPUDevice) => Promise<void>;
  stopSimulation: () => void;
  resetSimulation: () => void;
  setTargetFPS: (fps: number) => void;
  setBackend: (backend: SimulationBackend) => void;
  setTheta: (theta: number) => void;
  setOctreeRebuildInterval: (interval: number) => void;
  updateStats: () => void;
}

export type RealtimeStore = RealtimeState & RealtimeActions;

export const useRealtimeStore = create<RealtimeStore>((set, get) => ({
  // Initial state
  simulation: null,
  backend: 'gpu-barnes-hut', // Default to GPU Barnes-Hut for best performance
  isRunning: false,
  physicsFrameCount: 0,
  theta: 0.8, // Default Barnes-Hut opening angle (higher = faster, lower = more accurate)
  octreeRebuildInterval: 4, // Rebuild octree every 4 physics frames
  stats: {
    physicsFPS: 0,
    renderFPS: 0,
    physicsAvg: 0,
    physicsP95: 0,
    renderAvg: 0,
    renderP95: 0,
  },
  error: null,

  // Actions
  startSimulation: async (numParticles: number, device: GPUDevice) => {
    try {
      // Clean up existing simulation
      const { simulation: existing } = get();
      if (existing) {
        existing.stop();
        existing.destroy();
      }

      const { backend, theta } = get();
      let simulation: SimulationInstance;

      if (backend === 'gpu') {
        // GPU brute-force O(N²) - slow but simple
        simulation = new RealtimeNBodySimulation(device, {
          numParticles,
          targetPhysicsFPS: 20,
          deltaT: 0.01,
        });
      } else {
        // GPU Barnes-Hut O(N log N) - fast!
        simulation = new RealtimeNBodySimulationGPUBarnesHut({
          device,
          numParticles,
          targetPhysicsFPS: 20,
          deltaT: 0.01,
          theta,
        });
      }

      set({ simulation, isRunning: true, error: null });

      // Start physics loop
      await simulation.start();
    } catch (error) {
      console.error('Failed to start simulation:', error);
      set({
        error: error instanceof Error ? error.message : 'Unknown error',
        isRunning: false,
      });
    }
  },

  stopSimulation: () => {
    const { simulation } = get();
    if (simulation) {
      simulation.stop();
      set({ isRunning: false });
    }
  },

  resetSimulation: () => {
    const { simulation } = get();
    if (simulation) {
      simulation.stop();
      simulation.destroy();
    }
    set({
      simulation: null,
      isRunning: false,
      physicsFrameCount: 0,
      stats: {
        physicsFPS: 0,
        renderFPS: 0,
        physicsAvg: 0,
        physicsP95: 0,
        renderAvg: 0,
        renderP95: 0,
      },
      error: null,
    });
  },

  setTargetFPS: (fps: number) => {
    const { simulation } = get();
    if (simulation) {
      simulation.setTargetPhysicsFPS(fps);
    }
  },

  setBackend: (backend: SimulationBackend) => {
    set({ backend });
  },

  setTheta: (theta: number) => {
    const { simulation } = get();
    set({ theta });
    if (simulation && 'setTheta' in simulation) {
      simulation.setTheta(theta);
    }
  },

  setOctreeRebuildInterval: (interval: number) => {
    const { simulation } = get();
    set({ octreeRebuildInterval: interval });
    if (simulation && 'setOctreeRebuildInterval' in simulation) {
      simulation.setOctreeRebuildInterval(interval);
    }
  },

  updateStats: () => {
    const { simulation } = get();
    if (simulation) {
      set({
        stats: simulation.monitor.getStats(),
        physicsFrameCount: simulation.getPhysicsFrameCount(),
      });
    }
  },
}));
