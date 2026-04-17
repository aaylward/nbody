import { create } from 'zustand';
import { SimulationMode } from '../types';

interface SimulationStore {
  // Global state
  mode: SimulationMode;
  setMode: (mode: SimulationMode) => void;

  // N-Body state
  nbody: {
    snapshots: Float32Array[];
    currentFrame: number;
    playing: boolean;
    particleSize: number;
    animationSpeed: number;
    numParticles: number;
    numSnapshots: number;
    deltaT: number;
  };
  setNBodySnapshots: (snapshots: Float32Array[]) => void;
  setNBodyFrame: (frame: number) => void;
  setNBodyPlaying: (playing: boolean) => void;
  setNBodyParticleSize: (size: number) => void;
  setNBodyAnimationSpeed: (speed: number) => void;
  setNBodyNumParticles: (count: number) => void;
  setNBodyNumSnapshots: (count: number) => void;
  setNBodyDeltaT: (deltaT: number) => void;
}

export const useSimulationStore = create<SimulationStore>((set) => ({
  mode: 'nbody-realtime',
  setMode: (mode) => set({ mode }),

  nbody: {
    snapshots: [],
    currentFrame: 0,
    playing: false,
    particleSize: 2,
    animationSpeed: 1,
    numParticles: 1000,
    numSnapshots: 500,
    deltaT: 0.05,
  },
  setNBodySnapshots: (snapshots) =>
    set((state) => ({
      nbody: { ...state.nbody, snapshots, currentFrame: 0 },
    })),
  setNBodyFrame: (frame) =>
    set((state) => ({ nbody: { ...state.nbody, currentFrame: frame } })),
  setNBodyPlaying: (playing) =>
    set((state) => ({ nbody: { ...state.nbody, playing } })),
  setNBodyParticleSize: (size) =>
    set((state) => ({ nbody: { ...state.nbody, particleSize: size } })),
  setNBodyAnimationSpeed: (speed) =>
    set((state) => ({ nbody: { ...state.nbody, animationSpeed: speed } })),
  setNBodyNumParticles: (count) =>
    set((state) => ({ nbody: { ...state.nbody, numParticles: count } })),
  setNBodyNumSnapshots: (count) =>
    set((state) => ({ nbody: { ...state.nbody, numSnapshots: count } })),
  setNBodyDeltaT: (deltaT) =>
    set((state) => ({ nbody: { ...state.nbody, deltaT } })),
}));
