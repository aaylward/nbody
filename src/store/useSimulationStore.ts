import { create } from 'zustand';
import {
  SimulationMode,
  Photon,
  MCTallyData,
  MCVisualizationMode,
  ColorScale,
} from '../types';

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
    isRealTime: boolean;
    simulationTimestamp: number;
  };
  setNBodySnapshots: (snapshots: Float32Array[]) => void;
  setNBodyFrame: (frame: number) => void;
  setNBodyPlaying: (playing: boolean) => void;
  setNBodyParticleSize: (size: number) => void;
  setNBodyAnimationSpeed: (speed: number) => void;
  setNBodyNumParticles: (count: number) => void;
  setNBodyNumSnapshots: (count: number) => void;
  setNBodyDeltaT: (deltaT: number) => void;
  setNBodyRealTime: (isRealTime: boolean) => void;
  setNBodySimulationTimestamp: (timestamp: number) => void;

  // Monte Carlo state
  montecarlo: {
    photons: Photon[];
    tallyData: MCTallyData | null;
    animating: boolean;
    absorbed: number;
    escaped: number;
    vizMode: MCVisualizationMode;
    numPhotons: number;
    particleSpeed: number;
    opacity: number;
    colorScale: ColorScale;
  };
  setMCPhotons: (photons: Photon[]) => void;
  setMCTallyData: (data: MCTallyData | null) => void;
  setMCAnimating: (animating: boolean) => void;
  setMCAbsorbed: (count: number) => void;
  setMCEscaped: (count: number) => void;
  setMCVizMode: (mode: MCVisualizationMode) => void;
  setMCNumPhotons: (count: number) => void;
  setMCParticleSpeed: (speed: number) => void;
  setMCOpacity: (opacity: number) => void;
  setMCColorScale: (scale: ColorScale) => void;
}

export const useSimulationStore = create<SimulationStore>((set) => ({
  mode: 'nbody',
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
    isRealTime: false,
    simulationTimestamp: 0,
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
  setNBodyRealTime: (isRealTime) =>
    set((state) => ({ nbody: { ...state.nbody, isRealTime } })),
  setNBodySimulationTimestamp: (timestamp) =>
    set((state) => ({ nbody: { ...state.nbody, simulationTimestamp: timestamp } })),

  montecarlo: {
    photons: [],
    tallyData: null,
    animating: false,
    absorbed: 0,
    escaped: 0,
    vizMode: 'particles',
    numPhotons: 100,
    particleSpeed: 1,
    opacity: 0.5,
    colorScale: 'hot',
  },
  setMCPhotons: (photons) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, photons } })),
  setMCTallyData: (tallyData) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, tallyData } })),
  setMCAnimating: (animating) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, animating } })),
  setMCAbsorbed: (absorbed) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, absorbed } })),
  setMCEscaped: (escaped) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, escaped } })),
  setMCVizMode: (vizMode) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, vizMode } })),
  setMCNumPhotons: (numPhotons) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, numPhotons } })),
  setMCParticleSpeed: (particleSpeed) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, particleSpeed } })),
  setMCOpacity: (opacity) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, opacity } })),
  setMCColorScale: (colorScale) =>
    set((state) => ({ montecarlo: { ...state.montecarlo, colorScale } })),
}));
