export interface Particle {
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  mass?: number;
}

export interface NBodySnapshot {
  particles: Particle[];
  timestep: number;
}

export interface Photon {
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  dz: number;
  alive: boolean;
  path: Array<{ x: number; y: number; z: number }>;
}

export interface MCTallyData {
  data: number[][][];
  maxCount: number;
}

export type SimulationMode = 'nbody' | 'montecarlo';

export type MCVisualizationMode = 'particles' | 'tally';

export type ColorScale = 'hot' | 'cool' | 'jet' | 'viridis';

export interface NBodyState {
  snapshots: Particle[][];
  currentFrame: number;
  playing: boolean;
  particleSize: number;
  animationSpeed: number;
}

export interface MonteCarloState {
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
}
