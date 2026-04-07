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

export type SimulationMode = 'nbody' | 'nbody-realtime';

export interface NBodyState {
  snapshots: Float32Array[];
  currentFrame: number;
  playing: boolean;
  particleSize: number;
  animationSpeed: number;
}
