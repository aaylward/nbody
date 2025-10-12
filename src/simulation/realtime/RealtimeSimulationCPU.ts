/**
 * Real-time N-body simulation using CPU Barnes-Hut algorithm
 * Phase 2: O(N log N) physics with temporal interpolation
 */

import {
  createParticleArray,
  setParticle,
  removeCenterOfMassVelocity,
} from '../particleData';
import { stepBarnesHut, BarnesHutOptions } from '../barnesHut/cpuBarnesHut';
import { PerformanceMonitor } from './performanceMonitor';

export interface RealtimeSimulationCPUOptions {
  numParticles: number;
  deltaT?: number;
  targetPhysicsFPS?: number;
  theta?: number; // Barnes-Hut opening angle
}

export class RealtimeNBodySimulationCPU {
  // Particle data (double buffered on CPU)
  private particlesCurrent: Float32Array;
  private particlesNext: Float32Array;

  // Simulation state
  private numParticles: number;
  private deltaT: number;
  private running = false;
  private physicsFrameCount = 0;
  private lastPhysicsTime = 0;

  // Barnes-Hut options
  private barnesHutOptions: BarnesHutOptions;

  // Performance
  public monitor: PerformanceMonitor;
  public targetPhysicsFPS: number;

  constructor(options: RealtimeSimulationCPUOptions) {
    this.numParticles = options.numParticles;
    this.deltaT = options.deltaT ?? 0.01;
    this.targetPhysicsFPS = options.targetPhysicsFPS ?? 20;

    // Barnes-Hut settings
    this.barnesHutOptions = {
      theta: options.theta ?? 0.8, // Favor speed by default
      G: 1.0,
      softening: 2.0,
    };

    // Create particle arrays
    this.particlesCurrent = createParticleArray(this.numParticles);
    this.particlesNext = createParticleArray(this.numParticles);

    // Initialize performance monitor
    this.monitor = new PerformanceMonitor();

    // Initialize particles
    this.initializeParticles();
  }

  private initializeParticles(): void {
    // Central massive object
    setParticle(this.particlesCurrent, 0, {
      x: 0,
      y: 0,
      z: 0,
      vx: 0,
      vy: 0,
      vz: 0,
      mass: 5000,
    });

    // Orbiting particles
    for (let i = 1; i < this.numParticles; i++) {
      const r = 20 + Math.random() * 60;
      const theta = Math.random() * Math.PI * 2;
      const z = (Math.random() - 0.5) * 5;

      const x = r * Math.cos(theta);
      const y = r * Math.sin(theta);

      // Circular orbit velocity: v = sqrt(GM/r)
      const v = Math.sqrt(5000 / r);
      const vx = -v * Math.sin(theta) + (Math.random() - 0.5) * 0.5;
      const vy = v * Math.cos(theta) + (Math.random() - 0.5) * 0.5;
      const vz = (Math.random() - 0.5) * 0.2;

      setParticle(this.particlesCurrent, i, { x, y, z, vx, vy, vz, mass: 1 });
    }

    // Remove net momentum
    removeCenterOfMassVelocity(this.particlesCurrent);

    // Copy to next buffer
    this.particlesNext.set(this.particlesCurrent);
  }

  async start(): Promise<void> {
    console.log(`Starting CPU Barnes-Hut simulation with ${this.numParticles} particles...`);
    this.running = true;
    this.lastPhysicsTime = performance.now();
    this.physicsLoop();
  }

  stop(): void {
    this.running = false;
  }

  private async physicsLoop(): Promise<void> {
    let frameCount = 0;
    while (this.running) {
      // Yield to event loop before heavy computation
      await new Promise((resolve) => setTimeout(resolve, 0));

      const startTime = performance.now();

      // Compute next physics step using Barnes-Hut
      this.computePhysicsStep();

      if (frameCount === 0) {
        console.log(`First physics frame computed in ${(performance.now() - startTime).toFixed(1)}ms`);
      }
      frameCount++;

      // Swap buffers
      [this.particlesCurrent, this.particlesNext] = [this.particlesNext, this.particlesCurrent];

      this.physicsFrameCount++;
      const elapsed = performance.now() - startTime;
      this.monitor.recordPhysicsFrame(elapsed);

      // Maintain target framerate
      const targetFrameTime = 1000 / this.targetPhysicsFPS;
      const sleepTime = Math.max(0, targetFrameTime - elapsed);

      if (sleepTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, sleepTime));
      }

      this.lastPhysicsTime = performance.now();
    }
  }

  private computePhysicsStep(): void {
    // Copy current state to next buffer
    this.particlesNext.set(this.particlesCurrent);

    // Compute one timestep using Barnes-Hut
    stepBarnesHut(this.particlesNext, this.deltaT, this.barnesHutOptions);
  }

  getCurrentFrame(): Float32Array {
    return this.particlesCurrent;
  }

  getNextFrame(): Float32Array {
    return this.particlesNext;
  }

  getPhysicsProgress(): number {
    // Calculate progress to next physics frame (0.0 to 1.0)
    const frameTime = 1000 / this.targetPhysicsFPS;
    const elapsed = performance.now() - this.lastPhysicsTime;
    return Math.min(1.0, elapsed / frameTime);
  }

  getParticleCount(): number {
    return this.numParticles;
  }

  getPhysicsFrameCount(): number {
    return this.physicsFrameCount;
  }

  setTargetPhysicsFPS(fps: number): void {
    this.targetPhysicsFPS = Math.max(1, Math.min(60, fps));
  }

  setTheta(theta: number): void {
    this.barnesHutOptions.theta = Math.max(0.1, Math.min(1.5, theta));
  }

  getTheta(): number {
    return this.barnesHutOptions.theta ?? 0.5;
  }

  // No GPU resources to clean up
  destroy(): void {
    this.running = false;
  }
}
