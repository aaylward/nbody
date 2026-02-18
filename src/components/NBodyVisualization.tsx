import { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useSimulationStore } from '../store/useSimulationStore';
import { extractPositions, calculateColors, getCenterOfMass, getParticleCount } from '../simulation/particleData';
import { activeSimulation, GPU_FLOATS_PER_PARTICLE } from '../simulation/nbody';

export function NBodyVisualization() {
  const { nbody, setNBodyFrame } = useSimulationStore();
  const pointsRef = useRef<THREE.Points>(null);
  const lastFrameTime = useRef(0);
  const { controls } = useThree();
  const isReadingBack = useRef(false);

  // Initialize geometry for Real-time mode
  const realTimeGeometry = useMemo(() => {
    if (!nbody.isRealTime || !activeSimulation) return null;

    const geom = new THREE.BufferGeometry();
    const count = activeSimulation.numParticles;

    // Allocate buffer for particles * 8 floats (32 bytes)
    const buffer = new THREE.InterleavedBuffer(new Float32Array(count * GPU_FLOATS_PER_PARTICLE), GPU_FLOATS_PER_PARTICLE);
    buffer.setUsage(THREE.DynamicDrawUsage);

    // Position: offset 0 (x, y, z)
    geom.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
    // Velocity: offset 4 (vx, vy, vz) - used for color in shader
    geom.setAttribute('velocity', new THREE.InterleavedBufferAttribute(buffer, 3, 4));

    return geom;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nbody.isRealTime, nbody.simulationTimestamp]); // Re-create on new simulation (timestamp change)

  // Shader Material for Real-time
  const realTimeMaterial = useMemo(() => {
      return new THREE.ShaderMaterial({
          vertexShader: `
            attribute vec3 velocity;
            varying vec3 vColor;
            void main() {
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_Position = projectionMatrix * mvPosition;
                gl_PointSize = 2.0 * (300.0 / -mvPosition.z);

                // Color based on velocity magnitude
                float v = length(velocity);
                // Adjust scaling based on simulation parameters (approximate)
                float t = min(v / 5.0, 1.0);
                vColor = mix(vec3(0.2, 0.5, 1.0), vec3(1.0, 0.8, 0.4), t);
            }
          `,
          fragmentShader: `
            varying vec3 vColor;
            void main() {
                // Circular particle
                vec2 coord = gl_PointCoord - vec2(0.5);
                if (length(coord) > 0.5) discard;

                // Soft edge
                float dist = length(coord);
                float alpha = 1.0 - smoothstep(0.4, 0.5, dist);

                gl_FragColor = vec4(vColor, 0.8 * alpha);
            }
          `,
          transparent: true,
          depthWrite: false,
          blending: THREE.AdditiveBlending,
      });
  }, []);

  // Persistent geometry for Snapshot mode to avoid reallocation
  const snapshotGeometry = useMemo(() => new THREE.BufferGeometry(), []);

  const currentSnapshot = nbody.snapshots[nbody.currentFrame];

  // Optimization: Memoize center of mass to avoid O(N) calculation every frame
  const centerOfMass = useMemo(() => {
    if (!currentSnapshot || getParticleCount(currentSnapshot) === 0) return null;
    return getCenterOfMass(currentSnapshot);
  }, [currentSnapshot]);

  // Update Snapshot Geometry
  useEffect(() => {
    if (nbody.isRealTime || !currentSnapshot) return;

    const count = getParticleCount(currentSnapshot);
    const geom = snapshotGeometry;

    // Resize or Init buffers if needed
    let positions = geom.getAttribute('position') as THREE.BufferAttribute;
    let colors = geom.getAttribute('color') as THREE.BufferAttribute;

    if (!positions || positions.count !== count) {
        // Reallocate only if size changes
        positions = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
        colors = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
        geom.setAttribute('position', positions);
        geom.setAttribute('color', colors);
    }

    // Optimization: Write directly to existing buffers to avoid allocation
    extractPositions(currentSnapshot, positions.array as Float32Array);
    calculateColors(currentSnapshot, 10, colors.array as Float32Array);

    positions.needsUpdate = true;
    colors.needsUpdate = true;

    geom.computeBoundingSphere();

  }, [nbody.isRealTime, currentSnapshot, snapshotGeometry]);

  // Cleanup geometry on unmount
  useEffect(() => {
    return () => {
      snapshotGeometry?.dispose();
      realTimeGeometry?.dispose();
      realTimeMaterial?.dispose();
    };
  }, [snapshotGeometry, realTimeGeometry, realTimeMaterial]);

  useFrame((state) => {
    if (nbody.isRealTime) {
         if (!nbody.playing || !activeSimulation || !realTimeGeometry || isReadingBack.current) return;

         isReadingBack.current = true;
         activeSimulation.step(nbody.deltaT);

         // Readback
         const attr = realTimeGeometry.getAttribute('position') as THREE.InterleavedBufferAttribute;
         const buffer = attr.data;
         const outData = buffer.array as Float32Array;

         activeSimulation.getParticleData(outData).then(() => {
             buffer.needsUpdate = true;
             isReadingBack.current = false;
         }).catch(e => {
             console.error("Readback failed", e);
             isReadingBack.current = false;
         });

         // Skip center of mass tracking for real-time (too expensive for 1M particles on CPU)

    } else {
        // Snapshot mode
        if (centerOfMass) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            if (controls && (controls as any).target) {
                // Smoothly update the orbit controls target to follow the center of mass
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                (controls as any).target.set(centerOfMass.x, centerOfMass.y, centerOfMass.z);
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                (controls as any).update();
            }
        }

        if (!nbody.playing || nbody.snapshots.length === 0) return;

        const elapsed = state.clock.getElapsedTime() * 1000;
        const interval = 1000 / (50 * nbody.animationSpeed);

        if (elapsed - lastFrameTime.current > interval) {
            const nextFrame = (nbody.currentFrame + 1) % nbody.snapshots.length;
            setNBodyFrame(nextFrame);
            lastFrameTime.current = elapsed;
        }
    }
  });

  const geometry = nbody.isRealTime ? realTimeGeometry : snapshotGeometry;

  if (nbody.snapshots.length === 0 || !geometry) return null;

  if (nbody.isRealTime) {
      return (
        <points ref={pointsRef} geometry={geometry} material={realTimeMaterial} />
      );
  }

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        size={0.5}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  );
}
