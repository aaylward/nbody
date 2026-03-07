import { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useSimulationStore } from '../store/useSimulationStore';
import { getCenterOfMass, getParticleCount } from '../simulation/particleData';
import { activeSimulation, GPU_FLOATS_PER_PARTICLE, RENDER_FLOATS_PER_PARTICLE } from '../simulation/nbody';

export function NBodyVisualization() {
  // Use granular selectors to avoid re-rendering the entire 3D component on every frame
  const isRealTime = useSimulationStore((state) => state.nbody.isRealTime);
  const simulationTimestamp = useSimulationStore((state) => state.nbody.simulationTimestamp);
  const hasSnapshots = useSimulationStore((state) => state.nbody.snapshots.length > 0);

  const pointsRef = useRef<THREE.Points>(null);
  const lastFrameTime = useRef(0);
  const lastRenderedFrame = useRef(-1);
  const lastRenderedTimestamp = useRef(0);
  const { controls } = useThree();
  const isReadingBack = useRef(false);

  // Initialize geometry for Real-time mode
  const realTimeGeometry = useMemo(() => {
    if (!isRealTime || !activeSimulation) return null;

    const geom = new THREE.BufferGeometry();
    const count = activeSimulation.numParticles;

    // Allocate buffer for particles * 6 floats (24 bytes) - optimized readback layout
    // This matches the compact buffer layout from the GPU
    const stride = RENDER_FLOATS_PER_PARTICLE;
    const buffer = new THREE.InterleavedBuffer(new Float32Array(count * stride), stride);
    buffer.setUsage(THREE.DynamicDrawUsage);

    // Position: offset 0 (x, y, z)
    geom.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
    // Velocity: offset 3 (vx, vy, vz) - used for color in shader
    geom.setAttribute('velocity', new THREE.InterleavedBufferAttribute(buffer, 3, 3));

    // Optimization: Set large bounding sphere to avoid recomputation and culling issues
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);

    return geom;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isRealTime, simulationTimestamp]); // Re-create on new simulation (timestamp change)

  // Shared Shader Material for both Real-time and Snapshot modes
  const particleMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        uMaxVelocity: { value: 10.0 }, // Matches calculateColors maxVelocity
        uSize: { value: 2.0 }, // Adjusted for visual parity
      },
      vertexShader: `
            uniform float uMaxVelocity;
            uniform float uSize;
            attribute vec3 velocity;
            varying vec3 vColor;
            void main() {
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_Position = projectionMatrix * mvPosition;
                // Size attenuation
                gl_PointSize = uSize * (300.0 / -mvPosition.z);

                // Color based on velocity magnitude
                float v = length(velocity);
                float t = min(v / uMaxVelocity, 1.0);

                // Blue-ish to Orange-ish gradient
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
  const snapshotGeometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    // Optimization: Set large bounding sphere to avoid recomputation
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);
    return geom;
  }, []);

  // Optimization: Track initial snapshot to compute center of mass once.
  // We use a granular selector so this doesn't trigger a render when the snapshot array is appended.
  const initialSnapshot = useSimulationStore((state) => state.nbody.snapshots[0]);

  // Optimization: Memoize center of mass to avoid O(N) calculation every frame.
  // We use the first snapshot because the simulation enforces zero net momentum,
  // so the Center of Mass position is effectively constant.
  const centerOfMass = useMemo(() => {
    if (!initialSnapshot || getParticleCount(initialSnapshot) === 0) return null;
    return getCenterOfMass(initialSnapshot);
  }, [initialSnapshot]);

  // Cleanup geometry on unmount
  useEffect(() => {
    return () => {
      snapshotGeometry?.dispose();
      realTimeGeometry?.dispose();
      particleMaterial?.dispose();
    };
  }, [snapshotGeometry, realTimeGeometry, particleMaterial]);

  useFrame((state) => {
    const nbody = useSimulationStore.getState().nbody;

    if (isRealTime) {
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

        const currentSnapshot = nbody.snapshots[nbody.currentFrame];
        if (!currentSnapshot) return;

        // Update geometry buffer outside React render loop if the frame changed
        if (lastRenderedFrame.current !== nbody.currentFrame || lastRenderedTimestamp.current !== nbody.simulationTimestamp) {
            const count = getParticleCount(currentSnapshot);
            const geom = snapshotGeometry;

            const positionAttribute = geom.getAttribute('position') as THREE.InterleavedBufferAttribute;

            if (!positionAttribute || positionAttribute.count !== count) {
                const buffer = new THREE.InterleavedBuffer(new Float32Array(count * GPU_FLOATS_PER_PARTICLE), GPU_FLOATS_PER_PARTICLE);
                buffer.setUsage(THREE.DynamicDrawUsage);

                geom.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
                geom.setAttribute('velocity', new THREE.InterleavedBufferAttribute(buffer, 3, 4));
            }

            const buffer = (geom.getAttribute('position') as THREE.InterleavedBufferAttribute).data;
            buffer.set(currentSnapshot, 0);
            buffer.needsUpdate = true;

            lastRenderedFrame.current = nbody.currentFrame;
            lastRenderedTimestamp.current = nbody.simulationTimestamp;
        }

        if (!nbody.playing || nbody.snapshots.length === 0) return;

        const elapsed = state.clock.getElapsedTime() * 1000;
        const interval = 1000 / (50 * nbody.animationSpeed);

        if (elapsed - lastFrameTime.current > interval) {
            const nextFrame = (nbody.currentFrame + 1) % nbody.snapshots.length;
            useSimulationStore.getState().setNBodyFrame(nextFrame);
            lastFrameTime.current = elapsed;
        }
    }
  });

  const geometry = isRealTime ? realTimeGeometry : snapshotGeometry;

  if (!hasSnapshots || !geometry) return null;

  return (
    <points ref={pointsRef} geometry={geometry} material={particleMaterial} frustumCulled={false} />
  );
}
