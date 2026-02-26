import { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useSimulationStore } from '../store/useSimulationStore';
import { getCenterOfMass, getParticleCount } from '../simulation/particleData';
import { activeSimulation, GPU_FLOATS_PER_PARTICLE, RENDER_FLOATS_PER_PARTICLE } from '../simulation/nbody';

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
  }, [nbody.isRealTime, nbody.simulationTimestamp]); // Re-create on new simulation (timestamp change)

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

  const currentSnapshot = nbody.snapshots[nbody.currentFrame];

  // Optimization: Memoize center of mass to avoid O(N) calculation every frame.
  // We use the first snapshot because the simulation enforces zero net momentum,
  // so the Center of Mass position is effectively constant.
  const centerOfMass = useMemo(() => {
    const snapshot = nbody.snapshots[0];
    if (!snapshot || getParticleCount(snapshot) === 0) return null;
    return getCenterOfMass(snapshot);
  }, [nbody.snapshots]);

  // Update Snapshot Geometry
  useEffect(() => {
    if (nbody.isRealTime || !currentSnapshot) return;

    const count = getParticleCount(currentSnapshot);
    const geom = snapshotGeometry;

    // Check if we need to reinitialize buffer (size mismatch or first run)
    const positionAttribute = geom.getAttribute('position') as THREE.InterleavedBufferAttribute;

    if (!positionAttribute || positionAttribute.count !== count) {
        // Create InterleavedBuffer
        // We reuse the currentSnapshot array for the buffer if possible?
        // No, InterleavedBuffer takes a TypedArray. snapshot data is TypedArray.
        // We allocate a new buffer of the correct size to hold the interleaved data.
        // Actually, currentSnapshot IS interleaved data (8 floats per particle).
        // So we can just set it.

        const buffer = new THREE.InterleavedBuffer(new Float32Array(count * GPU_FLOATS_PER_PARTICLE), GPU_FLOATS_PER_PARTICLE);
        buffer.setUsage(THREE.DynamicDrawUsage);

        geom.setAttribute('position', new THREE.InterleavedBufferAttribute(buffer, 3, 0));
        geom.setAttribute('velocity', new THREE.InterleavedBufferAttribute(buffer, 3, 4));
    }

    const buffer = (geom.getAttribute('position') as THREE.InterleavedBufferAttribute).data;

    // Optimization: Fast memcpy instead of iterating
    // This replaces extractPositions and calculateColors loops
    buffer.set(currentSnapshot, 0);
    buffer.needsUpdate = true;

  }, [nbody.isRealTime, currentSnapshot, snapshotGeometry]);

  // Cleanup geometry on unmount
  useEffect(() => {
    return () => {
      snapshotGeometry?.dispose();
      realTimeGeometry?.dispose();
      particleMaterial?.dispose();
    };
  }, [snapshotGeometry, realTimeGeometry, particleMaterial]);

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

  return (
    <points ref={pointsRef} geometry={geometry} material={particleMaterial} frustumCulled={false} />
  );
}
