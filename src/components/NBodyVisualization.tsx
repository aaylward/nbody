import { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useSimulationStore } from '../store/useSimulationStore';
import { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import { extractPositions, calculateColors, getCenterOfMass, getParticleCount } from '../simulation/particleData';

export function NBodyVisualization() {
  const { nbody, setNBodyFrame } = useSimulationStore();
  const pointsRef = useRef<THREE.Points>(null);
  const lastFrameTime = useRef(0);
  const { controls } = useThree();

  const geometry = useMemo(() => {
    const snapshot = nbody.snapshots[nbody.currentFrame];
    if (!snapshot) return null;

    const geom = new THREE.BufferGeometry();

    // Extract positions and colors directly from TypedArray
    const positions = extractPositions(snapshot);
    const colors = calculateColors(snapshot, 10);

    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    return geom;
  }, [nbody.snapshots, nbody.currentFrame]);

  // Cleanup geometry on unmount
  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  useFrame((state) => {
    // Track the center of mass with the camera
    const snapshot = nbody.snapshots[nbody.currentFrame];
    if (snapshot && getParticleCount(snapshot) > 0) {
      const centerOfMass = getCenterOfMass(snapshot);
      if (controls && controls instanceof OrbitControlsImpl) {
        // Smoothly update the orbit controls target to follow the center of mass
        controls.target.set(centerOfMass.x, centerOfMass.y, centerOfMass.z);
        controls.update();
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
  });

  if (nbody.snapshots.length === 0 || !geometry) return null;

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
