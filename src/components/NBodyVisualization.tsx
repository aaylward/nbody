import { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useSimulationStore } from '../store/useSimulationStore';

export function NBodyVisualization() {
  const { nbody, setNBodyFrame } = useSimulationStore();
  const pointsRef = useRef<THREE.Points>(null);
  const lastFrameTime = useRef(0);

  const geometry = useMemo(() => {
    const snapshot = nbody.snapshots[nbody.currentFrame];
    if (!snapshot) return null;

    const geom = new THREE.BufferGeometry();

    const positions = new Float32Array(snapshot.length * 3);
    const colors = new Float32Array(snapshot.length * 3);

    snapshot.forEach((p, i) => {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;

      const v = Math.sqrt(p.vx ** 2 + p.vy ** 2 + p.vz ** 2);
      const norm = Math.min(v / 10, 1);
      colors[i * 3] = 0.5 + norm * 0.5; // R
      colors[i * 3 + 1] = 0.5; // G
      colors[i * 3 + 2] = 1 - norm * 0.5; // B
    });

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
        size={nbody.particleSize * 0.5}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  );
}
