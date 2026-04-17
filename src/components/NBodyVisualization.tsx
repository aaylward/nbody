import { useRef, useMemo, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { useSimulationStore } from '../store/useSimulationStore';
import { getCenterOfMass, getParticleCount } from '../simulation/particleData';
import { GPU_FLOATS_PER_PARTICLE } from '../simulation/nbody';

export function NBodyVisualization() {
  const hasSnapshots = useSimulationStore((state) => state.nbody.snapshots.length > 0);

  const pointsRef = useRef<THREE.Points>(null);
  const lastFrameTime = useRef(0);
  const lastRenderedFrame = useRef(-1);
  const lastRenderedSnapshots = useRef<Float32Array[] | null>(null);
  const { controls } = useThree();

  const particleMaterial = useMemo(() => {
    return new THREE.ShaderMaterial({
      uniforms: {
        uMaxVelocity: { value: 10.0 },
        uSize: { value: 2.0 },
      },
      vertexShader: `
            uniform float uMaxVelocity;
            uniform float uSize;
            attribute vec3 velocity;
            varying vec3 vColor;
            void main() {
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_Position = projectionMatrix * mvPosition;
                gl_PointSize = uSize * (300.0 / -mvPosition.z);

                float v = length(velocity);
                float t = min(v / uMaxVelocity, 1.0);

                vColor = mix(vec3(0.2, 0.5, 1.0), vec3(1.0, 0.8, 0.4), t);
            }
          `,
      fragmentShader: `
            varying vec3 vColor;
            void main() {
                vec2 coord = gl_PointCoord - vec2(0.5);
                if (length(coord) > 0.5) discard;

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

  const snapshotGeometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.boundingSphere = new THREE.Sphere(new THREE.Vector3(), Infinity);
    return geom;
  }, []);

  // Granular selector: avoid re-render when snapshots are appended.
  const initialSnapshot = useSimulationStore((state) => state.nbody.snapshots[0]);

  // First snapshot's COM is reused: zero net momentum keeps it fixed.
  const centerOfMass = useMemo(() => {
    if (!initialSnapshot || getParticleCount(initialSnapshot) === 0) return null;
    return getCenterOfMass(initialSnapshot);
  }, [initialSnapshot]);

  useEffect(() => {
    return () => {
      snapshotGeometry?.dispose();
      particleMaterial?.dispose();
    };
  }, [snapshotGeometry, particleMaterial]);

  useFrame((state) => {
    const nbody = useSimulationStore.getState().nbody;

    if (centerOfMass) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if (controls && (controls as any).target) {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (controls as any).target.set(centerOfMass.x, centerOfMass.y, centerOfMass.z);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (controls as any).update();
      }
    }

    const currentSnapshot = nbody.snapshots[nbody.currentFrame];
    if (!currentSnapshot) return;

    const snapshotsChanged = lastRenderedSnapshots.current !== nbody.snapshots;
    if (lastRenderedFrame.current !== nbody.currentFrame || snapshotsChanged) {
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
      lastRenderedSnapshots.current = nbody.snapshots;
    }

    if (!nbody.playing || nbody.snapshots.length === 0) return;

    const elapsed = state.clock.getElapsedTime() * 1000;
    const interval = 1000 / (50 * nbody.animationSpeed);

    if (elapsed - lastFrameTime.current > interval) {
      const nextFrame = (nbody.currentFrame + 1) % nbody.snapshots.length;
      useSimulationStore.getState().setNBodyFrame(nextFrame);
      lastFrameTime.current = elapsed;
    }
  });

  if (!hasSnapshots) return null;

  return (
    <points ref={pointsRef} geometry={snapshotGeometry} material={particleMaterial} frustumCulled={false} />
  );
}
