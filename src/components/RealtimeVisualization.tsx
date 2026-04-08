/**
 * Real-time N-body visualization component
 * Supports both GPU and CPU backends
 */

import { useRef, useMemo, useEffect, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { useRealtimeStore } from '../store/useRealtimeStore';

export function RealtimeVisualization() {
  const simulation = useRealtimeStore((state) => state.simulation);
  const updateStats = useRealtimeStore((state) => state.updateStats);
  const pointsRef = useRef<THREE.Points>(null);
  const [geometry, setGeometry] = useState<THREE.BufferGeometry | null>(null);
  const [stagingBuffer, setStagingBuffer] = useState<GPUBuffer | null>(null);
  const positionArrayRef = useRef<Float32Array | null>(null);
  const geometryRef = useRef<THREE.BufferGeometry | null>(null);
  const isMappingRef = useRef<boolean>(false);

  const material = useMemo(() => {
    return new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      sizeAttenuation: true,
    });
  }, []);

  // Create geometry and staging buffer
  useEffect(() => {
    if (!simulation) return;

    // Reset mapping flag when simulation changes
    isMappingRef.current = false;

    const numParticles = simulation.getParticleCount();

    // Create geometry
    const geom = new THREE.BufferGeometry();

    // Create position and color arrays
    const positions = new Float32Array(numParticles * 3);
    const colors = new Float32Array(numParticles * 3);

    // Initialize with basic colors
    for (let i = 0; i < numParticles; i++) {
      colors[i * 3 + 0] = 0.5;
      colors[i * 3 + 1] = 0.5;
      colors[i * 3 + 2] = 1.0;
    }

    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    positionArrayRef.current = positions;
    geometryRef.current = geom;

    // Create staging buffer for GPU backend only
    const device = simulation.getDevice();
    const staging = device.createBuffer({
      size: numParticles * 4 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    setStagingBuffer(staging);

    setGeometry(geom);

    return () => {
      isMappingRef.current = false;
      geom.dispose();
      staging.destroy();
    };
  }, [simulation]);

  // Real-time render loop
  useFrame(() => {
    if (!simulation || !geometryRef.current || !positionArrayRef.current) return;

    const startTime = performance.now();
    const numParticles = simulation.getParticleCount();

    if ('getRenderPositionBuffer' in simulation && stagingBuffer) {
      // GPU backend
      const renderBuffer = simulation.getRenderPositionBuffer();
      const device = simulation.getDevice();

      // Only start mapping if not already in progress
      if (!isMappingRef.current && stagingBuffer.mapState === 'unmapped') {
        isMappingRef.current = true;

        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(renderBuffer, 0, stagingBuffer, 0, numParticles * 4 * 4);
        device.queue.submit([commandEncoder.finish()]);

        stagingBuffer.mapAsync(GPUMapMode.READ).then(() => {
          try {
            const gpuData = new Float32Array(stagingBuffer.getMappedRange());

            for (let i = 0; i < numParticles; i++) {
              positionArrayRef.current![i * 3 + 0] = gpuData[i * 4 + 0];
              positionArrayRef.current![i * 3 + 1] = gpuData[i * 4 + 1];
              positionArrayRef.current![i * 3 + 2] = gpuData[i * 4 + 2];
            }

            if (geometryRef.current) {
              const posAttr = geometryRef.current.getAttribute('position') as THREE.BufferAttribute;
              posAttr.needsUpdate = true;
            }

            stagingBuffer.unmap();
          } catch (error) {
            console.error('Error reading staging buffer:', error);
          } finally {
            isMappingRef.current = false;
          }
        }).catch((error) => {
          console.error('Error mapping staging buffer:', error);
          isMappingRef.current = false;
        });
      }
    }

    const elapsed = performance.now() - startTime;
    simulation.monitor.recordRenderFrame(elapsed);

    if (Math.random() < 0.1) {
      updateStats();
    }
  });

  if (!simulation || !geometry) return null;

  return <points ref={pointsRef} geometry={geometry} material={material} />;
}
