import * as THREE from 'three';

const numParticles = 50000;
const positions = new Float32Array(numParticles * 3);

// Fill with random data
for (let i = 0; i < positions.length; i++) {
    positions[i] = Math.random() * 100 - 50;
}

const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));

const start = performance.now();
for (let i = 0; i < 100; i++) {
    // Invalidate bounding sphere to force recompute?
    // Three.js computes it and stores it.
    // We need to set it to null to force recompute?
    geom.boundingSphere = null;
    geom.computeBoundingSphere();
}
const end = performance.now();

console.log(`Average time per computeBoundingSphere (N=${numParticles}): ${(end - start) / 100} ms`);
