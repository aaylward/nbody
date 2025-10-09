import { Photon, MCTallyData, ColorScale } from '../types';

export const MC_BINS = 50;
export const MC_BOX_SIZE = 100;
export const ABSORPTION_COEFF = 0.1;
export const SCATTERING_COEFF = 0.5;

export class PhotonSimulator {
  x: number;
  y: number;
  z: number;
  dx: number;
  dy: number;
  dz: number;
  alive: boolean;
  path: Array<{ x: number; y: number; z: number }>;

  constructor() {
    this.x = 0;
    this.y = 0;
    this.z = 0;
    this.alive = true;
    this.path = [];

    // Random initial direction
    const costheta = 2 * Math.random() - 1;
    const sintheta = Math.sqrt(1 - costheta * costheta);
    const phi = 2 * Math.PI * Math.random();
    this.dx = sintheta * Math.cos(phi);
    this.dy = sintheta * Math.sin(phi);
    this.dz = costheta;

    this.path.push({ x: 0, y: 0, z: 0 });
  }

  update(): { absorbed: boolean; escaped: boolean } {
    if (!this.alive) return { absorbed: false, escaped: false };

    // Sample free path
    const mu_t = ABSORPTION_COEFF + SCATTERING_COEFF;
    const distance = (-Math.log(Math.random()) / mu_t) * 5;

    this.x += this.dx * distance;
    this.y += this.dy * distance;
    this.z += this.dz * distance;

    this.path.push({ x: this.x, y: this.y, z: this.z });

    // Check boundaries
    if (
      Math.abs(this.x) > MC_BOX_SIZE / 2 ||
      Math.abs(this.y) > MC_BOX_SIZE / 2 ||
      Math.abs(this.z) > MC_BOX_SIZE / 2
    ) {
      this.alive = false;
      return { absorbed: false, escaped: true };
    }

    // Interaction
    if (Math.random() < ABSORPTION_COEFF / mu_t) {
      this.alive = false;
      return { absorbed: true, escaped: false };
    }

    // Scatter
    const costheta = 2 * Math.random() - 1;
    const sintheta = Math.sqrt(1 - costheta * costheta);
    const phi = 2 * Math.PI * Math.random();
    this.dx = sintheta * Math.cos(phi);
    this.dy = sintheta * Math.sin(phi);
    this.dz = costheta;

    return { absorbed: false, escaped: false };
  }

  toPhoton(): Photon {
    return {
      x: this.x,
      y: this.y,
      z: this.z,
      dx: this.dx,
      dy: this.dy,
      dz: this.dz,
      alive: this.alive,
      path: this.path,
    };
  }
}

export function generateMCTallyDemo(): MCTallyData {
  const data: number[][][] = new Array(MC_BINS)
    .fill(0)
    .map(() => new Array(MC_BINS).fill(0).map(() => new Array(MC_BINS).fill(0)));

  let maxCount = 0;
  const cx = MC_BINS / 2;
  const cy = MC_BINS / 2;
  const cz = MC_BINS / 2;

  for (let ix = 0; ix < MC_BINS; ix++) {
    for (let iy = 0; iy < MC_BINS; iy++) {
      for (let iz = 0; iz < MC_BINS; iz++) {
        const r = Math.sqrt((ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2);
        const count = Math.floor(
          10000 * Math.exp(-r / 8) * (0.7 + Math.random() * 0.3)
        );

        if (count > 0) {
          data[ix][iy][iz] = count;
          maxCount = Math.max(maxCount, count);
        }
      }
    }
  }

  return { data, maxCount };
}

export function getColorForValue(
  value: number,
  scale: ColorScale
): [number, number, number] {
  const scales: Record<ColorScale, number[][]> = {
    hot: [
      [0, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
      [1, 1, 1],
    ],
    cool: [
      [0, 0, 0.5],
      [0, 1, 1],
      [0, 1, 0],
    ],
    jet: [
      [0, 0, 0.5],
      [0, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
      [1, 0, 0],
    ],
    viridis: [
      [0.267, 0.005, 0.329],
      [0.128, 0.566, 0.551],
      [0.993, 0.906, 0.144],
    ],
  };

  const colorScale = scales[scale] || scales.hot;
  const segment = value * (colorScale.length - 1);
  const idx = Math.floor(segment);
  const t = segment - idx;

  if (idx >= colorScale.length - 1) {
    const last = colorScale[colorScale.length - 1];
    return [last[0], last[1], last[2]];
  }

  return [
    colorScale[idx][0] + t * (colorScale[idx + 1][0] - colorScale[idx][0]),
    colorScale[idx][1] + t * (colorScale[idx + 1][1] - colorScale[idx][1]),
    colorScale[idx][2] + t * (colorScale[idx + 1][2] - colorScale[idx][2]),
  ];
}
