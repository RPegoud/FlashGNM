// Toy state for clip 3 (two-stage sampling): clip 1's 18-node graph, grown a few steps.
// Everything shown in the clip is computed here with src/lib/gnm.ts (kernel conventions).
import {Random} from '@motion-canvas/core';
import {Mat, distances, samplePair, wiring, zeros} from './gnm';

export const ETA = -3;
export const GAMMA = 1.5;

// Same node layout as clip 1 (src/scenes/wiring.tsx), so D is identical.
const RAW: [number, number][] = [
  [-300, -60], [-225, -175], [-95, -235], [60, -245], [205, -190], [315, -85],
  [345, 45], [285, 165], [150, 235], [0, 255], [-150, 225], [-285, 130],
  [-165, -70], [-45, -125], [110, -85], [205, 45], [60, 100], [-100, 85],
];
export const POS: [number, number][] = RAW.map(([x, y]) => [x * 1.12, y * 1.12]);
export const N = POS.length;
export const D = distances(POS, 150);

/** Seed (each node wired to its two nearest neighbours, as in clip 1) plus `steps` model steps. */
export function toyAdjacency(steps: number, seed: number): Mat {
  const A: Mat = zeros(N);
  for (let i = 0; i < N; i++) {
    const order = [...Array(N).keys()].filter(j => j !== i).sort((a, b) => D[i][a] - D[i][b]);
    for (const b of order.slice(0, 2)) A[i][b] = A[b][i] = 1;
  }
  const rng = new Random(seed);
  for (let s = 0; s < steps; s++) {
    const [i, j] = samplePair(wiring(A, D, ETA, GAMMA), rng.nextFloat());
    A[i][j] = A[j][i] = 1;
  }
  return A;
}

export const rowSums = (P: Mat) => P.map(r => r.reduce((s, x) => s + x, 0));

/** Inverse CDF exactly as kernel.py `_sample_cdf`: number of inclusive-cdf entries <= u * sum(w). */
export function cdfPick(w: number[], u: number): number {
  let total = 0;
  for (const x of w) total += x;
  const t = u * total;
  let acc = 0;
  let count = 0;
  for (const x of w) {
    acc += x;
    if (acc <= t) count++;
  }
  return Math.min(count, w.length - 1);
}
