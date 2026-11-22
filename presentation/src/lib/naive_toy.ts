// Toy model for clip 2 (naive n^2 sampling). Pure TS so it can be checked outside the editor.
import {Mat, distances, wiring, zeros} from './gnm';

export const ETA = -2;
export const GAMMA = 1;

// 12 regions in a loose oval (arbitrary toy layout; distances are in units of 100 px).
const RAW: [number, number][] = [
  [-260, -40], [-190, -150], [-60, -200], [80, -190], [210, -120], [270, 10],
  [220, 140], [90, 200], [-60, 190], [-200, 120], [-90, -40], [70, 30],
];
export const N = RAW.length;
export const D: Mat = distances(RAW, 100);

/** Seed network: each node wired to its two nearest neighbours. */
export function seedA(): Mat {
  const A = zeros(N);
  for (let i = 0; i < N; i++) {
    const order = [...Array(N).keys()].filter(j => j !== i).sort((a, b) => D[i][a] - D[i][b]);
    for (const b of order.slice(0, 2)) A[i][b] = A[b][i] = 1;
  }
  return A;
}

export const P_of = (A: Mat) => wiring(A, D, ETA, GAMMA);

/**
 * The toolbox's sampler (gnm/model.py binary_update): flatten all n^2 entries row-major,
 * draw one entry in proportion to its value. Shown here as an inverse CDF on u.
 * Returns the entry index e, the flat values and the normalised cumulative sum.
 */
export function flatSample(P: Mat, u: number): {e: number; flat: number[]; cdf: number[]} {
  const flat = P.flat();
  const cdf: number[] = [];
  let s = 0;
  for (const x of flat) cdf.push((s += x));
  const t = u * s;
  let e = cdf.findIndex(c => c > t);
  if (e < 0) e = flat.length - 1;
  return {e, flat, cdf: cdf.map(c => c / s)};
}
