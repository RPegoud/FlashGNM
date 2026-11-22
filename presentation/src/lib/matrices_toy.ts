// Toy state and one kernel step for clip 4, mirroring src/kernel.py (binary_gnm_kernel):
//   state per simulation: A bool [n,n], C int16 [n,n], deg int16 [n], R fp64 [n]
//   shared: D fp32 [n,n]; per simulation: eta, gamma fp32
// The step follows the kernel's order: row i ~ R, old rows of i (P_i from C, A, deg, D),
// column j ~ old P_i, old rows of j, then degrees, C rows (+ old A rows), new P rows,
// R patched by deltas with rows i and j summed exactly, and A[i,j] written last.
import {Mat, OFFSET, degrees, distances, shared, wiring, zeros} from './gnm';

export const N = 8;

// Node positions (only used for distances). Arbitrary but fixed.
const RAW: [number, number][] = [
  [-160, -90], [-40, -150], [110, -120], [170, 10],
  [90, 130], [-60, 140], [-170, 60], [10, -10],
];
export const D: Mat = distances(RAW, 100);
for (let i = 0; i < N; i++) D[i][i] = 1; // kernel convention: diagonal 1 (masked)

/** Seed network: each node wired to its two nearest neighbours. */
export function seedA(): Mat {
  const A = zeros(N);
  for (let i = 0; i < N; i++) {
    const order = [...Array(N).keys()].filter(j => j !== i).sort((a, b) => D[i][a] - D[i][b]);
    for (const b of order.slice(0, 2)) A[i][b] = A[b][i] = 1;
  }
  return A;
}

export interface State {
  A: Mat;
  C: Mat;
  deg: number[];
  R: number[];
}

export function initState(A: Mat, eta: number, gamma: number): State {
  const P = wiring(A, D, eta, gamma);
  return {
    A: A.map(r => r.slice()),
    C: shared(A),
    deg: degrees(A),
    R: P.map(r => r.reduce((s, x) => s + x, 0)),
  };
}

/** _row_P: row x of P from rows x of C, A, D and all degrees. */
export function rowP(
  Crow: number[], Arow: number[], Drow: number[], deg: number[], degx: number, x: number,
  eta: number, gamma: number,
): number[] {
  return Crow.map((c, m) => {
    let den = (degx + deg[m] - 2 * Arow[m]) / 2;
    if (den === 0) den = 1;
    let K = c / den;
    if (K === 0) K = OFFSET;
    const p = Math.exp(eta * Math.log(Drow[m])) * Math.exp(gamma * Math.log(K)) + OFFSET;
    return Arow[m] === 0 && m !== x ? p : 0;
  });
}

/** _sample_cdf: number of inclusive-cdf entries <= u * total. */
export function sampleCdf(W: number[], u: number): number {
  let acc = 0;
  const cdf = W.map(w => (acc += w));
  const t = u * cdf[cdf.length - 1];
  return Math.min(W.length - 1, cdf.filter(c => c <= t).length);
}

export interface Step {
  i: number;
  j: number;
  oldAi: number[];
  oldAj: number[];
  oldCi: number[];
  oldCj: number[];
  oldPi: number[];
  oldPj: number[];
  Ci: number[];
  Cj: number[];
  Pi: number[];
  Pj: number[];
  degOld: number[];
  degNew: number[];
  Rold: number[];
  Rnew: number[];
}

/** One kernel step; mutates s (A written last, as in the kernel). */
export function step(s: State, eta: number, gamma: number, u0: number, u1: number): Step {
  const Rold = s.R.slice();
  const degOld = s.deg.slice();
  const i = sampleCdf(s.R, u0);
  const oldAi = s.A[i].slice();
  const oldCi = s.C[i].slice();
  const oldPi = rowP(oldCi, oldAi, D[i], s.deg, s.deg[i], i, eta, gamma);
  const j = sampleCdf(oldPi, u1);
  const oldAj = s.A[j].slice();
  const oldCj = s.C[j].slice();
  const oldPj = rowP(oldCj, oldAj, D[j], s.deg, s.deg[j], j, eta, gamma);

  const deg = s.deg.map((d, m) => (m === i || m === j ? d + 1 : d));
  const Ai = oldAi.map((a, m) => a + (m === j ? 1 : 0));
  const Aj = oldAj.map((a, m) => a + (m === i ? 1 : 0));
  const Ci = oldCi.map((c, m) => c + oldAj[m] + (m === i ? 1 : 0));
  const Cj = oldCj.map((c, m) => c + oldAi[m] + (m === j ? 1 : 0));
  // _store_sym: rows, then columns
  for (let m = 0; m < N; m++) {
    s.C[i][m] = Ci[m];
    s.C[m][i] = Ci[m];
  }
  for (let m = 0; m < N; m++) {
    s.C[j][m] = Cj[m];
    s.C[m][j] = Cj[m];
  }
  s.deg = deg;
  const Pi = rowP(Ci, Ai, D[i], deg, deg[i], i, eta, gamma);
  const Pj = rowP(Cj, Aj, D[j], deg, deg[j], j, eta, gamma);
  const R = s.R.map((r, m) => r + (Pi[m] - oldPi[m]) + (Pj[m] - oldPj[m]));
  R[i] = Pi.reduce((a, b) => a + b, 0);
  R[j] = Pj.reduce((a, b) => a + b, 0);
  s.R = R;
  s.A[i][j] = s.A[j][i] = 1;
  return {i, j, oldAi, oldAj, oldCi, oldCj, oldPi, oldPj, Ci, Cj, Pi, Pj, degOld, degNew: deg, Rold, Rnew: R};
}

/** Max abs difference between s.R and a full recompute (a check on the delta update). */
export function rDrift(s: State, eta: number, gamma: number): number {
  const P = wiring(s.A, D, eta, gamma);
  return Math.max(...P.map((r, m) => Math.abs(r.reduce((a, b) => a + b, 0) - s.R[m])));
}
