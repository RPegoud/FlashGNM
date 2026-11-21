// A tiny CPU GNM (matching index, power-law rule) with the same conventions as src/kernel.py:
//   K_ij = C_ij / ((k_i + k_j - 2 A_ij) / 2),  C = A^T A,  zero K replaced by OFFSET
//   P_ij = D_ij^eta * K_ij^gamma + OFFSET, zero on edges and the diagonal.
export const OFFSET = 1e-6;

export type Mat = number[][];

export const zeros = (n: number): Mat => Array.from({length: n}, () => new Array(n).fill(0));

export function distances(pos: [number, number][], scale: number): Mat {
  const n = pos.length;
  const D = zeros(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      D[i][j] = Math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]) / scale;
  return D;
}

export function degrees(A: Mat): number[] {
  return A.map(r => r.reduce((s, x) => s + x, 0));
}

export function shared(A: Mat): Mat {
  const n = A.length;
  const Cm = zeros(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += A[k][i] * A[k][j];
      Cm[i][j] = s;
    }
  return Cm;
}

export function matching(A: Mat): Mat {
  const n = A.length;
  const Cm = shared(A);
  const deg = degrees(A);
  const K = zeros(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const den = (deg[i] + deg[j] - 2 * A[i][j]) / 2;
      const k = den > 0 ? Cm[i][j] / den : 0;
      K[i][j] = k === 0 ? OFFSET : k;
    }
  return K;
}

export function wiring(A: Mat, D: Mat, eta: number, gamma: number): Mat {
  const n = A.length;
  const K = matching(A);
  const P = zeros(n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (i === j || A[i][j]) continue;
      P[i][j] = Math.pow(D[i][j], eta) * Math.pow(K[i][j], gamma) + OFFSET;
    }
  return P;
}

/** Sample an unordered pair (i < j) in proportion to P over the upper triangle. */
export function samplePair(P: Mat, u: number): [number, number] {
  const n = P.length;
  let total = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) total += P[i][j];
  let acc = 0;
  const target = u * total;
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++) {
      acc += P[i][j];
      if (acc >= target) return [i, j];
    }
  return [n - 2, n - 1];
}
