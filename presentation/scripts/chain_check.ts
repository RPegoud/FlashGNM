// Sanity check for clip 3's toy (bundle with esbuild, run with node).
import {Random} from '@motion-canvas/core';
import {N, D, ETA, GAMMA, toyAdjacency, rowSums, cdfPick} from '../src/lib/chain_toy';
import {wiring, matching} from '../src/lib/gnm';

for (const steps of [0, 4, 8]) {
  const A = toyAdjacency(steps, 3);
  const P = wiring(A, D, ETA, GAMMA);
  const K = matching(A);
  let nz = 0, tiny = 0, mx = 0;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++) {
      if (P[i][j] > 0) {
        nz++;
        if (K[i][j] <= 1e-6) tiny++;
      }
      mx = Math.max(mx, P[i][j]);
    }
  const R = rowSums(P);
  console.log('steps', steps, 'nonedge', nz, 'K=0', tiny, 'R/max', R.map(x => (x / Math.max(...R)).toFixed(2)).join(' '));
  const rng = new Random(11);
  const i = cdfPick(R, rng.nextFloat());
  const j = cdfPick(P[i], rng.nextFloat());
  const A2 = A.map(r => r.slice());
  A2[i][j] = A2[j][i] = 1;
  const P2 = wiring(A2, D, ETA, GAMMA);
  let out = 0;
  for (let k = 0; k < N; k++)
    for (let l = 0; l < N; l++)
      if (k !== i && k !== j && l !== i && l !== j && P2[k][l] !== P[k][l]) out++;
  const Rd = R.map((r, k) => r + (P2[i][k] - P[i][k]) + (P2[j][k] - P[j][k]));
  const R2 = rowSums(P2);
  Rd[i] = R2[i];
  Rd[j] = R2[j];
  console.log(' pick', i, j, 'P_ij/max', (P[i][j] / mx).toFixed(3), 'changed outside cross', out,
    'maxerr', Math.max(...Rd.map((x, k) => Math.abs(x - R2[k]))));
}
