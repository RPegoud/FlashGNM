import {Rect, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeOutCubic, useRandom, waitFor} from '@motion-canvas/core';
import {C, ramp} from '../lib/theme';
import {beat, header} from '../lib/ui';
import {D, ETA, GAMMA, N, cdfPick, rowSums, toyAdjacency} from '../lib/chain_toy';
import {wiring} from '../lib/gnm';
import {CS, description, matrix, steps} from '../lib/maint';

// Maintainer video, scene 3: one step of the new kernel, in the order of src/kernel.py
// (binary_gnm_kernel, steps 1-5). P is never stored: rows are recomputed in registers.
// Toy: src/lib/chain_toy.ts (kernel conventions); the toy asserts that only rows and
// columns i, j of P change.

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '3', 'The new kernel: one step');
  const rng = useRandom(53);

  const A = toyAdjacency(6, 5);
  const P = wiring(A, D, ETA, GAMMA);
  const R = rowSums(P);
  const M = matrix(view, P, true, R);
  const S = steps(view, [
    {tex: 'i \\sim R', cost: 'O(n)'},
    {tex: 'P_i^{\\mathrm{old}} \\leftarrow f(C_i, A_i, k, D_i)', cost: 'O(n)'},
    {tex: 'j \\sim P_i^{\\mathrm{old}}', cost: 'O(n)'},
    {tex: 'P_j^{\\mathrm{old}} \\leftarrow f(C_j, A_j, k, D_j)', cost: 'O(n)'},
    {tex: 'k_i \\mathrel{+}= 1,\\ k_j \\mathrel{+}= 1', cost: 'O(1)'},
    {tex: 'C_i \\mathrel{+}= A_j^{\\mathrm{old}} + e_i,\\ \\ C_j \\mathrel{+}= A_i^{\\mathrm{old}} + e_j', cost: 'O(n)'},
    {tex: 'P_i,\\ P_j \\leftarrow f(\\text{new } C, A, k)', cost: 'O(n)'},
    {tex: 'R \\mathrel{+}= \\Delta P_i + \\Delta P_j', cost: 'O(n)'},
    {tex: 'A_{ij} = A_{ji} \\leftarrow 1', cost: 'O(1)'},
  ], -345, 47, 26);
  const desc = description(view, 110);

  const cur = createRef<Rect>();
  view.add(<Rect ref={cur} size={CS + 3} radius={3} stroke={C.mint} lineWidth={3} opacity={0} />);
  function* scan(cells: Rect[], pick: number) {
    cur().opacity(1);
    for (const c of cells) {
      cur().position(c.position());
      yield* waitFor(0.04);
    }
    cur().position(cells[pick].position());
  }

  // ======================================================== intro
  beat('A_intro');
  yield* title().opacity(1, 0.5);
  yield* all(M.g.opacity(1, 0.6), S.g.opacity(1, 0.6),
    desc.set('One GPU program per simulation runs all T steps. State: A (bool), C (int16), degrees, row sums R (fp64). P is not stored.'));
  yield* waitFor(1.8);

  // ======================================================== 1-4: draw the edge
  beat('B_draw');
  const i = cdfPick(R, rng.nextFloat());
  yield* all(S.focus(0), desc.set('Draw a row from the row sums: inverse CDF over n values.'));
  yield* scan(M.rCells, i);
  yield* M.rCells[i].lineWidth(3, 0.2);
  yield* waitFor(0.4);

  yield* all(S.focus(1), desc.set('Rebuild row i of P in registers, from row i of C and A, the degrees and row i of D.'),
    cur().opacity(0, 0.2), ...M.cells[i].map(c => c.lineWidth(2, 0.3)));
  yield* waitFor(0.9);

  const j = cdfPick(P[i], rng.nextFloat());
  yield* all(S.focus(2), desc.set('Draw a column within row i: inverse CDF over n values.'));
  yield* scan(M.cells[i], j);
  yield* all(M.cells[i][j].lineWidth(4, 0.2), M.cells[i][j].scale(1.35, 0.2, easeOutCubic));
  yield* waitFor(0.4);

  yield* all(S.focus(3), desc.set('Rebuild row j the same way: the old rows are needed for the row-sum update.'),
    cur().opacity(0, 0.2), ...M.cells[j].map(c => c.lineWidth(2, 0.3)));
  yield* waitFor(0.9);

  // ======================================================== 5-9: update
  beat('C_update');
  A[i][j] = A[j][i] = 1;
  const Pn = wiring(A, D, ETA, GAMMA);
  const Rn = rowSums(Pn);
  for (let a = 0; a < N; a++)
    for (let b = 0; b < N; b++)
      if (Pn[a][b] !== P[a][b] && a !== i && a !== j && b !== i && b !== j) throw new Error('toy: change outside rows/cols i, j');
  const cross: [number, number][] = [];
  for (let a = 0; a < N; a++) for (const b of [i, j]) { cross.push([a, b]); if (a !== i && a !== j) cross.push([b, a]); }

  yield* all(S.focus(4), desc.set('Both degrees go up by one.'),
    ...M.cells[i].map(c => c.lineWidth(0, 0.3)), ...M.cells[j].map(c => c.lineWidth(0, 0.3)),
    M.cells[i][j].scale(1, 0.3), M.cells[i][j].lineWidth(0, 0.3), M.rCells[i].lineWidth(0, 0.3));
  yield* waitFor(0.8);

  yield* all(S.focus(5), desc.set('Only rows and columns i and j of C change, by the old rows of A. Stored as rows; mirrored columns written only where they changed.'),
    ...cross.map(([a, b]) => M.cells[a][b].fill(C.mint, 0.3)));
  yield* waitFor(1.2);

  yield* all(S.focus(6), desc.set('New rows i and j of P, in registers.'),
    ...cross.map(([a, b]) => M.cells[a][b].fill(M.shade(Pn[a][b]), 0.5)));
  yield* waitFor(0.9);

  const rmax = Math.max(...R);
  yield* all(S.focus(7), desc.set('Patch every row sum by the change in columns i and j (fp64); rows i and j are re-summed exactly.'),
    ...M.rCells.map((c, a) => c.fill(C.mint, 0.2).to(ramp(0.15 + 0.85 * Math.min(1, Rn[a] / rmax)), 0.5)));
  yield* waitFor(0.9);

  yield* all(S.focus(8), desc.set('Finally write the new edge. It comes after the reductions over the old rows, so no thread reads it early.'),
    M.cells[i][j].fill(C.bg, 0.4), M.cells[j][i].fill(C.bg, 0.4));
  yield* waitFor(1.0);
  yield* desc.set('Every step is O(n), and a step never touches the full matrix.');
  yield* waitFor(2.2);
  beat('end');
});
