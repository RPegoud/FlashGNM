import {Latex, Rect, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeOutCubic, useRandom, waitFor} from '@motion-canvas/core';
import {C} from '../lib/theme';
import {beat, header} from '../lib/ui';
import {D, ETA, GAMMA, N, cdfPick, rowSums, toyAdjacency} from '../lib/chain_toy';
import {wiring} from '../lib/gnm';
import {CS, PX, counter, description, matrix} from '../lib/maint';

// Maintainer video, scene 2: the product rule. P(i, j) = P(i) P(j | i) with row sums
// R_i = sum_j P_ij, so one draw over n^2 entries becomes two draws over n entries
// (src/kernel.py: _sample_cdf on R, then on row i). Toy: src/lib/chain_toy.ts.

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '2', 'Breaking down the joint: the product rule');
  const rng = useRandom(41);

  const A = toyAdjacency(6, 5);
  const P = wiring(A, D, ETA, GAMMA);
  const R = rowSums(P);
  const M = matrix(view, P, true, R);
  const desc = description(view, 170);
  const cnt = counter(view, 'entries read', M.H + 60);

  // derivation, top right
  const L = [createRef<Latex>(), createRef<Latex>(), createRef<Latex>(), createRef<Latex>()];
  const tex = [
    'P(i,j) = \\dfrac{P_{ij}}{\\sum_{k,l} P_{kl}}',
    'R_i = \\textstyle\\sum_j P_{ij}',
    '= \\dfrac{R_i}{\\sum_k R_k} \\cdot \\dfrac{P_{ij}}{R_i}',
    '= P(i)\\,\\cdot\\,P(j \\mid i)',
  ];
  const ys = [-310, -205, -95, 25];
  tex.forEach((t, k) => view.add(<Latex ref={L[k]} tex={t} fill={k === 3 ? C.green : C.mint} fontSize={k === 1 ? 34 : 42} x={PX + 10} y={ys[k]} offset={[-1, 0]} opacity={0} />));

  // ======================================================== derivation
  beat('A_derive');
  yield* title().opacity(1, 0.5);
  yield* all(M.g.opacity(1, 0.6), cnt.g.opacity(1, 0.6), L[0]().opacity(1, 0.6),
    desc.set('The reference draws from the joint over all n² pairs.'));
  yield* waitFor(1.0);
  yield* all(L[1]().opacity(1, 0.5), ...M.rCells.map(c => c.scale(1.15, 0.3).to(1, 0.3)),
    desc.set('Keep one extra vector: the row sums R of P.'));
  yield* waitFor(0.8);
  yield* all(L[2]().opacity(1, 0.5), desc.set('Multiply and divide by R_i.'));
  yield* waitFor(0.8);
  yield* all(L[3]().opacity(1, 0.5), L[3]().scale(1.05, 0.25).to(1, 0.25),
    desc.set('The product rule: pick a row, then a column within it. Exactly the same distribution.'));
  yield* waitFor(1.8);

  // ======================================================== two O(n) draws
  beat('B_draws');
  const cur = createRef<Rect>();
  view.add(<Rect ref={cur} size={CS + 3} radius={3} stroke={C.mint} lineWidth={3} opacity={0} />);
  const i = cdfPick(R, rng.nextFloat());
  const j = cdfPick(P[i], rng.nextFloat());
  yield* desc.set('Draw 1, i ~ P(i): an inverse CDF over the n row sums.');
  cur().opacity(1);
  for (let k = 0; k < N; k++) {
    cur().position(M.rCells[k].position());
    cnt.v(k + 1);
    yield* waitFor(0.05);
  }
  cur().position(M.rCells[i].position());
  yield* all(M.rCells[i].lineWidth(3, 0.2), ...M.cells[i].map(c => c.lineWidth(2, 0.3)));
  yield* waitFor(0.6);
  yield* desc.set('Draw 2, j ~ P(j | i): an inverse CDF over row i alone.');
  for (let k = 0; k < N; k++) {
    cur().position(M.cells[i][k].position());
    cnt.v(N + k + 1);
    yield* waitFor(0.05);
  }
  cur().position(M.cells[i][j].position());
  yield* all(M.cells[i][j].lineWidth(4, 0.2), M.cells[i][j].scale(1.35, 0.2, easeOutCubic));
  cnt.txt('2n');
  yield* desc.set('Two O(n) draws instead of one O(n²) draw: 2,000 reads instead of 1,000,000 at n = 1,000.');
  yield* waitFor(2.2);
  beat('end');
});
