import {Rect, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, sequence, useRandom, waitFor} from '@motion-canvas/core';
import {C} from '../lib/theme';
import {beat, header} from '../lib/ui';
import {D, ETA, GAMMA, N, cdfPick, toyAdjacency} from '../lib/chain_toy';
import {wiring} from '../lib/gnm';
import {CS, counter, description, matrix, steps} from '../lib/maint';

// Maintainer video, scene 1: the reference step, as in the installed toolbox (gnm):
// generative_rules.py MatchingIndex computes torch.matmul(A^T, A) every step (dense, O(n^3));
// model.py normalises P over all n^2 entries and draws with torch.multinomial on the
// flattened n^2 matrix.
// Toy state: src/lib/chain_toy.ts (kernel conventions; cdfPick = kernel _sample_cdf).

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '1', 'The reference algorithm: one step');
  const rng = useRandom(31);

  const A = toyAdjacency(6, 5);
  const P = wiring(A, D, ETA, GAMMA);
  const M = matrix(view, P);
  const S = steps(view, [
    {tex: 'C \\leftarrow A^{\\top}A', cost: 'O(n³)'},
    {tex: 'K_{ij} \\leftarrow C_{ij} \\,/\\, \\tfrac12(k_i + k_j - 2A_{ij})', cost: 'O(n²)'},
    {tex: 'P \\leftarrow D^{\\eta} \\odot K^{\\gamma} + \\varepsilon', cost: 'O(n²)'},
    {tex: '(i,j) \\sim P \\,/\\, \\textstyle\\sum_{kl} P_{kl}\\ \\text{over all } n^2', cost: 'O(n²)'},
    {tex: 'A_{ij} = A_{ji} \\leftarrow 1', cost: 'O(1)'},
  ]);
  const desc = description(view, 60);
  const cnt = counter(view, 'entries touched this step', M.H + 60);

  // sweep every cell, row by row (the dense recomputation)
  function* sweepAll(d = 1.0) {
    yield* sequence(d / N, ...Array.from({length: N}, (_, i) =>
      all(...M.cells[i].map((c, j) => c.fill(C.muted, 0.08).to(M.shade(P[i][j]), 0.3)))));
  }
  let touched = 0;
  function* addTouched(k: number, d: number) {
    touched += k;
    yield* cnt.v(touched, d);
  }

  // ======================================================== intro
  beat('A_intro');
  yield* title().opacity(1, 0.5);
  yield* all(M.g.opacity(1, 0.6), S.g.opacity(1, 0.6), cnt.g.opacity(1, 0.6),
    desc.set('Each step adds one edge to the growing network. The toolbox rebuilds everything from scratch every step.'));
  yield* waitFor(1.5);

  // ======================================================== steps 1-3: dense recompute
  beat('B_recompute');
  const texts = [
    'Shared neighbours for every pair: a dense matrix product (torch.matmul), recomputed every step.',
    'Matching index for every pair, from the shared neighbours and degrees.',
    'Wiring weights for every pair: distance term times matching term, plus a small offset.',
  ];
  for (let k = 0; k < 3; k++) {
    yield* all(S.focus(k), desc.set(texts[k]));
    yield* all(sweepAll(1.0), addTouched(N * N, 1.0));
    yield* waitFor(0.5);
  }

  // ======================================================== step 4: inverse CDF over n^2
  beat('C_sample');
  yield* all(S.focus(3), desc.set('Normalise over all n² entries and draw one pair from the joint: one sample from the flattened n² matrix (torch.multinomial).'));
  const flat = P.flat();
  const k = cdfPick(flat, rng.nextFloat());
  const [pi, pj] = [Math.floor(k / N), k % N];
  const cur = createRef<Rect>();
  view.add(<Rect ref={cur} size={CS + 3} radius={3} stroke={C.mint} lineWidth={3} opacity={0} />);
  cur().opacity(1);
  for (let t = 0; t < N * N; t++) {
    cur().position(M.cells[Math.floor(t / N)][t % N].position());
    cnt.v(touched + t + 1);
    yield* waitFor(0.011);
  }
  touched += N * N;
  cur().position(M.cells[pi][pj].position());
  yield* all(M.cells[pi][pj].lineWidth(3, 0.2), M.cells[pi][pj].scale(1.35, 0.2));
  yield* waitFor(0.8);

  // ======================================================== step 5: add the edge
  beat('D_add');
  yield* all(S.focus(4), desc.set('Add the edge. Next step: start over, again over all n² pairs.'), cur().opacity(0, 0.2));
  yield* all(M.cells[pi][pj].fill(C.bg, 0.4), M.cells[pj][pi].fill(C.bg, 0.4),
    M.cells[pi][pj].scale(1, 0.4), M.cells[pj][pi].lineWidth(3, 0.4));
  cnt.txt('n³ + 3 × n²');
  yield* waitFor(0.6);
  yield* desc.set('Per step: one O(n³) product and O(n²) passes over every pair, repeated for every one of the T edges.');
  yield* waitFor(2.0);
  beat('end');
});
