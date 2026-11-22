import {Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, createSignal, easeInOutCubic, easeOutCubic, useRandom, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {TileGrid} from '../lib/tiles';
import {D, ETA, GAMMA, N, cdfPick, rowSums, toyAdjacency} from '../lib/chain_toy';
import {wiring} from '../lib/gnm';
import talk from '../data/talk.json';

// Talk clip B: why the GPU makes this possible.
//  A sampling: inverse CDF over all n^2 pairs vs two-stage (row sums, then one row)
//  B update:   recompute every pair vs rows/columns i, j + row-sum deltas
//  C tricks:   what makes each simulation small and fast (src/kernel.py)
//  D side by side: how many simulations fit on one GPU, each with its own (eta, gamma)
// Toy values computed with src/lib/chain_toy.ts (kernel conventions); real-scale numbers
// from src/data/talk.json.

const P = talk.parallel;
const comma = (x: number) => Math.round(x).toLocaleString('en-US');
const CS = 21; // cell size
const H = (N * CS) / 2;
const LX = -500, RX = 380, GY = -10;
const PER_CELL = 0.012; // same scan speed on both sides

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '2', 'Why the GPU makes it possible');
  const cap = caption(view);
  const rng = useRandom(11);

  // ---------------------------------------------------------------- toy state
  const A = toyAdjacency(6, 5);
  let Pm = wiring(A, D, ETA, GAMMA);
  let R = rowSums(Pm);
  const pmax = Math.max(...Pm.flat());
  const shade = (v: number) => ramp(0.12 + 0.88 * Math.sqrt(v / pmax));

  const mkGrid = (cx: number) => {
    const cells: Rect[][] = [];
    const g = (<Node opacity={0} />) as Node;
    for (let i = 0; i < N; i++) {
      cells.push([]);
      for (let j = 0; j < N; j++) {
        const r = (<Rect x={cx + (j - (N - 1) / 2) * CS} y={GY + (i - (N - 1) / 2) * CS} size={CS - 3} radius={2} fill={shade(Pm[i][j])} lineWidth={0} stroke={C.mint} />) as Rect;
        cells[i].push(r);
        g.add(r);
      }
    }
    view.add(g);
    return {g, cells};
  };
  const L = mkGrid(LX);
  const Rg = mkGrid(RX);
  const rmax = Math.max(...R);
  const RCX = RX + H + 34;
  const rCells: Rect[] = R.map((v, i) => {
    const r = (<Rect x={RCX} y={GY + (i - (N - 1) / 2) * CS} size={CS - 3} radius={2} fill={ramp(0.15 + 0.85 * (v / rmax))} lineWidth={0} stroke={C.mint} opacity={0} />) as Rect;
    view.add(r);
    return r;
  });

  const labels = createRef<Node>();
  view.add(
    <Node ref={labels} opacity={0}>
      <Txt text="Reference toolbox" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.muted} x={LX} y={GY - H - 100} />
      <Txt text="Our GPU kernel" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.green} x={RX + 17} y={GY - H - 100} />
      <Txt text="P" fontFamily={FONT} fontSize={28} fill={C.muted} x={LX} y={GY - H - 30} />
      <Txt text="P" fontFamily={FONT} fontSize={28} fill={C.muted} x={RX} y={GY - H - 30} />
    </Node>,
  );
  const rLab = createRef<Txt>();
  view.add(<Txt ref={rLab} text="row sums" fontFamily={FONT} fontSize={22} fill={C.muted} x={RCX} y={GY - H - 30} opacity={0} />);

  // sub-titles and counters (changed per beat)
  const subL = createRef<Txt>(), subR = createRef<Txt>();
  const cntL = createSignal(0), cntR = createSignal(0);
  const cnt = createRef<Node>();
  const realL = createRef<Txt>(), realR = createRef<Txt>();
  const cntLabel = createRef<Txt>();
  view.add(
    <>
      <Txt ref={subL} text="" fontFamily={FONT} fontSize={28} fill={C.muted} x={LX} y={GY - H - 62} />
      <Txt ref={subR} text="" fontFamily={FONT} fontSize={28} fill={C.mint} x={RX + 17} y={GY - H - 62} />
      <Node ref={cnt} opacity={0}>
        <Txt ref={cntLabel} text="entries read" fontFamily={FONT} fontSize={24} fill={C.muted} x={(LX + RX) / 2} y={GY + H + 40} />
        <Txt text={() => comma(cntL())} fontFamily={MONO} fontSize={52} fill={C.muted} x={LX} y={GY + H + 60} />
        <Txt text={() => comma(cntR())} fontFamily={MONO} fontSize={52} fill={C.green} x={RX + 17} y={GY + H + 60} />
        <Txt ref={realL} text="" fontFamily={MONO} fontSize={26} fill={C.muted} x={LX} y={GY + H + 112} />
        <Txt ref={realR} text="" fontFamily={MONO} fontSize={26} fill={C.green} x={RX + 17} y={GY + H + 112} />
      </Node>
    </>,
  );

  // ======================================================== beat A: sampling
  beat('A_sampling');
  subL().text('one draw over all n² pairs');
  subR().text('pick a row, then a column');
  realL().text(`n = ${comma(P.n)}: ${comma(P.n * P.n)}`);
  realR().text(`n = ${comma(P.n)}: ${comma(2 * P.n)}`);
  yield* title().opacity(1, 0.6);
  yield* all(L.g.opacity(1, 0.5), Rg.g.opacity(1, 0.5), labels().opacity(1, 0.5), subL().opacity(1, 0.5), subR().opacity(1, 0.5),
    ...rCells.map(c => c.opacity(1, 0.5)), rLab().opacity(1, 0.5),
    cap.set('Step 1: draw the next connection in proportion to P'));
  yield* cnt().opacity(1, 0.4);

  // draws (computed exactly as the two methods would)
  const flat = Pm.flat();
  const kL = cdfPick(flat, rng.nextFloat());
  const [li, lj] = [Math.floor(kL / N), kL % N];
  const si = cdfPick(R, rng.nextFloat());
  const sj = cdfPick(Pm[si], rng.nextFloat());

  // left scans every entry up to the pick (then the rest is still summed: full n^2 read)
  const cursorL = createRef<Rect>(), cursorR = createRef<Rect>();
  view.add(<Rect ref={cursorL} size={CS + 3} radius={3} stroke={C.mint} lineWidth={3} opacity={0} />);
  view.add(<Rect ref={cursorR} size={CS + 3} radius={3} stroke={C.mint} lineWidth={3} opacity={0} />);
  const posL = (i: number, j: number) => L.cells[i][j].position();
  yield* all(
    (function* () {
      cursorL().opacity(1);
      for (let t = 0; t < N * N; t++) {
        const i = Math.floor(t / N), j = t % N;
        cursorL().position(posL(i, j));
        cntL(t + 1);
        yield* waitFor(PER_CELL);
      }
      cursorL().position(posL(li, lj));
      L.cells[li][lj].lineWidth(3);
      L.cells[li][lj].scale(1.35);
    })(),
    (function* () {
      cursorR().opacity(1);
      for (let i = 0; i < N; i++) {
        cursorR().position(rCells[i].position());
        cntR(i + 1);
        yield* waitFor(PER_CELL);
      }
      cursorR().position(rCells[si].position());
      rCells[si].lineWidth(3);
      for (let j = 0; j < N; j++) {
        cursorR().position(Rg.cells[si][j].position());
        cntR(N + j + 1);
        yield* waitFor(PER_CELL);
      }
      cursorR().position(Rg.cells[si][sj].position());
      Rg.cells[si][sj].lineWidth(3);
      Rg.cells[si][sj].scale(1.35);
    })(),
  );
  cursorL().opacity(0);
  cursorR().opacity(0);
  yield* cap.set('Same distribution, a tiny fraction of the reads');
  yield* waitFor(2.0);

  // ======================================================== beat B: update
  beat('B_update');
  const [ai, aj] = [si, sj];
  A[ai][aj] = A[aj][ai] = 1;
  const Pn = wiring(A, D, ETA, GAMMA);
  const Rn = rowSums(Pn);
  const changed: [number, number][] = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (Pn[i][j] !== Pm[i][j]) changed.push([i, j]);
  const outside = changed.filter(([i, j]) => i !== ai && i !== aj && j !== ai && j !== aj).length;
  if (outside > 0) throw new Error(`toy: ${outside} changed entries outside rows/cols i, j`);
  L.cells[li][lj].lineWidth(0); L.cells[li][lj].scale(1);
  Rg.cells[si][sj].lineWidth(0); Rg.cells[si][sj].scale(1);
  rCells[si].lineWidth(0);
  cntLabel().text('entries recomputed');
  cntL(0); cntR(0);
  realL().text(`n = ${comma(P.n)}: ${comma(P.n * P.n)}`);
  realR().text(`n = ${comma(P.n)}: ~${comma(P.rows_per_step * P.n)}`);
  yield* all(subL().opacity(0, 0.3), subR().opacity(0, 0.3));
  subL().text('recompute every pair');
  subR().text('update rows & columns i, j only');
  yield* all(subL().opacity(1, 0.3), subR().opacity(1, 0.3),
    cap.set('Step 2: add the connection and update P'));

  const lit = new Set<string>();
  for (let i = 0; i < N; i++) for (const j of [ai, aj]) { lit.add(`${i},${j}`); lit.add(`${j},${i}`); }
  const litCells = [...lit].map(s => s.split(',').map(Number) as [number, number]);
  yield* all(
    // left: every cell, row by row
    (function* () {
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) L.cells[i][j].fill(C.muted); // every pair redone
        cntL((i + 1) * N);
        yield* waitFor(PER_CELL * N);
      }
    })(),
    // right: the cross through i and j, then the row-sum deltas
    (function* () {
      for (const [i, j] of litCells) Rg.cells[i][j].fill(C.mint); // only the cross through i, j
      cntR(litCells.length);
      yield* waitFor(PER_CELL * N);
      for (let i = 0; i < N; i++) rCells[i].fill(C.mint); // every row sum patched by a delta
      cntR(litCells.length + N);
    })(),
  );
  Pm = Pn; R = Rn;
  yield* cap.set('Only rows and columns i and j change, so the row sums are patched, not recomputed');
  yield* waitFor(2.0);

  // ======================================================== beat C: tricks
  beat('C_tricks');
  yield* all(L.g.opacity(0, 0.5), Rg.g.opacity(0, 0.5), labels().opacity(0, 0.5), subL().opacity(0, 0.5), subR().opacity(0, 0.5),
    cnt().opacity(0, 0.5), rLab().opacity(0, 0.5), ...rCells.map(c => c.opacity(0, 0.5)));
  const tricks = [
    ['One fused kernel per simulation', 'a single GPU program runs all T steps, no round trips'],
    ['P is recomputed, never stored', 'rows rebuilt from C, A, degrees and distances when needed'],
    ['Narrow data types', 'bool A, int16 C: 3 bytes per pair instead of 9'],
    ['Coalesced loads', 'matrices are symmetric, so read rows (contiguous), never columns'],
    ['Exact, drift-free sampling', 'row sums in fp64, patched by deltas each step'],
  ];
  const trickNodes = tricks.map(([h, s], k) => {
    const y = -300 + k * 120;
    const g = (
      <Node opacity={0} x={-30}>
        <Rect x={-780} y={y} width={14} height={14} radius={3} fill={C.green} />
        <Txt text={h} fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.mint} x={-750} y={y - 20} offset={[-1, 0]} />
        <Txt text={s} fontFamily={FONT} fontSize={28} fill={C.muted} x={-750} y={y + 24} offset={[-1, 0]} />
      </Node>
    ) as Node;
    view.add(g);
    return g;
  });
  yield* cap.set('What makes each simulation small and fast');
  for (const g of trickNodes) {
    yield* all(g.opacity(1, 0.4), g.x(0, 0.4, easeOutCubic));
    yield* waitFor(0.9);
  }
  yield* waitFor(1.5);

  // ======================================================== beat D: side by side
  beat('D_side_by_side');
  yield* all(...trickNodes.map(g => g.opacity(0, 0.5)));
  const BOX = 420;
  const TX = -470, TY = -20;
  const tl = new TileGrid({x: TX, y: TY});
  tl.k = Math.round(Math.sqrt(P.toolbox.B));
  tl.size = BOX;
  tl.base = C.dim;
  const tr = new TileGrid({x: -TX, y: TY});
  tr.k = Math.round(Math.sqrt(P.ours_max.B));
  tr.size = BOX;
  tr.base = C.deep;
  view.add(tl);
  view.add(tr);
  const mem = createRef<Node>();
  view.add(
    <Node ref={mem} opacity={0}>
      <Txt text="Reference toolbox" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.muted} x={TX} y={TY - BOX / 2 - 60} />
      <Txt text="Our GPU kernel" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.green} x={-TX} y={TY - BOX / 2 - 60} />
      <Rect x={TX} y={TY} size={BOX + 16} stroke={C.dim} lineWidth={2} radius={6} />
      <Rect x={-TX} y={TY} size={BOX + 16} stroke={C.dim} lineWidth={2} radius={6} />
      <Txt text={`${comma(P.toolbox.B)} simulations`} fontFamily={MONO} fontSize={40} fill={C.muted} x={TX} y={TY + BOX / 2 + 60} />
      <Txt text={`${P.toolbox.peak_gb.toFixed(1)} GB · one (η, γ) for all`} fontFamily={FONT} fontSize={26} fill={C.muted} x={TX} y={TY + BOX / 2 + 106} />
      <Txt text={`${comma(P.ours_max.B)} simulations`} fontFamily={MONO} fontSize={40} fill={C.green} x={-TX} y={TY + BOX / 2 + 60} />
      <Txt text={`${P.ours_max.peak_gb.toFixed(1)} GB · each its own (η, γ)`} fontFamily={FONT} fontSize={26} fill={C.mint} x={-TX} y={TY + BOX / 2 + 106} />
    </Node>,
  );
  yield* all(mem().opacity(1, 0.5), cap.set('Small simulations fit side by side on one GPU'));
  yield* all(tl.reveal(1, 1.6, easeInOutCubic), tr.reveal(1, 3.0, easeInOutCubic));
  yield* all(tr.sweep(1, 1.2, easeInOutCubic),
    cap.set(`Same memory, ${Math.round(P.ours_max.B / P.toolbox.B)}× more simulations, a whole parameter sweep in one launch (n = ${comma(P.n)})`));
  yield* waitFor(2.5);
  beat('end');
});
