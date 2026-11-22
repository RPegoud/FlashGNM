import {Latex, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutCubic,
  sequence,
  useLogger,
  useRandom,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {wiring} from '../lib/gnm';
import {D, ETA, GAMMA, N, cdfPick, rowSums, toyAdjacency} from '../lib/chain_toy';

// ---------------------------------------------------------------- toy state (computed)
// Clip 1's graph after its seed plus 8 model steps; P and R follow src/kernel.py conventions.
const A = toyAdjacency(8, 3);
const P = wiring(A, D, ETA, GAMMA);
const R = rowSums(P);
let PMAX = 0;
for (const r of P) for (const x of r) PMAX = Math.max(PMAX, x);
const RMAX = Math.max(...R);
const RSUM = R.reduce((s, x) => s + x, 0);

// ---------------------------------------------------------------- layout
const CS = 30; // heatmap pitch
const CELL = 27;
const HX = -430; // heatmap centre
const HY = 10;
const HALF = (N * CS) / 2; // 270
const LEFT = HX - HALF;
const TOP = HY - HALF;
const RX = LEFT + N * CS + 50; // R column centre
const SY = TOP + N * CS + 45; // column strip centre (below the heatmap)
const PX = 450; // right panel centre
const cx = (j: number) => LEFT + (j + 0.5) * CS;
const cy = (i: number) => TOP + (i + 0.5) * CS;

const ZERO = '#060a08'; // exact zeros: diagonal and existing edges
const pColor = (x: number) => (x === 0 ? ZERO : ramp(Math.sqrt(x / PMAX)));
const rColor = (x: number) => ramp(x / RMAX);

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const rng = useRandom(11);
  const log = useLogger();

  const title = header(view, '03', 'Two-stage sampling');
  const cap = caption(view);

  // ---------------------------------------------------------------- heatmap + R
  const heat = createRef<Node>();
  const cells: Rect[][] = [];
  view.add(<Node ref={heat} />);
  for (let i = 0; i < N; i++) {
    cells.push([]);
    for (let j = 0; j < N; j++) {
      const r = (
        <Rect x={cx(j)} y={cy(i)} size={CELL} radius={3} fill={pColor(P[i][j])} opacity={0} stroke={C.green} lineWidth={0} />
      ) as Rect;
      cells[i].push(r);
      heat().add(r);
    }
  }
  const rCells: Rect[] = [];
  for (let k = 0; k < N; k++) {
    const r = (<Rect x={RX} y={cy(k)} size={CELL} radius={3} fill={rColor(R[k])} opacity={0} stroke={C.green} lineWidth={0} />) as Rect;
    rCells.push(r);
    view.add(r);
  }
  const pLabel = createRef<Latex>();
  const rLabel = createRef<Latex>();
  const nLabel = createRef<Txt>();
  view.add(
    <>
      <Latex ref={pLabel} tex="P" fill={C.mint} fontSize={44} x={HX} y={TOP - 40} opacity={0} />
      <Latex ref={rLabel} tex="R" fill={C.mint} fontSize={44} x={RX} y={TOP - 40} opacity={0} />
      <Txt ref={nLabel} text={`toy: n = ${N}`} fontFamily={MONO} fontSize={26} fill={C.muted} x={LEFT} y={TOP - 40} offset={[-1, 0]} opacity={0} />
    </>,
  );

  // ======================================================== beat A: naive
  beat('A_naive');
  const naiveEq = createRef<Latex>();
  const naiveBox = createRef<Rect>();
  const naiveTxt = createRef<Txt>();
  const naiveCost = createRef<Latex>();
  view.add(
    <>
      <Rect ref={naiveBox} layout x={PX} y={-270} padding={[12, 20]} radius={10} stroke={C.green} lineWidth={0} opacity={0}>
        <Latex ref={naiveEq} tex={['P(i,j)', '=', '\\frac{P_{ij}}{\\sum_{k,l} P_{kl}}']} fill={C.mint} fontSize={60} />
      </Rect>
      <Txt ref={naiveTxt} text="one draw over all n² entries" fontFamily={FONT} fontSize={30} fill={C.muted} x={PX} y={-160} opacity={0} />
      <Latex ref={naiveCost} tex="O(n^2)\ \text{per step}" fill={C.green} fontSize={50} x={PX} y={-90} opacity={0} />
    </>,
  );

  yield* title().opacity(1, 0.6);
  yield* all(
    cap.set('Naive: draw one pair from all n² entries of P'),
    pLabel().opacity(1, 0.6),
    nLabel().opacity(1, 0.6),
    sequence(0.035, ...cells.map(row => all(...row.map(c => c.opacity(1, 0.4))))),
  );
  yield* all(naiveBox().opacity(1, 0.6), naiveBox().y(-280, 0.6, easeOutCubic));
  // A cursor visits every entry: what the naive sampler pays each step.
  const cursor = createRef<Rect>();
  view.add(<Rect ref={cursor} x={cx(0)} y={cy(0)} size={CS + 2} radius={4} stroke={C.mint} lineWidth={3} opacity={0} />);
  function* scan() {
    cursor().opacity(1);
    for (let i = 0; i < N; i++) {
      cursor().position([cx(0), cy(i)]);
      yield* cursor().x(cx(N - 1), 0.08);
    }
    yield* cursor().opacity(0, 0.2);
  }
  yield* all(naiveTxt().opacity(1, 0.5), scan());
  yield* naiveCost().opacity(1, 0.5);
  yield* waitFor(1.6);

  // ======================================================== beat B: row sums
  beat('B_rowsums');
  const rDef = createRef<Latex>();
  view.add(<Latex ref={rDef} tex="R_i = \textstyle\sum_j P_{ij}" fill={C.mint} fontSize={56} x={PX} y={-120} opacity={0} />);
  yield* all(
    naiveTxt().opacity(0, 0.4),
    naiveCost().opacity(0, 0.4),
    cap.set('Sum each row of P into a vector R'),
  );
  // A bar sweeps down the rows; each row's sum appears beside it.
  const sweep = createRef<Rect>();
  view.add(<Rect ref={sweep} x={(LEFT + RX + CS / 2) / 2} y={cy(0)} width={RX + CS / 2 - LEFT + 8} height={CS + 6} radius={6} stroke={C.mint} lineWidth={3} opacity={0} />);
  yield* sweep().opacity(1, 0.2);
  for (let k = 0; k < N; k++) {
    yield* all(sweep().y(cy(k), 0.1), rCells[k].opacity(1, 0.1));
  }
  yield* sweep().opacity(0, 0.3);
  yield* all(rLabel().opacity(1, 0.5), rDef().opacity(1, 0.6), rDef().y(-130, 0.6, easeOutCubic));
  yield* waitFor(1.6);

  // ======================================================== beat C: chain rule
  beat('C_chain');
  const chain = createRef<Latex>();
  const chainBox = createRef<Rect>();
  const sameTxt = createRef<Txt>();
  const G = '\\color{#7FEE64}';
  view.add(
    <>
      <Rect ref={chainBox} layout x={PX} y={20} padding={[12, 20]} radius={10} stroke={C.green} lineWidth={0}>
        <Latex ref={chain} tex={['P(i,j)', '=', 'P(i)', '\\,\\cdot\\,', 'P(j \\mid i)']} fill={C.mint} fontSize={56} opacity={0} />
      </Rect>
      <Txt ref={sameTxt} text="the same distribution, sampled in two steps" fontFamily={FONT} fontSize={30} fill={C.muted} x={PX} y={150} opacity={0} />
    </>,
  );
  yield* all(cap.set('Chain rule: pick a row, then a column within it'), chain().opacity(1, 0.6));
  yield* waitFor(1.2);
  yield* chain().tex(['P(i,j)', '=', '\\frac{R_i}{\\sum_k R_k}', '\\,\\cdot\\,', '\\frac{P_{ij}}{R_i}'], 1.0);
  yield* waitFor(0.8);
  yield* chain().tex(['P(i,j)', '=', `\\frac{${G}{R_i}}{\\sum_k R_k}`, '\\,\\cdot\\,', `\\frac{P_{ij}}{${G}{R_i}}`], 0.5);
  yield* waitFor(0.6);
  yield* chain().tex(['P(i,j)', '=', '\\frac{P_{ij}}{\\sum_k R_k}'], 1.0);
  yield* waitFor(0.5);
  yield* chain().tex(['P(i,j)', '=', '\\frac{P_{ij}}{\\sum_{k,l} P_{kl}}'], 1.0);
  yield* all(naiveBox().lineWidth(3, 0.4), chainBox().lineWidth(3, 0.4), sameTxt().opacity(1, 0.5));
  yield* waitFor(1.8);

  // ======================================================== beat D: sample
  beat('D_sample');
  const s1 = createRef<Latex>();
  const s1c = createRef<Latex>();
  const s2 = createRef<Latex>();
  const s2c = createRef<Latex>();
  const how1 = createRef<Txt>();
  const how2 = createRef<Txt>();
  const total = createRef<Latex>();
  const SX = PX - 200;
  view.add(
    <>
      <Latex ref={s1} tex="i \sim R" fill={C.mint} fontSize={56} x={SX} y={-250} offset={[-1, 0]} opacity={0} />
      <Latex ref={s1c} tex="O(n)" fill={C.green} fontSize={50} x={SX + 330} y={-250} offset={[-1, 0]} opacity={0} />
      <Latex ref={s2} tex="j \sim P_{i\,\cdot}" fill={C.mint} fontSize={56} x={SX} y={-140} offset={[-1, 0]} opacity={0} />
      <Latex ref={s2c} tex="O(n)" fill={C.green} fontSize={50} x={SX + 330} y={-140} offset={[-1, 0]} opacity={0} />
      <Txt ref={how1} text="inverse CDF: stack the weights," fontFamily={FONT} fontSize={30} fill={C.muted} x={PX} y={-20} opacity={0} />
      <Txt ref={how2} text="drop one uniform u (fp64 cumsum)" fontFamily={FONT} fontSize={30} fill={C.muted} x={PX} y={25} opacity={0} />
      <Latex ref={total} tex="O(n^2) \;\to\; O(n)\ \text{per step}" fill={C.green} fontSize={56} x={PX} y={170} opacity={0} />
    </>,
  );
  yield* all(
    naiveBox().opacity(0, 0.4),
    chainBox().opacity(0, 0.4),
    sameTxt().opacity(0, 0.4),
    rDef().opacity(0, 0.4),
    cap.set('Row: inverse CDF over R, O(n)'),
  );
  yield* all(s1().opacity(1, 0.5), s1c().opacity(1, 0.5), how1().opacity(1, 0.5), how2().opacity(1, 0.5));

  // Stage 1: stack R into a CDF, drop u.
  const u1 = rng.nextFloat();
  const u2 = rng.nextFloat();
  const si = cdfPick(R, u1);
  const sj = cdfPick(P[si], u2);
  log.info(`SAMPLE u1=${u1.toFixed(3)} i=${si} u2=${u2.toFixed(3)} j=${sj} P_ij/max=${(P[si][sj] / PMAX).toFixed(3)}`);

  const H = N * CS;
  let acc = 0;
  const segR = R.map(r => {
    const top = TOP + (acc / RSUM) * H;
    acc += r;
    const bot = TOP + (acc / RSUM) * H;
    return [top, bot];
  });
  yield* all(
    ...rCells.map((c, k) => all(c.y((segR[k][0] + segR[k][1]) / 2, 0.8, easeInOutCubic), c.height(Math.max(segR[k][1] - segR[k][0] - 2, 1), 0.8, easeInOutCubic), c.radius(1, 0.8))),
  );
  const uLine = createRef<Line>();
  const uLab = createRef<Latex>();
  const uy = TOP + u1 * H;
  view.add(
    <>
      <Line ref={uLine} points={[[RX - 30, uy], [RX + 30, uy]]} stroke={C.mint} lineWidth={4} end={0} />
      <Latex ref={uLab} tex="u" fill={C.mint} fontSize={36} x={RX + 52} y={uy} opacity={0} />
    </>,
  );
  yield* all(uLine().end(1, 0.5), uLab().opacity(1, 0.5));
  yield* rCells[si].lineWidth(3, 0.3);
  yield* waitFor(0.5);
  // Back to one cell per row; highlight the sampled row.
  const iLab = createRef<Latex>();
  const rowBox = createRef<Rect>();
  view.add(
    <>
      <Latex ref={iLab} tex="i" fill={C.green} fontSize={40} x={LEFT - 30} y={cy(si)} opacity={0} />
      <Rect ref={rowBox} x={(LEFT + RX + CS / 2) / 2} y={cy(si)} width={RX + CS / 2 - LEFT + 8} height={CS + 6} radius={6} stroke={C.green} lineWidth={3} opacity={0} />
    </>,
  );
  yield* all(
    uLine().opacity(0, 0.4),
    uLab().opacity(0, 0.4),
    ...rCells.map((c, k) => all(c.y(cy(k), 0.7, easeInOutCubic), c.height(CELL, 0.7, easeInOutCubic), c.radius(3, 0.7))),
  );
  yield* all(
    rCells[si].lineWidth(0, 0.3),
    rowBox().opacity(1, 0.5),
    iLab().opacity(1, 0.5),
    ...cells.flatMap((row, k) => (k === si ? [] : row.map(c => c.opacity(0.25, 0.5)))),
    ...rCells.map((c, k) => (k === si ? c.opacity(1, 0.5) : c.opacity(0.25, 0.5))),
  );
  yield* waitFor(1.2);

  // Stage 2: copy row i below the heatmap, stack it, drop u.
  yield* all(cap.set('Column: inverse CDF over row i only, O(n)'), s2().opacity(1, 0.5), s2c().opacity(1, 0.5));
  const strip: Rect[] = [];
  for (let j = 0; j < N; j++) {
    const r = (<Rect x={cx(j)} y={cy(si)} size={CELL} radius={3} fill={pColor(P[si][j])} stroke={C.green} lineWidth={0} opacity={0} />) as Rect;
    strip.push(r);
    view.add(r);
  }
  yield* all(...strip.map(r => all(r.opacity(1, 0.2), r.y(SY, 0.8, easeInOutCubic))));
  const W = N * CS;
  const rowSum = R[si];
  acc = 0;
  const segC = P[si].map(p => {
    const l = LEFT + (acc / rowSum) * W;
    acc += p;
    return [l, LEFT + (acc / rowSum) * W];
  });
  yield* all(
    ...strip.map((r, j) => all(r.x((segC[j][0] + segC[j][1]) / 2, 0.8, easeInOutCubic), r.width(Math.max(segC[j][1] - segC[j][0] - 2, 0), 0.8, easeInOutCubic), r.radius(1, 0.8))),
  );
  const vLine = createRef<Line>();
  const vLab = createRef<Latex>();
  const ux = LEFT + u2 * W;
  view.add(
    <>
      <Line ref={vLine} points={[[ux, SY - 28], [ux, SY + 28]]} stroke={C.mint} lineWidth={4} end={0} />
      <Latex ref={vLab} tex="u" fill={C.mint} fontSize={36} x={ux} y={SY + 50} opacity={0} />
    </>,
  );
  yield* all(vLine().end(1, 0.5), vLab().opacity(1, 0.5));
  yield* strip[sj].lineWidth(3, 0.3);
  yield* waitFor(0.5);
  const jLab = createRef<Latex>();
  view.add(<Latex ref={jLab} tex="j" fill={C.green} fontSize={40} x={cx(sj)} y={TOP - 40} opacity={0} />);
  yield* all(
    vLine().opacity(0, 0.4),
    vLab().opacity(0, 0.4),
    ...strip.map((r, j) => all(r.x(cx(j), 0.7, easeInOutCubic), r.width(CELL, 0.7, easeInOutCubic), r.radius(3, 0.7))),
  );
  yield* all(
    ...strip.map(r => r.opacity(0, 0.5)),
    cells[si][sj].lineWidth(4, 0.4),
    cells[si][sj].scale(1.3, 0.4),
    jLab().opacity(1, 0.5),
  );
  yield* total().opacity(1, 0.6);
  yield* waitFor(1.8);

  // ======================================================== beat E: keep R up to date
  beat('E_update');
  const A2 = A.map(r => r.slice());
  A2[si][sj] = A2[sj][si] = 1;
  const P2 = wiring(A2, D, ETA, GAMMA);
  const R2full = rowSums(P2);
  // Kernel update (src/kernel.py step 4): R += dP_i + dP_j (symmetric P, so column deltas),
  // then rows i and j summed exactly.
  const R2 = R.map((r, k) => r + (P2[si][k] - P[si][k]) + (P2[sj][k] - P[sj][k]));
  R2[si] = R2full[si];
  R2[sj] = R2full[sj];
  let outside = 0;
  let changed = 0;
  for (let k = 0; k < N; k++)
    for (let l = 0; l < N; l++) {
      if (P2[k][l] !== P[k][l]) changed++;
      if (k !== si && k !== sj && l !== si && l !== sj && P2[k][l] !== P[k][l]) outside++;
    }
  const err = Math.max(...R2.map((x, k) => Math.abs(x - R2full[k])));
  log.info(`UPDATE changed=${changed} outside_cross=${outside} max|R_delta-R_full|=${err.toExponential(2)}`);

  const u1t = createRef<Latex>();
  const u1n = createRef<Txt>();
  const u2t = createRef<Latex>();
  const u2n = createRef<Txt>();
  const u3n = createRef<Txt>();
  view.add(
    <>
      <Latex ref={u1t} tex="R_k \mathrel{+}= \Delta P_{ki} + \Delta P_{kj}" fill={C.mint} fontSize={54} x={PX} y={-250} opacity={0} />
      <Txt ref={u1n} text="every row: two deltas, accumulated in fp64" fontFamily={FONT} fontSize={30} fill={C.muted} x={PX} y={-170} opacity={0} />
      <Latex ref={u2t} tex="R_i = \sum_l P_{il}, \quad R_j = \sum_l P_{jl}" fill={C.mint} fontSize={54} x={PX} y={-40} opacity={0} />
      <Txt ref={u2n} text="rows i and j: re-summed exactly" fontFamily={FONT} fontSize={30} fill={C.muted} x={PX} y={40} opacity={0} />
      <Txt ref={u3n} text="O(n) per step; P itself is never stored" fontFamily={FONT} fontSize={30} fill={C.green} x={PX} y={170} opacity={0} />
    </>,
  );

  // The cross: rows and columns i and j.
  const cross: Rect[] = [];
  for (const k of [si, sj]) {
    cross.push((<Rect x={HX} y={cy(k)} width={N * CS + 6} height={CS + 4} radius={5} stroke={C.green} lineWidth={3} opacity={0} />) as Rect);
    cross.push((<Rect x={cx(k)} y={HY} width={CS + 4} height={N * CS + 6} radius={5} stroke={C.green} lineWidth={3} opacity={0} />) as Rect);
  }
  cross.forEach(r => view.add(r));

  yield* all(
    s1().opacity(0, 0.4), s1c().opacity(0, 0.4), s2().opacity(0, 0.4), s2c().opacity(0, 0.4),
    how1().opacity(0, 0.4), how2().opacity(0, 0.4), total().opacity(0, 0.4),
    rowBox().opacity(0, 0.4),
    cells[si][sj].lineWidth(0, 0.4), cells[si][sj].scale(1, 0.4),
    ...cells.flat().map(c => c.opacity(1, 0.4)),
    ...rCells.map(c => c.opacity(1, 0.4)),
    iLab().y(cy(si), 0.4),
    cap.set('Adding edge (i, j) changes only rows and columns i and j'),
  );
  // Label j on the left too (row j), i on top (column i).
  const jLab2 = createRef<Latex>();
  const iLab2 = createRef<Latex>();
  view.add(
    <>
      <Latex ref={jLab2} tex="j" fill={C.green} fontSize={40} x={LEFT - 30} y={cy(sj)} opacity={0} />
      <Latex ref={iLab2} tex="i" fill={C.green} fontSize={40} x={cx(si)} y={TOP - 40} opacity={0} />
    </>,
  );
  yield* all(...cross.map(r => r.opacity(1, 0.5)), jLab2().opacity(1, 0.5), iLab2().opacity(1, 0.5), pLabel().opacity(0, 0.3));
  yield* waitFor(0.4);
  const cellAnims = [];
  for (let k = 0; k < N; k++)
    for (let l = 0; l < N; l++)
      if (P2[k][l] !== P[k][l]) cellAnims.push(cells[k][l].fill(pColor(P2[k][l]), 0.8));
  yield* all(...cellAnims);
  yield* waitFor(0.8);

  yield* all(cap.set('So R is patched by deltas in O(n), no n² recompute'), u1t().opacity(1, 0.5), u1n().opacity(1, 0.5));
  yield* sequence(
    0.04,
    ...rCells.map((c, k) => (k === si || k === sj ? waitFor(0) : all(c.fill(rColor(R2[k]), 0.4), c.scale(1.2, 0.2).to(1, 0.2)))),
  );
  yield* all(u2t().opacity(1, 0.5), u2n().opacity(1, 0.5));
  yield* all(
    ...[si, sj].map(k => all(rCells[k].fill(rColor(R2[k]), 0.5), rCells[k].lineWidth(3, 0.5))),
  );
  yield* u3n().opacity(1, 0.5);
  yield* waitFor(2.5);
  beat('end');
});
