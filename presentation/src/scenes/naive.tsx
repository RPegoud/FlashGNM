import {Circle, Latex, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutCubic,
  linear,
  sequence,
  tween,
  useLogger,
  useRandom,
  Vector2,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {Mat} from '../lib/gnm';
import {N, P_of, flatSample, seedA} from '../lib/naive_toy';
import bench from '../data/clip2_naive.json';

// ---------------------------------------------------------------- layout
const CS = 46; // matrix cell
const MX = -470;
const MY = 10;
const SW = 11; // strip cell width (N*N*SW = 1584 px)
const SH = 44;
const SX0 = -(N * N * SW) / 2;
const SY = 300;
const PY0 = 230; // cdf = 0
const PY1 = -290; // cdf = 1
const NN = N * N;

const gridPos = (e: number) => {
  const i = Math.floor(e / N), j = e % N;
  return new Vector2(MX + (j - (N - 1) / 2) * CS, MY + (i - (N - 1) / 2) * CS);
};
const stripPos = (e: number) => new Vector2(SX0 + (e + 0.5) * SW, SY);
const cdfY = (c: number) => PY0 + (PY1 - PY0) * c;

const fmt = (x: number) => x.toLocaleString('en-US');

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const rng = useRandom(11);
  const log = useLogger();

  const A: Mat = seedA();
  let P = P_of(A);

  const colorOf = (p: number, mx: number) => (p === 0 ? '#000000' : ramp(Math.sqrt(p / mx)));
  const maxOf = (M: Mat) => Math.max(...M.flat());

  const title = header(view, '02', 'Sampling over n² pairs');
  const cap = caption(view);

  // ---------------------------------------------------------------- matrix cells
  const matLayer = createRef<Node>();
  const fxLayer = createRef<Node>();
  view.add(
    <>
      <Node ref={matLayer} />
      <Node ref={fxLayer} />
    </>,
  );
  const cells: Rect[] = [];
  let mx = maxOf(P);
  for (let e = 0; e < NN; e++) {
    const i = Math.floor(e / N), j = e % N;
    const p = P[i][j];
    const r = (
      <Rect
        position={gridPos(e)}
        size={CS - 4}
        radius={4}
        fill={colorOf(p, mx)}
        stroke={C.dim}
        lineWidth={p === 0 ? 2 : 0}
        opacity={0}
      />
    ) as Rect;
    cells.push(r);
    matLayer().add(r);
  }
  // Dots marking existing edges (A_ij = 1).
  const dots = new Map<number, Circle>();
  function addDot(e: number, opacity = 0) {
    const d = (<Circle position={gridPos(e)} size={12} fill={C.deep} opacity={opacity} />) as Circle;
    matLayer().add(d);
    dots.set(e, d);
    return d;
  }
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (A[i][j]) addDot(i * N + j);

  // Row / column indices.
  const idx = createRef<Node>();
  view.add(<Node ref={idx} opacity={0} />);
  for (let k = 0; k < N; k++) {
    idx().add(
      <Txt text={`${k}`} fontFamily={MONO} fontSize={26} fill={C.muted}
        x={MX - ((N - 1) / 2) * CS - 48} y={MY + (k - (N - 1) / 2) * CS} />,
    );
    idx().add(
      <Txt text={`${k}`} fontFamily={MONO} fontSize={26} fill={C.muted}
        x={MX + (k - (N - 1) / 2) * CS} y={MY - ((N - 1) / 2) * CS - 44} />,
    );
  }
  idx().add(<Latex tex="i" fill={C.mint} fontSize={38} x={MX - ((N - 1) / 2) * CS - 92} y={MY} />);
  idx().add(<Latex tex="j" fill={C.mint} fontSize={38} x={MX} y={MY - ((N - 1) / 2) * CS - 90} />);

  // ======================================================== beat A: the P matrix
  beat('A_matrix');
  const eq = createRef<Latex>();
  const legend = createRef<Node>();
  const sym = createRef<Latex>();
  view.add(
    <Latex ref={eq} tex="P_{ij} = D_{ij}^{\,\eta}\, K_{ij}^{\,\gamma} + \varepsilon" fill={C.mint} fontSize={60}
      x={430} y={-250} opacity={0} />,
  );
  view.add(
    <Node ref={legend} opacity={0}>
      <Rect layout direction="row" gap={22} alignItems="center" x={120} y={-110} offset={[-1, 0]}>
        <Rect layout direction="row" width={150} height={34} radius={4} clip>
          {Array.from({length: 10}, (_, s) => <Rect width={15} height={34} fill={ramp(s / 9)} />)}
        </Rect>
        <Txt text="P for every pair (brighter = more likely)" fontFamily={FONT} fontSize={28} fill={C.mint} />
      </Rect>
      <Rect layout direction="row" gap={22} alignItems="center" x={120} y={-40} offset={[-1, 0]}>
        <Rect width={150} height={34} justifyContent="center" alignItems="center">
          <Rect width={34} height={34} radius={4} fill={'#000'} stroke={C.dim} lineWidth={2} justifyContent="center" alignItems="center">
            <Circle size={12} fill={C.deep} />
          </Rect>
        </Rect>
        <Txt text="existing edge: P = 0" fontFamily={FONT} fontSize={28} fill={C.mint} />
      </Rect>
      <Rect layout direction="row" gap={22} alignItems="center" x={120} y={30} offset={[-1, 0]}>
        <Rect width={150} height={34} justifyContent="center" alignItems="center">
          <Rect width={34} height={34} radius={4} fill={'#000'} stroke={C.dim} lineWidth={2} />
        </Rect>
        <Txt text="diagonal: P = 0" fontFamily={FONT} fontSize={28} fill={C.mint} />
      </Rect>
    </Node>,
  );

  const nTxt = createRef<Txt>();
  view.add(
    <Txt ref={nTxt} text={`toy network: n = ${N} regions, n² = ${NN} entries`} fontFamily={FONT}
      fontSize={30} fill={C.muted} x={120} y={140} offset={[-1, 0]} opacity={0} />,
  );

  // Symmetric pair to highlight: the largest entry.
  let hi = 0, hj = 1;
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (P[i][j] > P[hi][hj]) { hi = i; hj = j; }
  const boxA = createRef<Rect>();
  const boxB = createRef<Rect>();
  fxLayer().add(
    <>
      <Rect ref={boxA} position={gridPos(hi * N + hj)} size={CS + 2} radius={6} stroke={C.mint} lineWidth={4} opacity={0} />
      <Rect ref={boxB} position={gridPos(hj * N + hi)} size={CS + 2} radius={6} stroke={C.mint} lineWidth={4} opacity={0} />
    </>,
  );
  view.add(
    <Latex ref={sym} tex={`P_{${hi},${hj}} = P_{${hj},${hi}}\\quad\\text{(symmetric)}`} fill={C.mint} fontSize={40}
      x={120} y={230} offset={[-1, 0]} opacity={0} />,
  );

  yield* title().opacity(1, 0.6);
  yield* all(
    cap.set('The reference keeps a wiring weight P for every pair'),
    sequence(0.012, ...cells.map(c => c.opacity(1, 0.3))),
    idx().opacity(1, 0.6),
  );
  yield* all(...[...dots.values()].map(d => d.opacity(1, 0.4)));
  yield* all(eq().opacity(1, 0.6), eq().y(-260, 0.6, easeOutCubic));
  yield* legend().opacity(1, 0.6);
  yield* nTxt().opacity(1, 0.5);
  yield* all(boxA().opacity(1, 0.4), boxB().opacity(1, 0.4), sym().opacity(1, 0.4));
  yield* waitFor(1.6);

  // ======================================================== beat B: flatten + cumsum
  beat('B_flatten');
  const {flat, cdf} = flatSample(P, 0);
  // staircase of the normalised cumulative sum: jump at the left edge of each entry
  const stair: Vector2[] = [new Vector2(SX0, cdfY(0))];
  let prev = 0;
  for (let e = 0; e < NN; e++) {
    const x0 = SX0 + e * SW;
    stair.push(new Vector2(x0, cdfY(prev)), new Vector2(x0, cdfY(cdf[e])));
    prev = cdf[e];
  }
  stair.push(new Vector2(SX0 + NN * SW, cdfY(1)));

  const plot = createRef<Node>();
  const stairLine = createRef<Line>();
  const rowLabels = createRef<Node>();
  view.add(
    <Node ref={plot} opacity={0}>
      <Line points={[[SX0, PY1 - 20], [SX0, PY0], [SX0 + NN * SW + 20, PY0]]} stroke={C.muted} lineWidth={2} />
      <Line points={[[SX0 - 10, PY1], [SX0 + NN * SW, PY1]]} stroke={C.dim} lineWidth={2} lineDash={[8, 8]} />
      <Txt text="0" fontFamily={MONO} fontSize={26} fill={C.muted} x={SX0 - 20} y={PY0} offset={[1, 0]} />
      <Txt text="1" fontFamily={MONO} fontSize={26} fill={C.muted} x={SX0 - 20} y={PY1} offset={[1, 0]} />
      <Txt text="cumulative sum of P (normalised)" fontFamily={FONT} fontSize={28} fill={C.muted}
        x={SX0 + 20} y={PY1 - 10} offset={[-1, 1]} />
    </Node>,
  );
  view.add(<Line ref={stairLine} points={stair} stroke={C.green} lineWidth={4} end={0} />);
  view.add(<Node ref={rowLabels} opacity={0} />);
  for (let i = 0; i < N; i++) {
    rowLabels().add(
      <Txt text={`i=${i}`} fontFamily={MONO} fontSize={26} fill={C.muted} x={SX0 + (i + 0.5) * N * SW} y={SY + 50} />,
    );
    if (i > 0)
      rowLabels().add(<Line points={[[SX0 + i * N * SW, SY - 30], [SX0 + i * N * SW, SY + 30]]} stroke={C.muted} lineWidth={2} />);
  }

  yield* all(
    cap.set(`Flatten P into one strip of n² = ${NN} entries, row by row`),
    eq().opacity(0, 0.4),
    legend().opacity(0, 0.4),
    nTxt().opacity(0, 0.4),
    sym().opacity(0, 0.4),
    boxA().opacity(0, 0.4),
    boxB().opacity(0, 0.4),
    idx().opacity(0, 0.4),
    ...[...dots.values()].map(d => d.opacity(0, 0.4)),
  );
  yield* sequence(
    0.1,
    ...Array.from({length: N}, (_, i) =>
      all(
        ...Array.from({length: N}, (_, j) => {
          const e = i * N + j;
          return all(
            cells[e].position(stripPos(e), 0.8, easeInOutCubic),
            cells[e].size([SW, SH], 0.8, easeInOutCubic),
            cells[e].radius(0, 0.8),
            cells[e].lineWidth(0, 0.8),
          );
        }),
      ),
    ),
  );
  yield* rowLabels().opacity(1, 0.4);
  yield* waitFor(0.4);
  yield* cap.set('Build the cumulative sum: each entry adds a step as tall as its P');
  yield* plot().opacity(1, 0.4);
  yield* stairLine().end(1, 2.0, linear);
  yield* waitFor(1.4);

  // ======================================================== beat C: sample
  beat('C_sample');
  const u = rng.nextFloat();
  const {e: pick} = flatSample(P, u);
  const pi = Math.floor(pick / N), pj = pick % N;
  log.info(`SAMPLE u=${u.toFixed(3)} e=${pick} (${pi},${pj}) P=${flat[pick].toExponential(2)}`);
  const xHit = SX0 + pick * SW;
  const uLine = createRef<Line>();
  const dropLine = createRef<Line>();
  const uLab = createRef<Latex>();
  const mapEq = createRef<Latex>();
  const pickBox = createRef<Rect>();
  view.add(
    <>
      <Line ref={uLine} points={[[SX0, cdfY(u)], [xHit, cdfY(u)]]} stroke={C.mint} lineWidth={3} end={0} />
      <Line ref={dropLine} points={[[xHit + SW / 2, cdfY(u)], [xHit + SW / 2, SY - SH / 2 - 6]]} stroke={C.mint}
        lineWidth={3} lineDash={[8, 8]} end={0} />
      <Latex ref={uLab} tex={`u = ${u.toFixed(2)}`} fill={C.mint} fontSize={36} x={SX0 + 20} y={cdfY(u) - 30}
        offset={[-1, 0]} opacity={0} />
      <Rect ref={pickBox} position={stripPos(pick)} size={[SW + 10, SH + 14]} radius={4} stroke={C.mint} lineWidth={4} opacity={0} />
      <Latex ref={mapEq}
        tex={`e = ${pick}\\;\\Rightarrow\\; (i, j) = \\left(\\lfloor e / n \\rfloor,\\ e \\bmod n\\right) = (${pi}, ${pj})`}
        fill={C.green} fontSize={44} x={330} y={-390} opacity={0} />
    </>,
  );

  yield* all(cap.set('Draw u ~ U(0, 1) and find where the cumulative sum first passes it'), uLab().opacity(1, 0.4));
  yield* uLine().end(1, 0.9, easeInOutCubic);
  yield* dropLine().end(1, 0.5, easeOutCubic);
  yield* all(pickBox().opacity(1, 0.3), cells[pick].fill(C.mint, 0.3));
  yield* waitFor(0.6);
  yield* all(cap.set('Map the entry index back to (i, j)'), mapEq().opacity(1, 0.5));
  yield* waitFor(1.6);

  // back to the matrix, add the edge
  const mirror = pj * N + pi;
  yield* all(
    plot().opacity(0, 0.4),
    stairLine().opacity(0, 0.4),
    uLine().opacity(0, 0.4),
    dropLine().opacity(0, 0.4),
    uLab().opacity(0, 0.4),
    rowLabels().opacity(0, 0.4),
    pickBox().opacity(0, 0.3),
    mapEq().opacity(0, 0.4),
  );
  yield* all(
    ...cells.map((c, e) => {
      const p = flat[e];
      return all(
        c.position(gridPos(e), 0.9, easeInOutCubic),
        c.size(CS - 4, 0.9, easeInOutCubic),
        c.radius(4, 0.9),
        c.lineWidth(p === 0 ? 2 : 0, 0.9),
      );
    }),
    idx().opacity(1, 0.6),
    ...[...dots.values()].map(d => d.opacity(1, 0.6)),
  );
  const box1 = createRef<Rect>();
  const box2 = createRef<Rect>();
  fxLayer().add(
    <>
      <Rect ref={box1} position={gridPos(pick)} size={CS + 2} radius={6} stroke={C.green} lineWidth={4} opacity={0} />
      <Rect ref={box2} position={gridPos(mirror)} size={CS + 2} radius={6} stroke={C.green} lineWidth={4} opacity={0} />
    </>,
  );
  const addTxt = createRef<Latex>();
  view.add(
    <Latex ref={addTxt} tex={`A_{${pi},${pj}} = A_{${pj},${pi}} = 1`} fill={C.green} fontSize={52} x={430} y={-60} opacity={0} />,
  );
  A[pi][pj] = A[pj][pi] = 1;
  yield* all(
    cap.set(`Add the edge (${pi}, ${pj}): both entries become 0`),
    box1().opacity(1, 0.4),
    box2().opacity(1, 0.4),
    addTxt().opacity(1, 0.4),
  );
  yield* all(
    ...[pick, mirror].map(e => all(cells[e].fill('#000000', 0.5), cells[e].lineWidth(2, 0.5))),
    addDot(pick).opacity(1, 0.5),
    addDot(mirror).opacity(1, 0.5),
  );
  yield* waitFor(1.6);

  // ======================================================== beat D: every step, all n^2
  beat('D_repeat');
  yield* all(box1().opacity(0, 0.4), box2().opacity(0, 0.4), addTxt().opacity(0, 0.4));

  const panel = createRef<Node>();
  const rowBox: Rect[] = [];
  const stepTxt = createRef<Txt>();
  const rows: [string, string][] = [
    ['C = A^{\\top}\\!A', 'dense n × n matrix product'],
    ['K,\\ P', `recomputed on all n² = ${NN} entries`],
    ['\\text{sample}', `normalise and scan all ${NN} entries`],
  ];
  view.add(
    <Node ref={panel} opacity={0}>
      <Txt text="After every added edge, the reference:" fontFamily={FONT} fontSize={32} fill={C.muted}
        x={60} y={-250} offset={[-1, 0]} />
    </Node>,
  );
  rows.forEach(([tex, txt], k) => {
    const r = (
      <Rect layout direction="row" alignItems="center" gap={30} padding={[12, 20]} radius={10}
        stroke={C.green} lineWidth={0} x={40} y={-160 + k * 100} offset={[-1, 0]}>
        <Rect width={210} justifyContent="center">
          {tex.startsWith('\\text')
            ? <Txt text={tex.slice(6, -1)} fontFamily={FONT} fontSize={40} fill={C.mint} />
            : <Latex tex={tex} fill={C.mint} fontSize={44} />}
        </Rect>
        <Txt text={txt} fontFamily={FONT} fontSize={30} fill={C.mint} />
      </Rect>
    ) as Rect;
    rowBox.push(r);
    panel().add(r);
  });
  panel().add(
    <Txt ref={stepTxt} text="step 2 of T" fontFamily={MONO} fontSize={34} fill={C.green} x={60} y={190} offset={[-1, 0]} />,
  );

  const cursor = createRef<Rect>();
  fxLayer().add(<Rect ref={cursor} size={CS + 2} radius={6} stroke={C.mint} lineWidth={4} opacity={0} position={gridPos(0)} />);

  // A wave over all n^2 cells in row-major order (one tween, cheap per frame): cells near
  // the wave front dim briefly, cells behind it take their new colour.
  function* sweep(d: number, color: (e: number) => string | null) {
    const target = cells.map((_, e) => color(e));
    const W = 12; // width of the wave front in cells
    yield* tween(d, v => {
      const front = v * (NN + W);
      for (let e = 0; e < NN; e++) {
        const k = front - e;
        if (k < 0) continue;
        cells[e].opacity(k < W ? 0.35 + 0.65 * (k / W) : 1);
        const t = target[e];
        if (t && k >= W / 2) cells[e].fill(t);
      }
    }, linear);
  }
  function* step(stepNo: number, d: number, labels: boolean) {
    stepTxt().text(`step ${stepNo} of T`);
    // 1. C = A^T A
    if (labels) yield* rowBox[0].lineWidth(3, 0.3);
    yield* sweep(d, () => null);
    if (labels) yield* rowBox[0].lineWidth(0, 0.2);
    // 2. K, P on all n^2 entries
    P = P_of(A);
    mx = maxOf(P);
    const Pn = P;
    if (labels) yield* rowBox[1].lineWidth(3, 0.3);
    yield* sweep(d, e => colorOf(Pn[Math.floor(e / N)][e % N], mx));
    if (labels) yield* rowBox[1].lineWidth(0, 0.2);
    // 3. scan all n^2 entries, pick one
    const uu = rng.nextFloat();
    const {e} = flatSample(P, uu);
    const i = Math.floor(e / N), j = e % N;
    log.info(`STEP ${stepNo} u=${uu.toFixed(3)} e=${e} (${i},${j})`);
    if (labels) yield* rowBox[2].lineWidth(3, 0.3);
    cursor().position(gridPos(0));
    cursor().opacity(1);
    yield* tween(d, v => cursor().position(gridPos(Math.min(NN - 1, Math.floor(v * NN)))), linear);
    cursor().opacity(0);
    if (labels) yield* rowBox[2].lineWidth(0, 0.2);
    const m = j * N + i;
    A[i][j] = A[j][i] = 1;
    yield* all(
      ...[e, m].map(k => all(cells[k].fill(C.green, 0.25).to('#000000', 0.35), cells[k].lineWidth(2, 0.5))),
      addDot(e).opacity(1, 0.6),
      addDot(m).opacity(1, 0.6),
    );
  }

  yield* all(cap.set(`Then it starts over: all n² entries, every step`), panel().opacity(1, 0.5));
  yield* step(2, 1.2, true);
  yield* step(3, 0.7, false);
  yield* step(4, 0.5, false);
  yield* all(cap.set('Cost per step grows with n², and there are T steps'), stepTxt().text('step 5 … T', 0.3));
  yield* waitFor(1.6);

  // ======================================================== beat E: scale
  beat('E_scale');
  const n = bench.n;
  const T = bench.T;
  const [b1, bB] = bench.toolbox;
  const entries = n * n;
  const total = entries * T;
  const exp = Math.floor(Math.log10(total));
  const mant = total / Math.pow(10, exp);
  const simS = (b1.ms_per_step * T) / 1000;
  const batchH = (bB.ms_per_step * T) / 1000 / 3600;
  log.info(`SCALE n2=${entries} total=${total} sim_s=${simS} batch_h=${batchH}`);

  const scale = createRef<Node>();
  const L = -770;
  view.add(
    <Node ref={scale} opacity={0}>
      <Txt text="Real target: Griffa, n = 1,000 regions" fontFamily={FONT} fontSize={32} fill={C.muted}
        x={L} y={-310} offset={[-1, 0]} />
      <Txt text={`n² = ${fmt(entries)} entries per step`} fontFamily={FONT} fontSize={48} fill={C.mint}
        x={L} y={-230} offset={[-1, 0]} />
      <Txt text={`× T = ${fmt(T)} steps (mean edge count per subject)`} fontFamily={FONT} fontSize={48} fill={C.mint}
        x={L} y={-150} offset={[-1, 0]} />
      <Rect layout direction="row" alignItems="center" gap={14} x={L} y={-60} offset={[-1, 0]}>
        <Latex tex={`\\approx ${mant.toFixed(1)} \\times 10^{${exp}}`} fill={C.green} fontSize={60} />
        <Txt text="entries touched per simulation" fontFamily={FONT} fontSize={48} fill={C.mint} />
      </Rect>
    </Node>,
  );
  const measured = createRef<Node>();
  view.add(
    <Node ref={measured} opacity={0}>
      <Line points={[[L, 30], [770, 30]]} stroke={C.dim} lineWidth={2} />
      <Txt text={`Toolbox, unchanged, on one ${bench.gpu.replace('NVIDIA ', '').replace(' 80GB HBM3', '')}`}
        fontFamily={FONT} fontSize={32} fill={C.muted} x={L} y={90} offset={[-1, 0]} />
      <Txt text={`1 simulation:  ${b1.ms_per_step.toFixed(1)} ms / step  →  ≈ ${Math.round(simS)} s`}
        fontFamily={FONT} fontSize={44} fill={C.mint} x={L} y={165} offset={[-1, 0]} />
      <Txt text={`1,024 simulations (one batch):  ${(bB.ms_per_step / 1000).toFixed(2)} s / step  →  ≈ ${batchH.toFixed(1)} h`}
        fontFamily={FONT} fontSize={44} fill={C.mint} x={L} y={235} offset={[-1, 0]} />
      <Txt text={`Extrapolated from ${bench.probe_steps} timed steps · ${bench.source.split('/')[1]}`}
        fontFamily={FONT} fontSize={26} fill={C.muted} x={L} y={315} offset={[-1, 0]} />
    </Node>,
  );

  yield* all(
    cap.set('At real scale'),
    panel().opacity(0, 0.5),
    matLayer().opacity(0, 0.5),
    idx().opacity(0, 0.5),
  );
  yield* all(scale().opacity(1, 0.8));
  yield* waitFor(1.0);
  yield* all(cap.set('Measured: the reference spends this on every step'), measured().opacity(1, 0.8));
  yield* waitFor(2.0);

  // ======================================================== beat F: the question
  beat('F_question');
  const known = createRef<Node>();
  const q = createRef<Txt>();
  view.add(
    <Node ref={known} opacity={0}>
      <Txt text="Known, and not our claim:" fontFamily={FONT} fontSize={32} fill={C.muted} x={L} y={-280} offset={[-1, 0]} />
      <Txt text="update AᵀA incrementally after each edge (BCT; Oldham et al.)" fontFamily={FONT} fontSize={42}
        fill={C.mint} x={L} y={-215} offset={[-1, 0]} />
      <Txt text="It leaves the sampling step: normalise and scan all n² entries" fontFamily={FONT} fontSize={42}
        fill={C.mint} x={L} y={-110} offset={[-1, 0]} />
      <Txt text="Hypothesis H1: sampling, not the matching index, is the bottleneck" fontFamily={FONT} fontSize={32}
        fill={C.muted} x={L} y={-40} offset={[-1, 0]} />
    </Node>,
  );
  view.add(
    <Txt ref={q} text="Can we sample without touching all n² entries?" fontFamily={FONT} fontSize={60} fontWeight={600}
      fill={C.green} x={0} y={140} opacity={0} />,
  );
  yield* all(scale().opacity(0, 0.5), measured().opacity(0, 0.5), cap.set('What can be made cheaper?'));
  yield* known().opacity(1, 0.8);
  yield* waitFor(1.5);
  yield* all(q().opacity(1, 0.8), q().y(120, 0.8, easeOutCubic), cap.set('Next: sampling without the n² scan'));
  yield* waitFor(2.5);
  beat('end');
});
