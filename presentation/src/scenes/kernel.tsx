import {Latex, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  createSignal,
  easeInOutCubic,
  easeOutCubic,
  linear,
  sequence,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {Mat, distances, wiring, zeros} from '../lib/gnm';
import bench from '../data/clip5_bench.json';

// ------------------------------------------------------------------ facts
// Bytes per pair (i, j) of per-simulation state. Prototype (prep/kernel/kernel.py):
// A bool, C int32, P fp32. v2 (src/kernel.py): A bool, C int16, P not stored. SPEC §4.0.
const BYTES = {A: 1, C32: 4, C16: 2, P: 4};
const PROTO_BYTES = BYTES.A + BYTES.C32 + BYTES.P; // 9
const NOP_BYTES = BYTES.A + BYTES.C32; // 5
const V2_BYTES = BYTES.A + BYTES.C16; // 3
const pct = (b: number) => Math.round(100 * (1 - b / PROTO_BYTES));

// Measurements, prototype vs v2 in the same run (scripts/clip5_export.py).
const M = bench.configs.find(c => c.n === 2000 && c.B === 1024)!;

// Cache-line picture: 128-byte lines, a warp of 32 lanes, fp32 elements.
const LINE_B = 128;
const LANES = 32;
const ELEM_B = 4;
const NM = LINE_B / ELEM_B; // toy matrix side: one row = one line (32)
// Real scale: C int16 at n = 1000.
const NREAL = 1000;
const ROW_LINES_REAL = Math.ceil((NREAL * BYTES.C16) / LINE_B); // 16

// ------------------------------------------------------------------ toy GNM (beat C)
// 10 nodes on a jittered circle, a ring plus chords, so the matching index is non-zero
// for many pairs and P is far from uniform.
const NT = 10;
const POS: [number, number][] = [...Array(NT).keys()].map(k => {
  const a = (2 * Math.PI * k) / NT;
  const r = 1 + 0.25 * Math.sin(3.1 * k);
  return [r * Math.cos(a), r * Math.sin(a)];
});
const DT = distances(POS, 0.5);
const A_OLD: Mat = zeros(NT);
const link = (A: Mat, a: number, b: number) => (A[a][b] = A[b][a] = 1);
for (let k = 0; k < NT; k++) link(A_OLD, k, (k + 1) % NT);
for (const [a, b] of [[0, 3], [1, 5], [4, 7], [6, 9], [3, 8]]) link(A_OLD, a, b);
const EI = 2;
const EJ = 6;
const A_NEW = A_OLD.map(r => r.slice());
link(A_NEW, EI, EJ);
const P_OLD = wiring(A_OLD, DT, -2, 1.5);
const P_NEW = wiring(A_NEW, DT, -2, 1.5);
let PMAX = 0;
for (const P of [P_OLD, P_NEW]) for (const r of P) for (const x of r) PMAX = Math.max(PMAX, x);
const heat = (x: number) => (x === 0 ? '#0a0f0c' : ramp(0.15 + 0.85 * Math.sqrt(x / PMAX)));

// ------------------------------------------------------------------ text helpers
const T = (p: {[k: string]: any}) => <Txt fontFamily={FONT} fill={C.mint} fontSize={30} {...p} />;

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '05', 'Kernel optimisations');
  const cap = caption(view);

  // ================================================================ beat A: state
  beat('A_state');
  const BYTE = 150;
  const X0 = -(PROTO_BYTES * BYTE) / 2;
  const BY = -170;
  const BH = 130;

  const memGroup = createRef<Node>();
  const segA = createRef<Rect>();
  const segC = createRef<Rect>();
  const segP = createRef<Rect>();
  const cType = createRef<Txt>();
  const pText = createRef<Node>();
  const total = createRef<Txt>();
  const ghost = createRef<Rect>();
  const kicker = createRef<Txt>();

  const seg = (ref: any, name: string, sub: string, bytes: number, fill: string, ink: string, subRef?: any, inner?: any) => (
    <Rect ref={ref} width={bytes * BYTE} height={BH} fill={fill} stroke={C.bg} lineWidth={4} clip
      layout direction="column" justifyContent="center" alignItems="center" gap={6}>
      <Rect ref={inner} layout direction="column" alignItems="center" gap={6}>
        <Txt text={name} fontFamily={MONO} fontWeight={600} fontSize={44} fill={ink} />
        <Txt ref={subRef} text={sub} fontFamily={FONT} fontSize={26} fill={ink} />
      </Rect>
    </Rect>
  );

  const ticks = [];
  for (let b = 0; b <= PROTO_BYTES; b++) {
    const x = X0 + b * BYTE;
    ticks.push(<Line points={[[x, BY + BH / 2 + 14], [x, BY + BH / 2 + 26]]} stroke={C.muted} lineWidth={2} />);
    ticks.push(<Txt text={`${b}`} x={x} y={BY + BH / 2 + 50} fontFamily={MONO} fontSize={26} fill={C.muted} />);
  }

  view.add(
    <Node ref={memGroup} opacity={0}>
      <Txt ref={kicker} text="State per simulation, bytes per pair (i, j)" x={X0} y={BY - BH / 2 - 50}
        offset={[-1, 0]} fontFamily={FONT} fontSize={30} fill={C.muted} />
      <Rect ref={ghost} x={X0} y={BY} offset={[-1, 0]} width={PROTO_BYTES * BYTE} height={BH}
        stroke={C.dim} lineWidth={2} lineDash={[10, 8]} opacity={0} />
      <Rect layout direction="row" x={X0} y={BY} offset={[-1, 0]} height={BH}>
        {seg(segA, 'A', 'bool · 1 B', BYTES.A, ramp(0.42), C.mint)}
        {seg(segC, 'C', 'int32 · 4 B', BYTES.C32, ramp(0.62), C.mint, cType)}
        {seg(segP, 'P', 'fp32 · 4 B', BYTES.P, C.green, C.bg, undefined, pText)}
      </Rect>
      {ticks}
      <Txt text="bytes" x={X0 + PROTO_BYTES * BYTE + 30} y={BY + BH / 2 + 50} offset={[-1, 0]}
        fontFamily={FONT} fontSize={26} fill={C.muted} />
      <Txt ref={total} text={`${PROTO_BYTES} B`}
        x={() => X0 + segA().width() + segC().width() + segP().width() + 30} y={BY}
        offset={[-1, 0]} fontFamily={FONT} fontWeight={600} fontSize={44} fill={C.mint} />
    </Node>,
  );

  yield* title().opacity(1, 0.6);
  yield* all(
    cap.set('Prototype: 9 bytes per pair, and P is 4 of them'),
    memGroup().opacity(1, 0.6),
  );
  yield* waitFor(0.4);
  yield* segP().stroke(C.mint, 0.3);
  yield* waitFor(1.6);

  // ================================================================ beat B: shrink
  beat('B_shrink');
  yield* all(
    cap.set('P is only a cache: rebuild its rows from C, A, degrees and D'),
    ghost().opacity(1, 0.4),
    segP().stroke(C.bg, 0.3),
  );
  yield* all(pText().opacity(0, 0.3), segP().fill(C.dim, 0.3));
  yield* segP().width(0, 0.9, easeInOutCubic);
  segP().lineWidth(0);
  total().text(`${NOP_BYTES} B  −${pct(NOP_BYTES)}%`);
  yield* total().fill(C.green, 0.3);
  yield* waitFor(1.0);

  yield* all(cap.set('C fits in int16: shared neighbours ≤ max degree < n'), total().fill(C.mint, 0.3));
  yield* all(cType().opacity(0, 0.25), segC().width(BYTES.C16 * BYTE, 0.8, easeInOutCubic));
  cType().text(`int16 · ${BYTES.C16} B`);
  total().text(`${V2_BYTES} B  −${pct(V2_BYTES)}%`);
  yield* all(cType().opacity(1, 0.3), total().fill(C.green, 0.3));
  yield* waitFor(0.6);

  // Measured peak memory, same scale for both bars.
  const MW = 1100;
  const MY = 120;
  const gbP = M.proto.peak_mem_gb;
  const gbV = M.v2.peak_mem_gb;
  const meas = createRef<Node>();
  const barP = createRef<Rect>();
  const barV = createRef<Rect>();
  view.add(
    <Node ref={meas} opacity={0}>
      <Txt text={`Measured peak GPU memory · H100, n = ${M.n}, B = ${M.B}`} x={X0} y={MY}
        offset={[-1, 0]} fontFamily={FONT} fontSize={30} fill={C.muted} />
      <Txt text="prototype" x={X0} y={MY + 70} offset={[-1, 0]} fontFamily={FONT} fontSize={30} fill={C.mint} />
      <Txt text="v2" x={X0} y={MY + 140} offset={[-1, 0]} fontFamily={FONT} fontSize={30} fill={C.mint} />
      <Rect ref={barP} x={X0 + 170} y={MY + 70} offset={[-1, 0]} width={0} height={40} fill={C.dim} radius={4} />
      <Rect ref={barV} x={X0 + 170} y={MY + 140} offset={[-1, 0]} width={0} height={40} fill={C.green} radius={4} />
      <Txt text={`${gbP.toFixed(1)} GB`} x={() => X0 + 170 + barP().width() + 20} y={MY + 70} offset={[-1, 0]}
        fontFamily={MONO} fontSize={32} fill={C.mint} />
      <Txt text={`${gbV.toFixed(1)} GB`} x={() => X0 + 170 + barV().width() + 20} y={MY + 140} offset={[-1, 0]}
        fontFamily={MONO} fontSize={32} fill={C.green} />
    </Node>,
  );
  yield* all(meas().opacity(1, 0.4), cap.set(`Measured: peak memory falls ${(gbP / gbV).toFixed(1)}×, as the byte count predicts`));
  yield* all(barP().width(MW, 0.7, easeOutCubic), barV().width((MW * gbV) / gbP, 0.7, easeOutCubic));
  yield* waitFor(2.0);

  // ================================================================ beat C: the trade
  beat('C_trade');
  yield* all(memGroup().opacity(0, 0.5), meas().opacity(0, 0.5));
  memGroup().remove();
  meas().remove();

  // Left: prototype with stored P (toy 10-node network, real P values).
  const CELL = 30;
  const LX = -700; // matrix centre
  const LY = -90;
  const left = createRef<Node>();
  const pCells: Rect[][] = [];
  const mx0 = LX - (NT * CELL) / 2 + CELL / 2;
  const my0 = LY - (NT * CELL) / 2 + CELL / 2;
  const cellsNode = createRef<Node>();
  const overlay = createRef<Node>();
  const ops = [createRef<Txt>(), createRef<Txt>(), createRef<Txt>()];
  view.add(
    <Node ref={left} opacity={0}>
      <Txt text="Prototype: P stored" x={-870} y={-320} offset={[-1, 0]} fontFamily={FONT} fontWeight={600} fontSize={34} fill={C.mint} />
      <Node ref={cellsNode} />
      <Node ref={overlay} />
      <Txt text="P (toy, n = 10)" x={LX} y={LY + (NT * CELL) / 2 + 36} fontFamily={FONT} fontSize={26} fill={C.muted} />
      <Txt ref={ops[0]} text="load rows i, j of P" x={-500} y={-190} offset={[-1, 0]} fontFamily={FONT} fontSize={28} fill={C.muted} />
      <Txt ref={ops[1]} text="store rows i, j of P" x={-500} y={-110} offset={[-1, 0]} fontFamily={FONT} fontSize={28} fill={C.muted} />
      <Txt ref={ops[2]} text="store columns i, j of P (strided)" x={-500} y={-30} offset={[-1, 0]} fontFamily={FONT} fontSize={28} fill={C.muted} />
    </Node>,
  );
  for (let r = 0; r < NT; r++) {
    pCells.push([]);
    for (let c = 0; c < NT; c++) {
      const cell = (<Rect x={mx0 + c * CELL} y={my0 + r * CELL} size={CELL - 3} radius={3} fill={heat(P_OLD[r][c])} />) as Rect;
      cellsNode().add(cell);
      pCells[r].push(cell);
    }
  }
  const rowBox = (r: number) => (<Rect x={LX} y={my0 + r * CELL} width={NT * CELL + 6} height={CELL + 4} radius={5} stroke={C.green} lineWidth={3} opacity={0} />) as Rect;
  const colBox = (c: number) => (<Rect x={mx0 + c * CELL} y={LY} width={CELL + 4} height={NT * CELL + 6} radius={5} stroke={C.mint} lineWidth={3} lineDash={[6, 5]} opacity={0} />) as Rect;
  const rbI = rowBox(EI), rbJ = rowBox(EJ), cbI = colBox(EI), cbJ = colBox(EJ);
  overlay().add([rbI, rbJ, cbI, cbJ]);
  const rowLabel = (r: number, s: string) => (
    <Latex tex={s} fill={C.mint} fontSize={32} x={LX - (NT * CELL) / 2 - 30} y={my0 + r * CELL} />
  );
  overlay().add(rowLabel(EI, 'i'));
  overlay().add(rowLabel(EJ, 'j'));
  const colLabel = (c: number, s: string) => (
    <Latex tex={s} fill={C.mint} fontSize={32} x={mx0 + c * CELL} y={LY - (NT * CELL) / 2 - 28} />
  );
  overlay().add(colLabel(EI, 'i'));
  overlay().add(colLabel(EJ, 'j'));

  // Right: v2 recomputes four rows in registers.
  const right = createRef<Node>();
  const SX = 230; // strip left edge
  const SC = 30;
  const stripY = [-220, -165, -75, -20];
  const stripRows = [P_OLD[EI], P_OLD[EJ], P_NEW[EI], P_NEW[EJ]];
  const stripTex = ['P_{i}', 'P_{j}', "P'_{i}", "P'_{j}"];
  const strips: Rect[][] = [];
  view.add(
    <Node ref={right} opacity={0}>
      <Txt text="v2: P recomputed" x={100} y={-320} offset={[-1, 0]} fontFamily={FONT} fontWeight={600} fontSize={34} fill={C.mint} />
      <Txt text="old rows" x={SX + NT * SC + 30} y={(stripY[0] + stripY[1]) / 2} offset={[-1, 0]} fontFamily={FONT} fontSize={28} fill={C.muted} />
      <Txt text="new rows" x={SX + NT * SC + 30} y={(stripY[2] + stripY[3]) / 2} offset={[-1, 0]} fontFamily={FONT} fontSize={28} fill={C.muted} />
      <Txt text="in registers, from rows of C, A, D and the degrees" x={100} y={40} offset={[-1, 0]} fontFamily={FONT} fontSize={26} fill={C.muted} />
    </Node>,
  );
  for (let s = 0; s < 4; s++) {
    right().add(<Latex tex={stripTex[s]} fill={C.mint} fontSize={36} x={SX - 55} y={stripY[s]} />);
    strips.push([]);
    for (let c = 0; c < NT; c++) {
      const cell = (<Rect x={SX + c * SC + SC / 2} y={stripY[s]} size={SC - 3} radius={3} fill={'#0a0f0c'} stroke={C.dim} lineWidth={1} />) as Rect;
      right().add(cell);
      strips[s].push(cell);
    }
  }
  const divider = createRef<Line>();
  view.add(<Line ref={divider} points={[[0, -340], [0, 50]]} stroke={C.dim} lineWidth={2} end={0} />);

  yield* all(
    cap.set('Each step adds edge (i, j): rows and columns i, j of P change'),
    left().opacity(1, 0.5),
    divider().end(1, 0.5),
  );
  // prototype memory ops
  yield* all(ops[0]().fill(C.green, 0.3), rbI.opacity(1, 0.3), rbJ.opacity(1, 0.3));
  yield* waitFor(0.4);
  yield* all(
    ops[0]().fill(C.muted, 0.3),
    ops[1]().fill(C.green, 0.3),
    ...[EI, EJ].flatMap(r => pCells[r].map((cell, c) => cell.fill(heat(P_NEW[r][c]), 0.5))),
  );
  yield* waitFor(0.3);
  yield* all(ops[1]().fill(C.muted, 0.3), ops[2]().fill(C.green, 0.3), rbI.opacity(0, 0.3), rbJ.opacity(0, 0.3), cbI.opacity(1, 0.3), cbJ.opacity(1, 0.3));
  // strided: one element at a time, down the column
  yield* sequence(
    0.07,
    ...[...Array(NT).keys()].map(r =>
      all(pCells[r][EI].fill(heat(P_NEW[r][EI]), 0.25), pCells[r][EJ].fill(heat(P_NEW[r][EJ]), 0.25)),
    ),
  );
  yield* waitFor(0.6);

  // v2: compute four rows
  yield* all(cap.set('v2 stores no P: it computes four rows instead', 0.5), right().opacity(1, 0.5));
  for (let s = 0; s < 4; s++) {
    yield* sequence(0.03, ...strips[s].map((cell, c) => all(cell.fill(heat(stripRows[s][c]), 0.2), cell.stroke(C.bg, 0.2))));
  }
  yield* waitFor(0.3);

  const eqR = createRef<Latex>();
  const eqNote = createRef<Txt>();
  const measT = createRef<Txt>();
  view.add(
    <>
      <Latex ref={eqR} tex="R \mathrel{+}= (P'_{i} - P_{i}) + (P'_{j} - P_{j})" fill={C.mint} fontSize={52} x={0} y={160} opacity={0} />
      <Txt ref={eqNote} text="row sums need the old and the new rows" x={0} y={240} fontFamily={FONT} fontSize={28} fill={C.muted} opacity={0} />
      <Txt ref={measT} x={0} y={330} fontFamily={FONT} fontSize={30} fill={C.green} opacity={0}
        text={`Measured run time, all v2 changes vs prototype (n = ${M.n}, B = ${M.B}): ${M.proto.run_s.toFixed(2)} s → ${M.v2.run_s.toFixed(2)} s per batch`} />
    </>,
  );
  yield* all(eqR().opacity(1, 0.5), eqR().y(150, 0.5, easeOutCubic), eqNote().opacity(1, 0.5));
  yield* waitFor(0.8);
  yield* all(cap.set('Arithmetic is cheaper than the memory traffic it replaces'), measT().opacity(1, 0.5));
  yield* waitFor(2.2);

  // ================================================================ beat D: coalescing
  beat('D_row');
  yield* all(left().opacity(0, 0.5), right().opacity(0, 0.5), divider().opacity(0, 0.5), eqR().opacity(0, 0.5), eqNote().opacity(0, 0.5), measT().opacity(0, 0.5));
  left().remove();
  right().remove();
  divider().remove();
  eqR().remove();
  eqNote().remove();
  measT().remove();

  const MC = 13; // cell pitch
  const MX = -600;
  const MYc = -110;
  const gx0 = MX - (NM * MC) / 2 + MC / 2;
  const gy0 = MYc - (NM * MC) / 2 + MC / 2;
  const RI = 9; // the row / column read
  // Cheap grid (a few dozen nodes, not NM^2): background, grid lines, and highlight cells
  // only for row RI and column RI.
  const mat = createRef<Node>();
  view.add(<Node ref={mat} opacity={0} />);
  mat().add(<Rect x={MX} y={MYc} size={NM * MC} fill={'#0d1812'} />);
  for (let k = 1; k < NM; k++) {
    const o = -(NM * MC) / 2 + k * MC;
    mat().add(<Line points={[[MX + o, MYc - (NM * MC) / 2], [MX + o, MYc + (NM * MC) / 2]]} stroke={C.bg} lineWidth={2} />);
    mat().add(<Line points={[[MX - (NM * MC) / 2, MYc + o], [MX + (NM * MC) / 2, MYc + o]]} stroke={C.bg} lineWidth={2} />);
  }
  const rowCells: Rect[] = [];
  const colCells: Rect[] = [];
  for (let k = 0; k < NM; k++) {
    const rc = (<Rect x={gx0 + k * MC} y={gy0 + RI * MC} size={MC - 2} fill={C.green} opacity={0} />) as Rect;
    const cc = (<Rect x={gx0 + RI * MC} y={gy0 + k * MC} size={MC - 2} fill={C.green} opacity={0} />) as Rect;
    mat().add(rc);
    mat().add(cc);
    rowCells.push(rc);
    colCells.push(cc);
  }
  mat().add(<Txt text={`${NM} × ${NM} matrix M, fp32, row-major`} x={MX} y={MYc + (NM * MC) / 2 + 36} fontFamily={FONT} fontSize={26} fill={C.muted} />);
  mat().add(<Latex tex="i" fill={C.mint} fontSize={30} x={gx0 - 30} y={gy0 + RI * MC} />);
  mat().add(<Latex tex="i" fill={C.mint} fontSize={30} x={gx0 + RI * MC} y={gy0 - 30} />);

  // Linear memory strip: one block per cache line (= one row here).
  const BW = 48;
  const BG = 4;
  const SXL = -((NM * (BW + BG) - BG) / 2);
  const SY = 270;
  const strip = createRef<Node>();
  const blocks: Rect[] = [];
  const ticksL: Rect[] = [];
  view.add(
    <Node ref={strip} opacity={0}>
      <Txt text={`Memory: one ${LINE_B} B cache line per row (${NM} × ${ELEM_B} B)`} x={SXL} y={SY - 62}
        offset={[-1, 0]} fontFamily={FONT} fontSize={28} fill={C.muted} />
      <Txt text="line 0" x={SXL} y={SY + 50} offset={[-1, 0]} fontFamily={MONO} fontSize={26} fill={C.muted} />
      <Txt text={`line ${NM - 1}`} x={-SXL} y={SY + 50} offset={[1, 0]} fontFamily={MONO} fontSize={26} fill={C.muted} />
    </Node>,
  );
  strip().add(<Latex tex="i" fill={C.green} fontSize={30} x={SXL + RI * (BW + BG) + BW / 2} y={SY + 50} />);
  for (let b = 0; b < NM; b++) {
    const x = SXL + b * (BW + BG) + BW / 2;
    const blk = (<Rect x={x} y={SY} width={BW} height={46} radius={3} fill={C.faint} stroke={C.dim} lineWidth={2} />) as Rect;
    const tk = (<Rect x={SXL + b * (BW + BG) + ((RI + 0.5) * BW) / NM} y={SY} width={4} height={46} fill={C.green} opacity={0} />) as Rect;
    strip().add(blk);
    strip().add(tk);
    blocks.push(blk);
    ticksL.push(tk);
  }

  // Right side: counters.
  const RX = -230;
  const nRow = createSignal(0);
  const nCol = createSignal(0);
  const cnt = createRef<Node>();
  const rowLbl = createRef<Txt>();
  const colLbl = createRef<Txt>();
  const realNote = createRef<Txt>();
  view.add(
    <Node ref={cnt} opacity={0}>
      <Txt text={`A warp: ${LANES} lanes read ${LANES} values at once`} x={RX} y={-280} offset={[-1, 0]} fontFamily={FONT} fontSize={30} fill={C.muted} />
      <Txt ref={rowLbl} text="read row i" x={RX} y={-180} offset={[-1, 0]} fontFamily={FONT} fontSize={38} fill={C.mint} />
      <Txt text={() => `${Math.round(nRow())}`} x={RX + 520} y={-180} offset={[1, 0]} fontFamily={MONO} fontWeight={600} fontSize={64} fill={C.green} />
      <Txt text="cache line" x={RX + 545} y={-180} offset={[-1, 0]} fontFamily={FONT} fontSize={34} fill={C.mint} />
      <Txt ref={colLbl} text="read column i" x={RX} y={-70} offset={[-1, 0]} fontFamily={FONT} fontSize={38} fill={C.mint} opacity={0} />
      <Txt text={() => `${Math.round(nCol())}`} x={RX + 520} y={-70} offset={[1, 0]} fontFamily={MONO} fontWeight={600} fontSize={64} fill={C.green} opacity={() => colLbl().opacity()} />
      <Txt text="cache lines" x={RX + 545} y={-70} offset={[-1, 0]} fontFamily={FONT} fontSize={34} fill={C.mint} opacity={() => colLbl().opacity()} />
      <Txt ref={realNote} x={RX} y={40} offset={[-1, 0]} fontFamily={FONT} fontSize={28} fill={C.muted} opacity={0}
        text={`At n = ${NREAL.toLocaleString('en-US')} (int16 C): a row spans ≈ ${ROW_LINES_REAL} lines, a column ${NREAL.toLocaleString('en-US')}`} />
    </Node>,
  );

  yield* all(
    cap.set('GPU memory moves in cache lines: one row is one contiguous line'),
    mat().opacity(1, 0.5),
    strip().opacity(1, 0.5),
    cnt().opacity(1, 0.5),
  );
  // row read
  yield* all(
    ...rowCells.map(cell => cell.opacity(1, 0.4)),
    blocks[RI].fill(C.green, 0.4),
    blocks[RI].stroke(C.green, 0.4),
    nRow(1, 0.4),
  );
  yield* waitFor(1.8);

  // ================================================================ beat E: column read
  beat('E_column');
  yield* all(
    ...rowCells.map(cell => cell.opacity(0, 0.3)),
    blocks[RI].fill(C.faint, 0.3),
    blocks[RI].stroke(C.dim, 0.3),
    colLbl().opacity(1, 0.3),
    cap.set('A column hits a different line for every element'),
  );
  // column read: element by element, one line each
  yield* all(
    sequence(
      0.05,
      ...[...Array(NM).keys()].map(r =>
        all(colCells[r].opacity(1, 0.15), blocks[r].stroke(C.green, 0.15), ticksL[r].opacity(1, 0.15)),
      ),
    ),
    nCol(NM, 0.05 * (NM - 1) + 0.15, linear),
  );
  yield* realNote().opacity(1, 0.5);
  yield* waitFor(2.0);

  // ================================================================ beat F: symmetry
  beat('F_symmetry');
  const symEq = createRef<Latex>();
  const notes = createRef<Node>();
  const diag = createRef<Line>();
  mat().add(<Line ref={diag} points={[[MX - (NM * MC) / 2, MYc - (NM * MC) / 2], [MX + (NM * MC) / 2, MYc + (NM * MC) / 2]]}
    stroke={C.mint} lineWidth={2} lineDash={[8, 8]} end={0} />);
  view.add(
    <>
      <Latex ref={symEq} tex="M_{ki} = M_{ik}" fill={C.mint} fontSize={56} x={RX} y={-260} offset={[-1, 0]} opacity={0} />
      <Node ref={notes} opacity={0}>
        <Txt text="Reads: rows only, coalesced" x={RX} y={-140} offset={[-1, 0]} fontFamily={FONT} fontSize={32} fill={C.green} />
        <Txt text="Writes to columns i, j of C stay strided," x={RX} y={-70} offset={[-1, 0]} fontFamily={FONT} fontSize={32} fill={C.mint} />
        <Txt text="but only where C changed" x={RX} y={-25} offset={[-1, 0]} fontFamily={FONT} fontSize={32} fill={C.mint} />
        <Txt text="P has no column stores at all in v2" x={RX} y={45} offset={[-1, 0]} fontFamily={FONT} fontSize={32} fill={C.mint} />
      </Node>
    </>,
  );
  yield* all(
    cnt().opacity(0, 0.4),
    cap.set('A, C and D are symmetric: column i equals row i'),
  );
  cnt().remove();
  yield* all(symEq().opacity(1, 0.4), diag().end(1, 0.6));
  // fold column i onto row i across the diagonal
  yield* all(
    ...colCells.map((g, k) => all(g.x(gx0 + k * MC, 1.2, easeInOutCubic), g.y(gy0 + RI * MC, 1.2, easeInOutCubic))),
    ...blocks.map((b, k) => (k === RI ? all(b.fill(C.green, 1.2), b.stroke(C.green, 1.2)) : b.stroke(C.dim, 1.2))),
    ...ticksL.map(t => t.opacity(0, 1.2)),
  );
  yield* waitFor(0.6);
  yield* all(notes().opacity(1, 0.5), cap.set('So the kernel reads rows: 1 line instead of 32'));
  yield* waitFor(2.5);
  beat('end');
});
