import {Latex, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutCubic,
  linear,
  sequence,
  useLogger,
  useRandom,
  Vector2,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {Mat, zeros} from '../lib/gnm';
import {D, N, State, Step, initState, rDrift, step} from '../lib/matrices_toy';

// ---------------------------------------------------------------- toy model
// Seed network (11 edges on 8 nodes, degrees 2 to 4): enough triangles and open triads
// that the matching index, hence P, is far from uniform.
const EDGES: [number, number][] = [
  [0, 1], [0, 6], [1, 2], [1, 7], [2, 3], [2, 7], [3, 4], [4, 5], [4, 7], [5, 6], [6, 7],
];
const A0: Mat = zeros(N);
for (const [a, b] of EDGES) A0[a][b] = A0[b][a] = 1;
const ETA = -2;
const GAMMA = 1;

// ---------------------------------------------------------------- layout (step view)
const S = 46; // cell
const GY = -60; // grid centre y
const TOP = GY - (N / 2) * S;
const BOT = GY + (N / 2) * S;
const AX = -600; // A grid centre x
const CX = -130; // C grid centre x
const KX = 150; // degree vector x
const RX0 = 215; // R bars start x
const RLEN = 170; // max R bar length
const PANEL = 430; // left edge of the right panel
const rowY = (r: number) => TOP + S / 2 + r * S;
const colX = (cx: number, c: number) => cx - (N / 2) * S + S / 2 + c * S;
const STRIP_Y = {oldPi: 178, oldPj: 230, Pi: 306, Pj: 358};

// ---------------------------------------------------------------- sim view
interface SimView {
  node: Node;
  aCells: Rect[][];
  cCells: Rect[][];
  cTxt: Txt[][];
  kCells: Rect[];
  kTxt: Txt[];
  rBars: Rect[];
}

const aFill = (v: number, diag: boolean) => (diag ? '#050806' : v ? C.deep : C.faint);
const cFill = '#0b140f';

function simView(s: State, rMax: number, light = false, need: (r: number, c: number) => boolean = () => false): SimView {
  const node = new Node({});
  const aCells: Rect[][] = [];
  const cCells: Rect[][] = [];
  const cTxt: Txt[][] = [];
  const kCells: Rect[] = [];
  const kTxt: Txt[] = [];
  const rBars: Rect[] = [];
  for (let r = 0; r < N; r++) {
    aCells.push([]);
    cCells.push([]);
    cTxt.push([]);
    for (let c = 0; c < N; c++) {
      const a = (
        <Rect x={colX(AX, c)} y={rowY(r)} size={S - 6} radius={4} fill={aFill(s.A[r][c], r === c)} />
      ) as Rect;
      aCells[r].push(a);
      node.add(a);
      const t = (
        <Txt text={s.C[r][c] ? `${s.C[r][c]}` : ''} fontFamily={MONO} fontSize={26} fill={C.mint} />
      ) as Txt;
      const cc = (
        <Rect x={colX(CX, c)} y={rowY(r)} size={S - 6} radius={4} fill={light ? ramp(s.C[r][c] / 5) : cFill} stroke={r === c ? C.dim : '#00000000'} lineWidth={2}>
          {light || (!s.C[r][c] && !need(r, c)) ? null : t}
        </Rect>
      ) as Rect;
      cCells[r].push(cc);
      cTxt[r].push(t);
      node.add(cc);
    }
    const kt = (<Txt text={`${s.deg[r]}`} fontFamily={MONO} fontSize={26} fill={C.mint} />) as Txt;
    const kc = (
      <Rect x={KX} y={rowY(r)} size={S - 6} radius={4} fill={light ? ramp(s.deg[r] / 5) : cFill}>
        {light ? null : kt}
      </Rect>
    ) as Rect;
    kCells.push(kc);
    kTxt.push(kt);
    node.add(kc);
    const rb = (
      <Rect x={RX0} y={rowY(r)} offset={[-1, 0]} height={S - 16} width={(s.R[r] / rMax) * RLEN} radius={3} fill={C.deep} />
    ) as Rect;
    rBars.push(rb);
    node.add(rb);
  }
  // R axis
  node.add(<Rect x={RX0 - 2} y={GY} width={2} height={N * S} fill={C.dim} />);
  // Labels: symbol, then dtype and shape.
  const lab = (x: number, tex: string, dtype: string) => {
    node.add(<Latex tex={tex} fill={C.mint} fontSize={40} x={x} y={TOP - 74} />);
    node.add(<Txt text={dtype} fontFamily={MONO} fontSize={26} fill={C.muted} x={x} y={TOP - 30} />);
  };
  lab(AX, 'A', 'bool  n×n');
  lab(CX, 'C = A^{\\top}\\!A', 'int16  n×n');
  lab(KX, 'k', 'int16  n');
  lab(RX0 + RLEN / 2, 'R', 'fp64  n');
  return {node, aCells, cCells, cTxt, kCells, kTxt, rBars};
}

// Bounding box of the sim view (for the batch cards).
const BOX = {x0: AX - (N / 2) * S, x1: RX0 + RLEN, y0: TOP - 100, y1: BOT};
const BOX_C = new Vector2((BOX.x0 + BOX.x1) / 2, (BOX.y0 + BOX.y1) / 2);
const CARD_SCALE = 0.5;
const CARD_W = (BOX.x1 - BOX.x0) * CARD_SCALE + 48;
const CARD_H = (BOX.y1 - BOX.y0) * CARD_SCALE + 24 + 58;
const cardCentre = (k: number) => new Vector2(-470 + 44 * k, 96 - 66 * k);
// Group position that puts the scaled content inside card k, below its header strip.
const groupPosFor = (k: number) => cardCentre(k).add(new Vector2(0, 24)).sub(BOX_C.scale(CARD_SCALE));

const fmt1 = (x: number) => (x < 0 ? '−' : '') + Math.abs(x).toFixed(1);

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const log = useLogger();

  // ------------------------------------------------------------ the step to show
  // Uniforms from a seeded RNG; take the first seed whose step touches rows with >= 2
  // neighbours each, so the C update is visible (bounded search).
  let st: Step | null = null;
  let sim: State = initState(A0, ETA, GAMMA);
  const init: State = initState(A0, ETA, GAMMA);
  for (let seed = 1; seed <= 60 && !st; seed++) {
    const r = useRandom(seed);
    const s = initState(A0, ETA, GAMMA);
    const cand = step(s, ETA, GAMMA, r.nextFloat(), r.nextFloat());
    if (cand.degOld[cand.i] >= 2 && cand.degOld[cand.j] >= 2 && cand.degOld[cand.i] + cand.degOld[cand.j] >= 6) {
      st = cand;
      sim = s;
      log.info(`SEED ${seed} i=${cand.i} j=${cand.j} drift=${rDrift(s, ETA, GAMMA)}`);
    }
  }
  if (!st) {
    const r = useRandom(1);
    sim = initState(A0, ETA, GAMMA);
    st = step(sim, ETA, GAMMA, r.nextFloat(), r.nextFloat());
  }
  const {i, j} = st;
  const rMax = Math.max(...init.R, ...st.Rnew) * 1.05;
  const pMax = Math.max(...st.oldPi, ...st.oldPj, ...st.Pi, ...st.Pj);

  const title = header(view, '04', 'The state of one simulation');
  const cap = caption(view);

  // ------------------------------------------------------------ layers
  const stackLayer = createRef<Node>();
  const simLayer = createRef<Node>();
  const fxLayer = createRef<Node>();
  const uiLayer = createRef<Node>();
  view.add(
    <>
      <Node ref={stackLayer} />
      <Node ref={simLayer} />
      <Node ref={fxLayer} />
      <Node ref={uiLayer} />
    </>,
  );

  const V = simView(init, rMax, false, (r, c) => r === i || r === j || c === i || c === j);
  simLayer().add(V.node);
  const allCells = [...V.aCells.flat(), ...V.cCells.flat(), ...V.kCells];
  allCells.forEach(c => c.scale(0));
  V.rBars.forEach(b => b.scale([0, 1]));
  const labels = V.node.children().filter(n => n instanceof Latex || n instanceof Txt);
  labels.forEach(l => l.opacity(0));

  // ======================================================== beat A: the state
  beat('A_state');
  yield* title().opacity(1, 0.6);
  yield* cap.set('One simulation keeps four arrays');
  const rowSeq = (cells: Rect[][]) =>
    sequence(0.05, ...cells.map(row => all(...row.map(c => c.scale(1, 0.35, easeOutCubic)))));
  yield* all(labels[0].opacity(1, 0.4), labels[1].opacity(1, 0.4), rowSeq(V.aCells));
  yield* all(labels[2].opacity(1, 0.4), labels[3].opacity(1, 0.4), rowSeq(V.cCells));
  yield* all(
    labels[4].opacity(1, 0.4), labels[5].opacity(1, 0.4),
    sequence(0.05, ...V.kCells.map(c => c.scale(1, 0.35, easeOutCubic))),
    labels[6].opacity(1, 0.4), labels[7].opacity(1, 0.4),
    sequence(0.05, ...V.rBars.map(b => b.scale([1, 1], 0.35, easeOutCubic))),
  );
  yield* waitFor(0.3);

  // Panel: what each array means.
  const legend = createRef<Node>();
  const legendRow = (y: number, tex: string, text: string) => (
    <>
      <Latex tex={tex} fill={C.mint} fontSize={36} x={PANEL} y={y} offset={[-1, 0]} />
      <Txt text={text} fontFamily={FONT} fontSize={28} fill={C.muted} x={PANEL + 90} y={y} offset={[-1, 0]} />
    </>
  );
  uiLayer().add(
    <Node ref={legend} opacity={0}>
      {legendRow(-250, 'A', 'adjacency')}
      {legendRow(-190, 'C', 'shared neighbours')}
      {legendRow(-130, 'k', 'degrees (= diag C)')}
      {legendRow(-70, 'R', 'row sums of P')}
      <Rect x={PANEL} y={-10} width={390} height={2} fill={C.dim} offset={[-1, 0]} />
      <Txt text="1 + 2 = 3 bytes per pair" fontFamily={FONT} fontWeight={500} fontSize={32} fill={C.mint} x={PANEL} y={45} offset={[-1, 0]} />
      <Txt text="K and P are not stored:" fontFamily={FONT} fontSize={28} fill={C.muted} x={PANEL} y={105} offset={[-1, 0]} />
      <Txt text="rows recomputed when needed" fontFamily={FONT} fontSize={28} fill={C.muted} x={PANEL} y={145} offset={[-1, 0]} />
    </Node>,
  );
  yield* legend().opacity(1, 0.6);
  // Diagonal of C is the degree vector.
  const diagBox = V.cCells.map((row, r) => row[r]);
  yield* all(
    ...diagBox.map(c => c.stroke(C.green, 0.4)),
    ...V.kCells.map(c => c.stroke(C.green, 0.4)),
    ...V.kCells.map(c => c.lineWidth(2, 0.4)),
  );
  yield* waitFor(0.8);
  yield* all(
    ...diagBox.map(c => c.stroke(C.dim, 0.4)),
    ...V.kCells.map(c => c.lineWidth(0, 0.4)),
    cap.set('A bool, C int16: 3 bytes per node pair'),
  );
  yield* waitFor(1.6);

  // ======================================================== beat B: the batch
  beat('B_batch');
  // Other simulations of the batch: own (eta, gamma), a few steps from the same seed.
  const rngB = useRandom(42);
  const others: {eta: number; gamma: number; s: State}[] = [];
  for (let b = 1; b <= 3; b++) {
    const eta = Math.round((-3 + 2.5 * rngB.nextFloat()) * 10) / 10;
    const gamma = Math.round((0.2 + 1.3 * rngB.nextFloat()) * 10) / 10;
    const s = initState(A0, eta, gamma);
    for (let t = 0; t < 2 + b; t++) step(s, eta, gamma, rngB.nextFloat(), rngB.nextFloat());
    others.push({eta, gamma, s});
  }
  const pidName = ['0', '1', '2', 'B−1'];
  const cards: Rect[] = [];
  const params = [{eta: ETA, gamma: GAMMA}, ...others];
  for (let k = 3; k >= 0; k--) {
    const c = cardCentre(k);
    const card = (
      <Rect x={c.x} y={c.y} width={CARD_W} height={CARD_H} radius={14} fill={'#030604'} stroke={k === 0 ? C.deep : C.dim} lineWidth={2} opacity={0}>
        <Txt
          text={`pid ${pidName[k]}   η = ${fmt1(params[k].eta)}   γ = ${params[k].gamma.toFixed(1)}`}
          fontFamily={MONO} fontSize={26} fill={k === 0 ? C.green : C.muted}
          x={-CARD_W / 2 + 22} y={-CARD_H / 2 + 30} offset={[-1, 0]}
        />
      </Rect>
    ) as Rect;
    if (k > 0) {
      const ov = simView(others[k - 1].s, Math.max(rMax, ...others[k - 1].s.R) * 1.05, true);
      ov.node.scale(CARD_SCALE);
      ov.node.position(groupPosFor(k).sub(c));
      card.add(ov.node);
    }
    stackLayer().add(card);
    cards.unshift(card);
  }

  // Right side: shared D, per-simulation (eta, gamma), the launch.
  const batchUI = createRef<Node>();
  const DS = 30;
  const DX = 300, DY = -110;
  let dMax = 0;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) if (r !== c) dMax = Math.max(dMax, D[r][c]);
  const dCells: Rect[] = [];
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++)
      dCells.push(
        (
          <Rect
            x={DX + (c - (N - 1) / 2) * DS} y={DY + (r - (N - 1) / 2) * DS} size={DS - 4} radius={3}
            fill={r === c ? '#050806' : ramp(1 - D[r][c] / dMax)}
          />
        ) as Rect,
      );
  const progress = createRef<Rect>();
  uiLayer().add(
    <Node ref={batchUI} opacity={0}>
      {dCells}
      <Latex tex="D" fill={C.mint} fontSize={40} x={DX} y={DY - (N / 2) * DS - 74} />
      <Txt text="fp32  n×n" fontFamily={MONO} fontSize={26} fill={C.muted} x={DX} y={DY - (N / 2) * DS - 30} />
      <Txt text="one copy, shared" fontFamily={FONT} fontSize={28} fill={C.muted} x={DX} y={DY + (N / 2) * DS + 36} />
      <Latex tex="(\eta_b,\ \gamma_b)" fill={C.mint} fontSize={40} x={680} y={DY - 40} />
      <Txt text="fp32, one pair" fontFamily={MONO} fontSize={26} fill={C.muted} x={680} y={DY + 14} />
      <Txt text="per simulation" fontFamily={MONO} fontSize={26} fill={C.muted} x={680} y={DY + 50} />
      <Txt text="binary_gnm_kernel[(B,)]" fontFamily={MONO} fontSize={28} fill={C.mint} x={150} y={150} offset={[-1, 0]} />
      <Txt text="one program per simulation" fontFamily={FONT} fontSize={28} fill={C.muted} x={150} y={192} offset={[-1, 0]} />
      <Txt text="for t in tl.range(T):" fontFamily={MONO} fontSize={28} fill={C.mint} x={150} y={258} offset={[-1, 0]} />
      <Txt text="all T steps in one launch" fontFamily={FONT} fontSize={28} fill={C.muted} x={150} y={300} offset={[-1, 0]} />
      <Rect x={560} y={258} width={300} height={10} radius={5} fill={C.faint} offset={[-1, 0]} />
      <Rect ref={progress} x={560} y={258} width={0} height={10} radius={5} fill={C.green} offset={[-1, 0]} />
    </Node>,
  );

  yield* all(legend().opacity(0, 0.4), cap.set('A batch: B simulations, each with its own (η, γ)'));
  yield* all(
    V.node.scale(CARD_SCALE, 0.9, easeInOutCubic),
    V.node.position(groupPosFor(0), 0.9, easeInOutCubic),
    cards[0].opacity(1, 0.9),
  );
  yield* sequence(0.15, ...cards.slice(1).map(c => c.opacity(1, 0.4)));
  yield* batchUI().opacity(1, 0.6);
  yield* waitFor(0.8);
  yield* cap.set('One GPU program per simulation, persistent over all T steps');
  yield* progress().width(300, 2.2, linear);
  yield* waitFor(1.6);

  // ======================================================== beat C: pick the edge
  beat('C_pick');
  yield* all(
    batchUI().opacity(0, 0.5),
    ...cards.map(c => c.opacity(0, 0.5)),
    V.node.scale(1, 0.9, easeInOutCubic),
    V.node.position([0, 0], 0.9, easeInOutCubic),
  );

  // Algorithm listing, in the kernel's order.
  const LINES: string[] = [
    'i \\sim R',
    'P_i^{\\mathrm{old}} \\leftarrow C_i,\\ A_i,\\ k,\\ D_i',
    'j \\sim P_i^{\\mathrm{old}}',
    'P_j^{\\mathrm{old}} \\leftarrow C_j,\\ A_j,\\ k,\\ D_j',
    'k_i \\mathrel{+}= 1,\\ \\ k_j \\mathrel{+}= 1',
    'C_i \\mathrel{+}= A_j^{\\mathrm{old}} + e_i',
    'C_j \\mathrel{+}= A_i^{\\mathrm{old}} + e_j',
    'P_i,\\ P_j \\leftarrow \\text{new } C,\\ A,\\ k',
    'R \\mathrel{+}= \\Delta P_i + \\Delta P_j',
    'A_{ij} = A_{ji} = 1',
  ];
  const LY0 = -300, LDY = 46;
  const lineNodes: Latex[] = [];
  const algo = createRef<Node>();
  const cursor = createRef<Rect>();
  uiLayer().add(
    <Node ref={algo} opacity={0}>
      <Rect ref={cursor} x={PANEL - 16} y={LY0} width={450} height={LDY - 4} radius={8} fill={C.faint} stroke={C.deep} lineWidth={2} offset={[-1, 0]} opacity={0} />
    </Node>,
  );
  LINES.forEach((tex, n) => {
    algo().add(<Txt text={`${n + 1}`} fontFamily={MONO} fontSize={26} fill={C.muted} x={PANEL} y={LY0 + n * LDY} offset={[-1, 0]} />);
    const l = (<Latex tex={tex} fill={C.mint} fontSize={32} x={PANEL + 44} y={LY0 + n * LDY} offset={[-1, 0]} opacity={0.3} />) as Latex;
    lineNodes.push(l);
    algo().add(l);
  });
  // The row function, below the listing.
  algo().add(
    <>
      <Txt text="each row of P, on the fly:" fontFamily={FONT} fontSize={26} fill={C.muted} x={PANEL} y={LY0 + 10 * LDY + 34} offset={[-1, 0]} />
      <Latex tex="P_x = D_x^{\,\eta} \odot K_x^{\,\gamma} + 10^{-6}" fill={C.mint} fontSize={30} x={PANEL} y={LY0 + 10 * LDY + 88} offset={[-1, 0]} />
      <Latex tex="K_x = C_x \,/\, \tfrac{1}{2}(k_x + k - 2A_x)" fill={C.mint} fontSize={30} x={PANEL} y={LY0 + 10 * LDY + 146} offset={[-1, 0]} />
    </>,
  );
  function* focus(n: number) {
    yield* all(
      cursor().opacity(1, 0.3),
      cursor().y(LY0 + n * LDY, 0.4, easeInOutCubic),
      ...lineNodes.map((l, m) => l.opacity(m === n ? 1 : m < n ? 0.6 : 0.3, 0.4)),
    );
  }

  // Row/column outlines and i, j markers.
  const outline = (x: number, y: number, w: number, h: number) => {
    const r = (<Rect x={x} y={y} width={w} height={h} radius={6} stroke={C.green} lineWidth={0} />) as Rect;
    fxLayer().add(r);
    return r;
  };
  const rowBox = (cx: number, r: number) => outline(cx, rowY(r), N * S + 4, S);
  const colBox = (cx: number, c: number) => outline(colX(cx, c), GY, S, N * S + 4);
  const marker = (tex: string, x: number, y: number) => {
    const m = (<Latex tex={tex} fill={C.green} fontSize={36} x={x} y={y} opacity={0} />) as Latex;
    fxLayer().add(m);
    return m;
  };
  const aRowI = rowBox(AX, i), cRowI = rowBox(CX, i);
  const aRowJ = rowBox(AX, j), cRowJ = rowBox(CX, j);
  const mI = [marker('i', AX - (N / 2) * S - 30, rowY(i)), marker('i', CX - (N / 2) * S - 28, rowY(i))];
  const mJ = [marker('j', AX - (N / 2) * S - 30, rowY(j)), marker('j', CX - (N / 2) * S - 28, rowY(j))];
  const kBox = outline(KX, GY, S, N * S + 4);

  // P strips (registers only: never written to memory).
  const pColor = (p: number) => ramp(Math.sqrt(p / pMax));
  function strip(vals: number[], y: number, tex: string, x: number) {
    const cells = vals.map((p, c) =>
      (
        <Rect
          x={colX(CX, c)} y={y} size={S - 6} radius={4}
          fill={p === 0 ? '#000000' : pColor(p)} stroke={p === 0 ? C.dim : '#00000000'} lineWidth={2} scale={0}
        />
      ) as Rect,
    );
    const lab = (<Latex tex={tex} fill={C.mint} fontSize={34} x={CX - (N / 2) * S - 24} y={y} offset={[1, 0]} opacity={0} />) as Latex;
    fxLayer().add(<>{cells}{lab}</>);
    return {cells, lab, show: () => all(lab.opacity(1, 0.3), sequence(0.06, ...cells.map(c => c.scale(1, 0.3, easeOutCubic))))};
  }
  const sOldPi = strip(st.oldPi, STRIP_Y.oldPi, `P_i^{\\mathrm{old}}`, i);
  const sOldPj = strip(st.oldPj, STRIP_Y.oldPj, `P_j^{\\mathrm{old}}`, j);
  const sPi = strip(st.Pi, STRIP_Y.Pi, `P_i`, i);
  const sPj = strip(st.Pj, STRIP_Y.Pj, `P_j`, j);
  const pNote = (
    <Txt text="registers only" fontFamily={FONT} fontSize={26} fill={C.muted} x={AX - (N / 2) * S} y={STRIP_Y.oldPi} offset={[-1, 0]} opacity={0} />
  ) as Txt;
  fxLayer().add(pNote);

  yield* all(algo().opacity(1, 0.6), cap.set('One step, in the kernel\'s order'));
  yield* waitFor(0.6);

  // 1. row i ~ R
  yield* all(focus(0), cap.set('Draw row i in proportion to the row sums R'));
  yield* all(...V.rBars.map((b, r) => b.fill(r === i ? C.green : C.dim, 0.5)));
  yield* all(aRowI.lineWidth(3, 0.4), cRowI.lineWidth(3, 0.4), ...mI.map(m => m.opacity(1, 0.4)));
  yield* waitFor(0.4);

  // 2. old P_i
  yield* all(focus(1), cap.set('Compute old row P_i from rows of C, A, D and all degrees'), kBox.lineWidth(3, 0.4));
  yield* all(sOldPi.show(), pNote.opacity(1, 0.4));
  yield* kBox.lineWidth(0, 0.3);

  // 3. column j ~ old P_i
  const jCell = sOldPi.cells[j];
  yield* all(focus(2), cap.set('Draw column j in proportion to that row'));
  yield* all(jCell.stroke(C.mint, 0.3), jCell.lineWidth(4, 0.3), jCell.scale(1.15, 0.3));
  yield* all(aRowJ.lineWidth(3, 0.4), cRowJ.lineWidth(3, 0.4), ...mJ.map(m => m.opacity(1, 0.4)));
  yield* jCell.scale(1, 0.2);

  // 4. old P_j
  yield* all(focus(3), cap.set('Old row P_j too: R\'s update needs the old values'));
  yield* sOldPj.show();
  yield* waitFor(1.6);

  // ======================================================== beat D: update C
  beat('D_update');
  // 5. degrees
  yield* all(focus(4), cap.set('Add edge (i, j): both degrees go up by one'));
  // A rows i and j change in registers; the stores to A come last (step 10).
  const pendI = outline(colX(AX, j), rowY(i), S - 2, S - 2);
  const pendJ = outline(colX(AX, i), rowY(j), S - 2, S - 2);
  pendI.lineDash([6, 5]);
  pendJ.lineDash([6, 5]);
  yield* all(
    pendI.lineWidth(3, 0.4), pendJ.lineWidth(3, 0.4),
    ...[i, j].map(r => all(V.kCells[r].fill(C.deep, 0.3), V.kTxt[r].fill(C.bg, 0.3))),
  );
  V.kTxt[i].text(`${st.degNew[i]}`);
  V.kTxt[j].text(`${st.degNew[j]}`);
  yield* all(...[i, j].map(r => V.kCells[r].scale(1.2, 0.2)));
  yield* all(...[i, j].map(r => V.kCells[r].scale(1, 0.2)));
  yield* waitFor(0.4);

  // 6, 7. C rows += old A rows (+ diagonal), columns mirror.
  function* addRow(x: number, y: number, oldAy: number[], newCx: number[], oldCx: number[], n: number) {
    const ghosts: Rect[] = [];
    for (let m = 0; m < N; m++)
      if (oldAy[m]) {
        const g = (<Rect x={colX(AX, m)} y={rowY(y)} size={S - 6} radius={4} fill={C.green} opacity={0.9} />) as Rect;
        fxLayer().add(g);
        ghosts.push(g);
      }
    const targets = [...Array(N).keys()].filter(m => oldAy[m]);
    yield* all(focus(n), ...(x === i ? [aRowJ.stroke(C.mint, 0.3)] : [aRowI.stroke(C.mint, 0.3)]));
    yield* all(
      ...ghosts.map((g, q) =>
        all(g.position(new Vector2(colX(CX, targets[q]), rowY(x)), 0.8, easeInOutCubic), g.size(S - 16, 0.8)),
      ),
    );
    // + e_x on the diagonal
    const changed = [...Array(N).keys()].filter(m => newCx[m] !== oldCx[m]);
    yield* all(
      ...ghosts.map(g => g.opacity(0, 0.3)),
      ...changed.map(m => V.cCells[x][m].fill(C.green, 0.3)),
      ...changed.map(m => V.cTxt[x][m].fill(C.bg, 0.3)),
    );
    for (const m of changed) V.cTxt[x][m].text(`${newCx[m]}`);
    ghosts.forEach(g => g.remove());
    yield* waitFor(0.3);
    // columns: same values (C symmetric), strided stores
    const cb = colBox(CX, x);
    const colChanged = changed.filter(m => m !== x);
    yield* all(
      cb.lineWidth(3, 0.3),
      ...colChanged.map(m => V.cCells[m][x].fill(C.green, 0.3)),
      ...colChanged.map(m => V.cTxt[m][x].fill(C.bg, 0.3)),
    );
    for (const m of colChanged) V.cTxt[m][x].text(`${newCx[m]}`);
    yield* waitFor(0.4);
    yield* all(
      cb.lineWidth(0, 0.3),
      ...(x === i ? [aRowJ.stroke(C.green, 0.3)] : [aRowI.stroke(C.green, 0.3)]),
      ...changed.map(m => all(V.cCells[x][m].fill(C.deep, 0.3), V.cTxt[x][m].fill(C.mint, 0.3))),
      ...colChanged.map(m => all(V.cCells[m][x].fill(C.deep, 0.3), V.cTxt[m][x].fill(C.mint, 0.3))),
    );
  }
  yield* cap.set('Rows i and j of C gain the old rows of A; columns mirror');
  yield* addRow(i, j, st.oldAj, st.Ci, st.oldCi, 5);
  yield* addRow(j, i, st.oldAi, st.Cj, st.oldCj, 6);
  yield* cap.set('Only 2 rows and 2 columns of C change: O(n)');
  yield* waitFor(1.6);

  // ======================================================== beat E: new rows, R, A
  beat('E_rows');
  yield* all(focus(7), cap.set('New rows P_i and P_j, from the updated C, A and k'));
  yield* sPi.show();
  yield* sPj.show();
  yield* waitFor(0.5);

  yield* all(focus(8), cap.set('R patched by the deltas (fp64); rows i, j re-summed exactly'));
  const rChanged = [...Array(N).keys()].filter(r => Math.abs(st!.Rnew[r] - st!.Rold[r]) > 1e-12);
  yield* all(
    ...V.rBars.map((b, r) =>
      all(b.width((st!.Rnew[r] / rMax) * RLEN, 0.9, easeInOutCubic), b.fill(rChanged.includes(r) ? C.green : C.dim, 0.5)),
    ),
  );
  yield* waitFor(0.8);

  yield* all(focus(9), cap.set('Finally write A[i, j] and A[j, i]; then step t + 1'));
  yield* all(
    V.aCells[i][j].fill(C.green, 0.4), V.aCells[j][i].fill(C.green, 0.4),
    pendI.lineWidth(0, 0.4), pendJ.lineWidth(0, 0.4),
  );
  yield* waitFor(1.0);
  yield* cap.set('Each step is O(n). Next: where the memory traffic goes');
  yield* waitFor(2.0);
  beat('end');
});
