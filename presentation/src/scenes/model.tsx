import {Circle, Latex, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeInOutCubic, easeOutCubic, sequence, useRandom, Vector2, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {D, N, POS} from '../lib/chain_toy';
import {Mat, samplePair, shared, wiring, zeros} from '../lib/gnm';
import talk from '../data/talk.json';

// Talk clip M: how one network grows (toy graph, kernel conventions: empty start, one
// edge per step until T) and how one person is fitted (real 68-region target on the left;
// distribution curves and the landscape are SCHEMATIC, labelled as such).

const TT = 34; // toy target edge count
// Toy parameters (conventions as in src/kernel.py). With a small gamma the distance term
// outweighs the 1e-6 offset from the first edge, so growth is visibly local.
const ETA = -3;
const GAMMA = 0.6;
const GC = new Vector2(-470, 40); // graph centre
const pt = (i: number) => GC.add(new Vector2(POS[i][0], POS[i][1]));
const key = (i: number, j: number) => (i < j ? `${i}-${j}` : `${j}-${i}`);
const LS = talk.landscape;
const tg = talk.target68;

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '2', 'How a model grows, and how a person is fitted');
  const cap = caption(view);
  const rng = useRandom(21);

  // ---------------------------------------------------------------- toy graph
  const graph = createRef<Node>();
  const edgeLayer = createRef<Node>();
  const fx = createRef<Node>();
  const nodeLayer = createRef<Node>();
  view.add(
    <Node ref={graph}>
      <Node ref={edgeLayer} />
      <Node ref={fx} />
      <Node ref={nodeLayer} />
    </Node>,
  );
  const nodes = POS.map((_, i) => {
    const c = (<Circle position={pt(i)} size={28} fill={'#08120c'} stroke={C.deep} lineWidth={3} scale={0} />) as Circle;
    nodeLayer().add(c);
    return c;
  });
  const A: Mat = zeros(N);
  const edges = new Map<string, Line>();
  const addEdge = (i: number, j: number) => {
    A[i][j] = A[j][i] = 1;
    const l = (<Line points={[pt(i), pt(j)]} stroke={C.green} lineWidth={4} end={0} />) as Line;
    edgeLayer().add(l);
    edges.set(key(i, j), l);
    return l;
  };
  const count = createRef<Txt>();
  view.add(<Txt ref={count} fontFamily={MONO} fontSize={30} fill={C.muted} x={GC.x} y={-330} opacity={0} text="" />);
  const setCount = () => count().text(`connections ${A.flat().reduce((s, x) => s + x, 0) / 2} / T = ${TT}`);

  function* growSteps(k: number, d: number) {
    for (let s = 0; s < k; s++) {
      const [i, j] = samplePair(wiring(A, D, ETA, GAMMA), rng.nextFloat());
      const l = addEdge(i, j);
      setCount();
      yield* l.end(1, d);
      yield* l.stroke(C.deep, d * 0.6);
    }
  }

  // ======================================================== beat A: grow from empty
  beat('A_grow');
  yield* title().opacity(1, 0.6);
  setCount();
  yield* all(sequence(0.03, ...nodes.map(n => n.scale(1, 0.35, easeOutCubic))), count().opacity(1, 0.5),
    cap.set('Start from the brain’s regions, with no connections'));
  yield* waitFor(0.6);
  yield* cap.set('Add one connection at a time');
  yield* growSteps(14, 0.22);
  yield* waitFor(1.0);

  // ======================================================== beat B: the rule
  beat('B_rule');
  const P0 = wiring(A, D, ETA, GAMMA);
  const Cm = shared(A);
  let fi = -1, fj = -1, best = -1;
  for (const minShared of [2, 1])
    if (fi < 0)
      for (let i = 0; i < N; i++)
        for (let j = i + 1; j < N; j++)
          if (!A[i][j] && Cm[i][j] >= minShared && D[i][j] >= 0.8 && P0[i][j] > best) { best = P0[i][j]; fi = i; fj = j; }
  const cand = createRef<Line>();
  fx().add(<Line ref={cand} points={[pt(fi), pt(fj)]} stroke={C.mint} lineWidth={3} lineDash={[10, 10]} end={0} />);

  const boxD = createRef<Rect>(), boxM = createRef<Rect>();
  const eq = createRef<Rect>();
  const term = (ref: any, tex: string) => (
    <Rect ref={ref} padding={[10, 16]} radius={10} stroke={C.green} lineWidth={0}>
      <Latex tex={tex} fill={C.mint} fontSize={56} />
    </Rect>
  );
  view.add(
    <Rect ref={eq} layout direction="row" alignItems="center" gap={8} x={370} y={-250} opacity={0}>
      <Latex tex="P(i,j)" fill={C.mint} fontSize={56} />
      <Latex tex="\propto" fill={C.mint} fontSize={56} />
      {term(boxD, 'D(i,j)^{\\,\\eta}')}
      <Latex tex="\cdot" fill={C.mint} fontSize={56} />
      {term(boxM, 'M(i,j)^{\\,\\gamma}')}
    </Rect>,
  );
  const RX = 60;
  const lines = [
    {k: 'D', t: 'η controls the cost of distance', s: 'D = distance between the two regions'},
    {k: 'M', t: 'γ controls the preference for shared neighbours', s: 'M = matching index: how many neighbours i and j share'},
    {k: 'T', t: 'Sample one missing connection, repeat', s: 'stop at the person’s connection count T'},
  ];
  const lineNodes = lines.map((l, k) => {
    const y = -110 + k * 120;
    const g = (
      <Node opacity={0}>
        <Rect x={RX} y={y} width={12} height={12} radius={2} fill={C.green} />
        <Txt text={l.t} fontFamily={FONT} fontSize={34} fill={C.mint} x={RX + 28} y={y - 18} offset={[-1, 0]} />
        <Txt text={l.s} fontFamily={FONT} fontSize={26} fill={C.muted} x={RX + 28} y={y + 22} offset={[-1, 0]} />
      </Node>
    ) as Node;
    view.add(g);
    return g;
  });
  yield* all(cap.set('Which missing connection comes next?'), cand().end(1, 0.6),
    nodes[fi].stroke(C.green, 0.4), nodes[fj].stroke(C.green, 0.4));
  yield* all(eq().opacity(1, 0.6));
  yield* cap.set('A probability for every missing connection');
  yield* waitFor(0.8);

  // distance
  yield* all(boxD().lineWidth(3, 0.4), cand().stroke(C.green, 0.4), cand().lineDash([1000, 0], 0.4), lineNodes[0].opacity(1, 0.5));
  yield* waitFor(1.2);
  // matching
  const sharedNb = [...Array(N).keys()].filter(k => A[fi][k] && A[fj][k]);
  const nbEdges = [...edges.entries()].filter(([kk]) => {
    const [a, b] = kk.split('-').map(Number);
    return (a === fi || b === fi || a === fj || b === fj) && (sharedNb.includes(a) || sharedNb.includes(b));
  });
  yield* all(boxD().lineWidth(0, 0.3), boxM().lineWidth(3, 0.4), cand().stroke(C.mint, 0.3), lineNodes[1].opacity(1, 0.5),
    ...nbEdges.map(([, l]) => all(l.stroke(C.green, 0.4), l.lineWidth(6, 0.4))),
    ...sharedNb.map(k => all(nodes[k].fill(C.green, 0.4), nodes[k].stroke(C.green, 0.4))));
  yield* cap.set('The two parameters trade off short connections against shared neighbours');
  yield* waitFor(1.8);

  // ======================================================== beat C: grow to T
  beat('C_toT');
  yield* all(boxM().lineWidth(0, 0.3), cand().opacity(0, 0.4),
    ...nbEdges.map(([, l]) => all(l.stroke(C.deep, 0.4), l.lineWidth(4, 0.4))),
    ...sharedNb.map(k => all(nodes[k].fill('#08120c', 0.4), nodes[k].stroke(C.deep, 0.4))),
    nodes[fi].stroke(C.deep, 0.4), nodes[fj].stroke(C.deep, 0.4),
    lineNodes[2].opacity(1, 0.5), cap.set('Sample, add, update, until the network has T connections'));
  const left = TT - A.flat().reduce((s, x) => s + x, 0) / 2;
  yield* growSteps(left, 0.12);
  yield* cap.set('One run of the model: a random network, different every time');
  yield* waitFor(1.8);

  // ======================================================== beat D: fit one person
  beat('D_fit');
  yield* all(eq().opacity(0, 0.5), ...lineNodes.map(g => g.opacity(0, 0.5)), count().opacity(0, 0.5),
    graph().scale(0.42, 0.9, easeInOutCubic), graph().position(new Vector2(200 - GC.x * 0.42, -250 - GC.y * 0.42), 0.9, easeInOutCubic));

  // the person's connectome (real, 68 regions, axial view)
  const tgt = createRef<Node>();
  const S = 3.4;
  const tp = (k: number) => new Vector2(-560 + tg.xy[k][0] * S, 30 - tg.xy[k][1] * S);
  view.add(
    <Node ref={tgt} opacity={0}>
      {tg.edges.map(([a, b]) => <Line points={[tp(a), tp(b)]} stroke={C.green} lineWidth={1.2} opacity={0.35} />)}
      {tg.xy.map((_, k) => <Circle position={tp(k)} size={9} fill={C.mint} />)}
      <Txt text={`The person: ${tg.n} regions, ${tg.T} connections`} fontFamily={FONT} fontSize={28} fill={C.mint} x={-560} y={-360} />
      <Txt text="real connectome (Griffa et al.)" fontFamily={FONT} fontSize={22} fill={C.muted} x={-560} y={-325} />
    </Node>,
  );
  const step1 = createRef<Txt>();
  view.add(<Txt ref={step1} text="1  Grow a network for one (η, γ)" fontFamily={FONT} fontSize={30} fill={C.mint} x={360} y={-370} offset={[-1, 0]} opacity={0} />);
  yield* all(tgt().opacity(1, 0.6), step1().opacity(1, 0.5), cap.set('To fit a person, compare model networks with their connectome'));
  yield* waitFor(0.8);

  // 2: four distributions (schematic curves)
  const step2 = createRef<Txt>();
  view.add(<Txt ref={step2} text="2  Compare four distributions" fontFamily={FONT} fontSize={30} fill={C.mint} x={-60} y={-110} offset={[-1, 0]} opacity={0} />);
  const names = ['degree', 'clustering', 'betweenness', 'edge length'];
  const PW = 200, PH = 120;
  const panels = names.map((nm, k) => {
    const x0 = -60 + k * 235, y0 = -40;
    const sig = (t: number, c: number, w: number) => 1 / (1 + Math.exp(-(t - c) / w));
    const shift = [0.08, 0.16, 0.12, 0.2][k];
    const tgtPts: Vector2[] = [], simPts: Vector2[] = [];
    let gap = 0, gx = 0, gy1 = 0, gy2 = 0;
    for (let s = 0; s <= 40; s++) {
      const t = s / 40;
      const a = sig(t, 0.45, 0.1), b = sig(t, 0.45 + shift, 0.12);
      tgtPts.push(new Vector2(x0 + t * PW, y0 + PH - a * PH));
      simPts.push(new Vector2(x0 + t * PW, y0 + PH - b * PH));
      if (Math.abs(a - b) > gap) { gap = Math.abs(a - b); gx = x0 + t * PW; gy1 = y0 + PH - a * PH; gy2 = y0 + PH - b * PH; }
    }
    const g = (
      <Node opacity={0}>
        <Line points={[[x0, y0 - 6], [x0, y0 + PH], [x0 + PW, y0 + PH]]} stroke={C.dim} lineWidth={2} />
        <Line points={tgtPts} stroke={C.mint} lineWidth={3} />
        <Line points={simPts} stroke={C.green} lineWidth={3} lineDash={[8, 6]} />
        <Line points={[[gx, gy1], [gx, gy2]]} stroke={C.green} lineWidth={4} />
        <Txt text="KS" fontFamily={MONO} fontSize={20} fill={C.green} x={gx + 22} y={(gy1 + gy2) / 2} />
        <Txt text={nm} fontFamily={FONT} fontSize={24} fill={C.mint} x={x0 + PW / 2} y={y0 + PH + 30} />
      </Node>
    ) as Node;
    view.add(g);
    return g;
  });
  const legend = createRef<Node>();
  view.add(
    <Node ref={legend} opacity={0}>
      <Line points={[[600, -110], [640, -110]]} stroke={C.mint} lineWidth={3} />
      <Txt text="person" fontFamily={FONT} fontSize={22} fill={C.muted} x={650} y={-110} offset={[-1, 0]} />
      <Line points={[[720, -110], [760, -110]]} stroke={C.green} lineWidth={3} lineDash={[8, 6]} />
      <Txt text="model" fontFamily={FONT} fontSize={22} fill={C.muted} x={770} y={-110} offset={[-1, 0]} />
    </Node>,
  );
  const ef = createRef<Latex>();
  view.add(<Latex ref={ef} tex="E = \max(\mathrm{KS}_{\text{degree}},\ \mathrm{KS}_{\text{clustering}},\ \mathrm{KS}_{\text{betweenness}},\ \mathrm{KS}_{\text{edge length}})"
    fill={C.mint} fontSize={34} x={410} y={170} opacity={0} />);
  const sch = createRef<Txt>();
  view.add(<Txt ref={sch} text="schematic" fontFamily={FONT} fontSize={20} fill={C.muted} x={870} y={-40} offset={[1, 0]} opacity={0} />);
  yield* all(step2().opacity(1, 0.5), legend().opacity(1, 0.5), sch().opacity(1, 0.5), sequence(0.15, ...panels.map(p => p.opacity(1, 0.5))));
  yield* waitFor(0.6);
  yield* all(ef().opacity(1, 0.6), cap.set('Energy E: the worst of the four mismatches (lower is better)'));
  yield* waitFor(2.0);

  // ======================================================== beat E: landscape
  beat('E_landscape');
  yield* all(step2().opacity(0, 0.5), legend().opacity(0, 0.5), ...panels.map(p => p.opacity(0, 0.5)), ef().opacity(0, 0.5),
    sch().opacity(0, 0.5), step1().opacity(0, 0.5), graph().opacity(0, 0.5));
  const step3 = createRef<Txt>();
  view.add(<Txt ref={step3} text="3  Repeat over a grid of (η, γ), average, keep the lowest" fontFamily={FONT} fontSize={30} fill={C.mint} x={-60} y={-370} offset={[-1, 0]} opacity={0} />);
  const G = 13, CS = 34;
  const LX0 = 150, LY0 = -20;
  const E = (a: number, b: number) => {
    const u = (a - 0.6) * 1.0 + (b - 0.45) * 0.9, v = (a - 0.6) - (b - 0.45) * 1.1;
    return Math.min(1, 0.12 + 1.6 * u * u + 0.35 * v * v);
  };
  const cells: Rect[] = [];
  let bi = 0, bj = 0, bv = 9;
  for (let r = 0; r < G; r++)
    for (let c = 0; c < G; c++) {
      const v = E(c / (G - 1), 1 - r / (G - 1));
      if (v < bv) { bv = v; bi = r; bj = c; }
      const cell = (<Rect x={LX0 + (c - (G - 1) / 2) * CS} y={LY0 + (r - (G - 1) / 2) * CS} size={CS - 3} radius={3} fill={ramp(1 - v)} opacity={0} />) as Rect;
      view.add(cell);
      cells.push(cell);
    }
  const lab = createRef<Node>();
  const HW = (G * CS) / 2;
  view.add(
    <Node ref={lab} opacity={0}>
      <Txt text="η" fontFamily={FONT} fontSize={30} fill={C.mint} x={LX0} y={LY0 + HW + 34} />
      <Txt text="γ" fontFamily={FONT} fontSize={30} fill={C.mint} x={LX0 - HW - 34} y={LY0} />
      <Txt text="schematic" fontFamily={FONT} fontSize={20} fill={C.muted} x={LX0 + HW} y={LY0 + HW + 34} offset={[1, 0]} />
      <Txt text="each square:" fontFamily={FONT} fontSize={26} fill={C.muted} x={LX0 + HW + 50} y={LY0 - 120} offset={[-1, 0]} />
      <Txt text={`${LS.repeats} simulated networks,`} fontFamily={FONT} fontSize={26} fill={C.mint} x={LX0 + HW + 50} y={LY0 - 84} offset={[-1, 0]} />
      <Txt text="mean energy (brighter = better)" fontFamily={FONT} fontSize={26} fill={C.mint} x={LX0 + HW + 50} y={LY0 - 48} offset={[-1, 0]} />
    </Node>,
  );
  const ring = createRef<Rect>();
  const bestLab = createRef<Txt>();
  const bp = new Vector2(LX0 + (bj - (G - 1) / 2) * CS, LY0 + (bi - (G - 1) / 2) * CS);
  view.add(
    <>
      <Rect ref={ring} position={bp} size={CS + 10} radius={6} stroke={C.mint} lineWidth={4} opacity={0} />
      <Txt ref={bestLab} text="best fit: lowest mean energy" fontFamily={FONT} fontSize={28} fill={C.green} x={LX0 + HW + 50} y={bp.y + 40} offset={[-1, 0]} opacity={0} />
    </>,
  );
  yield* all(step3().opacity(1, 0.5), lab().opacity(1, 0.5), cap.set('One network is random, so each grid point is simulated many times'));
  // fill in reading order of a diagonal sweep
  const order = cells.map((c, k) => ({c, k})).sort((a, b) => ((a.k % G) + Math.floor(a.k / G)) - ((b.k % G) + Math.floor(b.k / G)));
  yield* sequence(1.8 / cells.length, ...order.map(o => o.c.opacity(1, 0.25)));
  yield* all(ring().opacity(1, 0.4), bestLab().opacity(1, 0.4));
  yield* cap.set(`The real grid: ${LS.grid[0]} × ${LS.grid[1]} points × ${LS.repeats} repeats = ${LS.sims_per_subject.toLocaleString('en-US')} networks per person`);
  yield* waitFor(2.5);
  beat('end');
});
