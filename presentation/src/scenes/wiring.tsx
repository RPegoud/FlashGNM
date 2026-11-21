import {Circle, Latex, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  easeInOutCubic,
  easeOutCubic,
  sequence,
  useLogger,
  useRandom,
  Vector2,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {Mat, degrees, distances, samplePair, shared, wiring, zeros} from '../lib/gnm';

// ---------------------------------------------------------------- model setup
const ETA = -3;
const GAMMA = 1.5;
const CENTER: [number, number] = [-450, 40];
const RAW: [number, number][] = [
  [-300, -60], [-225, -175], [-95, -235], [60, -245], [205, -190], [315, -85],
  [345, 45], [285, 165], [150, 235], [0, 255], [-150, 225], [-285, 130],
  [-165, -70], [-45, -125], [110, -85], [205, 45], [60, 100], [-100, 85],
];
const POS: [number, number][] = RAW.map(([x, y]) => [CENTER[0] + x * 1.12, CENTER[1] + y * 1.12]);
const N = POS.length;
const D = distances(POS, 150);

// Seed network: each node wired to its two nearest neighbours. (A sparser seed closes
// every open triad within a few steps; K is then 0 everywhere and P goes uniform.)
const A0: Mat = zeros(N);
for (let i = 0; i < N; i++) {
  const order = [...Array(N).keys()].filter(j => j !== i).sort((a, b) => D[i][a] - D[i][b]);
  for (const b of order.slice(0, 2)) A0[i][b] = A0[b][i] = 1;
}
const SEED: [number, number][] = [];
for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (A0[i][j]) SEED.push([i, j]);

const key = (i: number, j: number) => (i < j ? `${i}-${j}` : `${j}-${i}`);
const pt = (i: number) => new Vector2(POS[i][0], POS[i][1]);

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const rng = useRandom(7);
  const A = A0.map(r => r.slice());

  const title = header(view, '01', 'The wiring rule');
  const cap = caption(view);

  // ---------------------------------------------------------------- layers
  const probLayer = createRef<Node>();
  const edgeLayer = createRef<Node>();
  const fxLayer = createRef<Node>();
  const nodeLayer = createRef<Node>();
  view.add(
    <>
      <Node ref={probLayer} />
      <Node ref={edgeLayer} />
      <Node ref={fxLayer} />
      <Node ref={nodeLayer} />
    </>,
  );

  const nodes: Circle[] = [];
  for (let i = 0; i < N; i++) {
    const c = (
      <Circle
        position={pt(i)}
        size={30}
        fill={'#08120c'}
        stroke={C.deep}
        lineWidth={3}
        scale={0}
      />
    ) as Circle;
    nodes.push(c);
    nodeLayer().add(c);
  }

  const edges = new Map<string, Line>();
  function makeEdge(i: number, j: number) {
    const l = (
      <Line points={[pt(i), pt(j)]} stroke={C.deep} lineWidth={4} end={0} opacity={0.9} />
    ) as Line;
    edgeLayer().add(l);
    edges.set(key(i, j), l);
    return l;
  }

  // Probability lines for every pair (hidden until beat E).
  const probs = new Map<string, Line>();
  for (let i = 0; i < N; i++)
    for (let j = i + 1; j < N; j++) {
      const l = (<Line points={[pt(i), pt(j)]} stroke={C.green} lineWidth={2} opacity={0} />) as Line;
      probLayer().add(l);
      probs.set(key(i, j), l);
    }

  // ======================================================== beat A: growth
  beat('A_growth');
  yield* title().opacity(1, 0.6);
  yield* all(
    cap.set('A connectome grows one edge at a time'),
    sequence(0.03, ...nodes.map(n => n.scale(1, 0.4, easeOutCubic))),
  );
  yield* sequence(0.06, ...SEED.map(([i, j]) => makeEdge(i, j).end(1, 0.4)));
  yield* waitFor(0.3);

  // A few model steps, quickly.
  for (let s = 0; s < 6; s++) {
    const P = wiring(A, D, ETA, GAMMA);
    const [i, j] = samplePair(P, rng.nextFloat());
    A[i][j] = A[j][i] = 1;
    const l = makeEdge(i, j);
    l.stroke(C.green);
    yield* l.end(1, 0.35);
    yield* l.stroke(C.deep, 0.3);
  }
  yield* waitFor(0.6);

  // Focus pair: the most likely missing edge with >= 2 shared neighbours.
  const Cm = shared(A);
  const deg = degrees(A);
  let P = wiring(A, D, ETA, GAMMA);
  // Focus pair: most likely missing edge, preferring >= 2 shared neighbours and enough
  // distance for the labels to breathe; relax the criteria if nothing qualifies.
  let fi = -1, fj = -1;
  for (const [minShared, minD] of [[2, 1.2], [2, 0.8], [1, 0.8], [1, 0]]) {
    let best = -1;
    for (let i = 0; i < N; i++)
      for (let j = i + 1; j < N; j++)
        if (!A[i][j] && Cm[i][j] >= minShared && D[i][j] >= minD && P[i][j] > best) {
          best = P[i][j];
          fi = i;
          fj = j;
        }
    if (fi >= 0) break;
  }
  useLogger().info(`FOCUS ${fi} ${fj} shared=${Cm[fi][fj]} deg=${deg[fi]},${deg[fj]} D=${D[fi][fj].toFixed(2)}`);
  const sharedNb = [...Array(N).keys()].filter(k => A[fi][k] && A[fj][k]);

  // ======================================================== beat B: which edge?
  beat('B_question');
  const cand = createRef<Line>();
  const li = createRef<Latex>();
  const lj = createRef<Latex>();
  const dir = pt(fj).sub(pt(fi)).normalized;
  const nrm = new Vector2(-dir.y, dir.x);
  const labelOff = (k: number, sign: number) => pt(k).add(dir.scale(sign * 34)).add(nrm.scale(-10));
  fxLayer().add(
    <>
      <Line
        ref={cand}
        points={[pt(fi), pt(fj)]}
        stroke={C.mint}
        lineWidth={3}
        lineDash={[10, 10]}
        end={0}
      />
      <Latex ref={li} tex="i" fill={C.mint} fontSize={40} position={labelOff(fi, -1)} opacity={0} />
      <Latex ref={lj} tex="j" fill={C.mint} fontSize={40} position={labelOff(fj, 1)} opacity={0} />
    </>,
  );

  // Equation (right half), one box per term so terms can be highlighted.
  const eq = createRef<Rect>();
  const boxP = createRef<Rect>();
  const boxD = createRef<Rect>();
  const boxK = createRef<Rect>();
  const term = (ref: any, tex: string) => (
    <Rect ref={ref} padding={[10, 16]} radius={10} stroke={C.green} lineWidth={0} fill={'#00000000'}>
      <Latex tex={tex} fill={C.mint} fontSize={72} />
    </Rect>
  );
  view.add(
    <Rect ref={eq} layout direction="row" alignItems="center" gap={10} x={470} y={-250} opacity={0}>
      {term(boxP, 'P_{ij}')}
      <Latex tex="\propto" fill={C.mint} fontSize={72} />
      {term(boxD, 'D_{ij}^{\\,\\eta}')}
      <Latex tex="\cdot" fill={C.mint} fontSize={72} />
      {term(boxK, 'K_{ij}^{\\,\\gamma}')}
    </Rect>,
  );

  yield* all(
    cap.set('Which missing edge comes next?'),
    nodes[fi].stroke(C.green, 0.4),
    nodes[fj].stroke(C.green, 0.4),
    nodes[fi].scale(1.25, 0.4),
    nodes[fj].scale(1.25, 0.4),
    cand().end(1, 0.6),
    li().opacity(1, 0.4),
    lj().opacity(1, 0.4),
  );
  yield* waitFor(0.4);
  yield* all(eq().opacity(1, 0.6), eq().y(-260, 0.6, easeOutCubic));
  yield* boxP().lineWidth(3, 0.4);
  yield* waitFor(1.0);
  yield* boxP().lineWidth(0, 0.3);

  // ======================================================== beat C: distance
  beat('C_distance');
  const dLabel = createRef<Latex>();
  const mid = pt(fi).add(pt(fj)).scale(0.5).add(nrm.scale(34));
  fxLayer().add(<Latex ref={dLabel} tex="D_{ij}" fill={C.green} fontSize={40} position={mid} opacity={0} />);

  // Mini plot of D^eta.
  const plot = createRef<Node>();
  const curve = createRef<Line>();
  const PW = 520, PH = 250, PX = 470, PY = 40;
  const curvePts: Vector2[] = [];
  for (let s = 0; s <= 60; s++) {
    const d = 0.6 + (s / 60) * 3.4;
    const y = Math.pow(d / 0.6, ETA);
    curvePts.push(new Vector2(PX - PW / 2 + (s / 60) * PW, PY + PH / 2 - y * PH));
  }
  view.add(
    <Node ref={plot} opacity={0}>
      <Line points={[[PX - PW / 2, PY - PH / 2 - 20], [PX - PW / 2, PY + PH / 2], [PX + PW / 2 + 20, PY + PH / 2]]} stroke={C.muted} lineWidth={2} />
      <Line ref={curve} points={curvePts} stroke={C.green} lineWidth={5} end={0} />
      <Txt text="distance" fontFamily={FONT} fontSize={26} fill={C.muted} x={PX + PW / 2} y={PY + PH / 2 + 34} offset={[1, 0]} />
      <Latex tex="D^{\eta},\ \eta < 0" fill={C.mint} fontSize={36} x={PX - PW / 2 + 30} y={PY - PH / 2 + 10} offset={[-1, 0]} />
    </Node>,
  );

  yield* all(
    boxD().lineWidth(3, 0.4),
    cap.set('Distance: long connections are costly'),
    cand().stroke(C.green, 0.4),
    cand().lineDash([1000, 0], 0.4),
    dLabel().opacity(1, 0.4),
  );
  yield* all(plot().opacity(1, 0.4), curve().end(1, 1.2, easeInOutCubic));
  yield* waitFor(1.5);

  // ======================================================== beat D: matching index
  beat('D_matching');
  const kDef = createRef<Latex>();
  const kNum = createRef<Latex>();
  view.add(
    <>
      <Latex
        ref={kDef}
        tex="K_{ij} = \frac{(A^{\top}A)_{ij}}{(k_i + k_j - 2A_{ij})\,/\,2}"
        fill={C.mint}
        fontSize={56}
        x={470}
        y={-20}
        opacity={0}
      />
      <Latex
        ref={kNum}
        tex={`= \\frac{${Cm[fi][fj]}}{(${deg[fi]} + ${deg[fj]})\\,/\\,2} \\approx ${(Cm[fi][fj] / ((deg[fi] + deg[fj]) / 2)).toFixed(2)}`}
        fill={C.green}
        fontSize={52}
        x={470}
        y={150}
        opacity={0}
      />
    </>,
  );
  const sharedTxt = createRef<Txt>();
  view.add(
    <Txt ref={sharedTxt} text="shared neighbours / mean degree" fontFamily={FONT} fontSize={28} fill={C.muted} x={470} y={270} opacity={0} />,
  );

  const iEdges = [...edges.entries()].filter(([k]) => k.split('-').map(Number).includes(fi) || k.split('-').map(Number).includes(fj));
  const sharedEdges = iEdges.filter(([k]) => {
    const [a, b] = k.split('-').map(Number);
    return sharedNb.includes(a) || sharedNb.includes(b);
  });
  yield* all(
    boxD().lineWidth(0, 0.3),
    plot().opacity(0, 0.3),
    dLabel().opacity(0, 0.3),
    cand().stroke(C.mint, 0.3),
    boxK().lineWidth(3, 0.4),
    cap.set('Matching index: how many neighbours i and j share'),
  );
  yield* all(...iEdges.map(([, l]) => l.stroke(C.mint, 0.4)), ...iEdges.map(([, l]) => l.opacity(0.6, 0.4)));
  yield* all(
    ...sharedEdges.map(([, l]) => all(l.stroke(C.green, 0.4), l.opacity(1, 0.4), l.lineWidth(6, 0.4))),
    ...sharedNb.map(k => all(nodes[k].fill(C.green, 0.4), nodes[k].stroke(C.green, 0.4), nodes[k].scale(1.2, 0.4))),
  );
  yield* all(kDef().opacity(1, 0.5), kDef().y(-40, 0.5, easeOutCubic));
  yield* waitFor(0.6);
  yield* all(kNum().opacity(1, 0.5), kNum().y(130, 0.5, easeOutCubic), sharedTxt().opacity(1, 0.5));
  yield* waitFor(2.0);

  // ======================================================== beat E: sampling
  beat('E_sample');
  yield* all(
    boxK().lineWidth(0, 0.3),
    kDef().opacity(0, 0.4),
    kNum().opacity(0, 0.4),
    sharedTxt().opacity(0, 0.4),
    cand().opacity(0, 0.4),
    li().opacity(0, 0.4),
    lj().opacity(0, 0.4),
    ...[...edges.values()].map(l => all(l.stroke(C.deep, 0.4), l.opacity(0.9, 0.4), l.lineWidth(4, 0.4))),
    ...nodes.map(n => all(n.fill('#08120c', 0.4), n.stroke(C.deep, 0.4), n.scale(1, 0.4))),
  );

  const counter = createRef<Txt>();
  view.add(
    <Txt ref={counter} fontFamily={MONO} fontSize={34} fill={C.muted} x={470} y={60} opacity={0}
      text={`edges  ${[...edges.keys()].length} / T`} />,
  );
  const note = createRef<Txt>();
  view.add(
    <Txt ref={note} fontFamily={FONT} fontSize={28} fill={C.muted} x={470} y={-140} opacity={0}
      text="line brightness ∝ P" />,
  );

  function* showProbs(P: Mat, d: number) {
    let mx = 0;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) mx = Math.max(mx, P[i][j]);
    const anims = [];
    for (let i = 0; i < N; i++)
      for (let j = i + 1; j < N; j++) {
        const p = A[i][j] ? 0 : Math.sqrt(P[i][j] / mx);
        const l = probs.get(key(i, j))!;
        anims.push(all(l.opacity(0.08 + 0.85 * p * (A[i][j] ? 0 : 1) * (p > 0.02 ? 1 : 0.3), d), l.lineWidth(1.5 + 5 * p, d)));
        if (A[i][j]) anims.push(l.opacity(0, d));
      }
    yield* all(...anims);
  }

  P = wiring(A, D, ETA, GAMMA);
  yield* all(
    cap.set('Every missing edge competes, weighted by P'),
    counter().opacity(1, 0.5),
    note().opacity(1, 0.5),
    showProbs(P, 0.8),
  );
  yield* waitFor(1.2);

  const steps = [1.0, 0.7, 0.5, 0.4];
  for (let s = 0; s < steps.length; s++) {
    const d = steps[s];
    if (s === 1) yield* cap.set('Sample one, add it, update P, repeat until T edges', 0.4);
    const [i, j] = samplePair(P, rng.nextFloat());
    const pl = probs.get(key(i, j))!;
    yield* all(pl.opacity(1, d * 0.4), pl.lineWidth(8, d * 0.4), nodes[i].stroke(C.green, d * 0.4), nodes[j].stroke(C.green, d * 0.4));
    A[i][j] = A[j][i] = 1;
    const l = makeEdge(i, j);
    l.stroke(C.green);
    l.lineWidth(6);
    pl.opacity(0);
    counter().text(`edges  ${edges.size} / T`);
    yield* l.end(1, d * 0.3);
    P = wiring(A, D, ETA, GAMMA);
    yield* all(
      showProbs(P, d * 0.6),
      l.stroke(C.deep, d * 0.6),
      l.lineWidth(4, d * 0.6),
      nodes[i].stroke(C.deep, d * 0.6),
      nodes[j].stroke(C.deep, d * 0.6),
    );
  }
  yield* waitFor(2.5);
  beat('end');
});
