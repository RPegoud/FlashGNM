import {Circle, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  createSignal,
  easeInOutCubic,
  easeOutCubic,
  sequence,
  SimpleSignal,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import data from '../data/science_s3.json';

// S3 "Recall = chance": per distance bin, how many of the subject's real edges the fitted
// model places, vs chance (the model's own edge density in that bin).
// Numbers: src/data/science_s3.json (scripts/science_export.py).

type Panel = typeof data.classic[0];
const NB = data.bins_mm.length - 1;
const binLabel = (k: number) => {
  const lo = data.bins_mm[k], hi = data.bins_mm[k + 1];
  return hi === null ? `> ${lo}` : `${lo}–${hi}`;
};

const X0 = -720, X1 = 380, BASE = 290, H = 500;
const BW = (X1 - X0) / NB;
const bx = (k: number) => X0 + BW * (k + 0.5);

// axis maxima whose quarters are round percentages
const NICE = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0];

// log axis for the "x chance" view
const L0 = Math.log10(0.5), L1 = Math.log10(150);
const ly = (r: number) => BASE - ((Math.log10(Math.max(r, 0.5)) - L0) / (L1 - L0)) * H;

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '09', 'Recall = chance');
  const cap = caption(view);

  // ---------------------------------------------------------------- linear view
  const ymax = createSignal(1);
  const chanceV: SimpleSignal<number>[] = Array.from({length: NB}, () => createSignal(0));
  const recallV: SimpleSignal<number>[] = Array.from({length: NB}, () => createSignal(0));
  const lin = createRef<Node>();
  const fmt = (v: number) => (v >= 0.1 ? `${(v * 100).toFixed(0)}%` : `${(v * 100).toFixed(1)}%`);
  view.add(
    <Node ref={lin} opacity={0}>
      {[0, 0.25, 0.5, 0.75, 1].map(f => (
        <Node>
          <Line points={[[X0, BASE - f * H], [X1, BASE - f * H]]} stroke={f === 0 ? C.muted : C.faint} lineWidth={f === 0 ? 2 : 1} />
          <Txt text={() => fmt(f * ymax())} fontFamily={MONO} fontSize={22} fill={C.muted} x={X0 - 16} y={BASE - f * H} offset={[1, 0]} />
        </Node>
      ))}
      {Array.from({length: NB}, (_, k) => (
        <Node>
          <Rect x={bx(k)} y={BASE} offset={[0, 1]} width={BW * 0.62} height={() => (chanceV[k]() / ymax()) * H} fill={C.dim} radius={[4, 4, 0, 0]} />
          <Rect x={bx(k)} y={BASE} offset={[0, 1]} width={BW * 0.3} height={() => (recallV[k]() / ymax()) * H} fill={C.green} radius={[3, 3, 0, 0]} />
        </Node>
      ))}
      <Txt text="recall of the subject’s real edges" fontFamily={FONT} fontSize={26} fill={C.mint} x={X0 - 110} y={BASE - H / 2} rotation={-90} />
    </Node>,
  );
  // shared x axis labels
  const xlab = createRef<Node>();
  view.add(
    <Node ref={xlab} opacity={0}>
      {Array.from({length: NB}, (_, k) => (
        <Txt text={binLabel(k)} fontFamily={MONO} fontSize={24} fill={C.muted} x={bx(k)} y={BASE + 30} />
      ))}
      <Txt text="edge length (mm, Euclidean)" fontFamily={FONT} fontSize={26} fill={C.mint} x={(X0 + X1) / 2} y={BASE + 72} />
    </Node>,
  );

  // legend + panel label (right column)
  const RX = 450;
  const legend = createRef<Node>();
  const pLabel = createRef<Txt>();
  const pSub = createRef<Txt>();
  view.add(
    <>
      <Txt ref={pLabel} fontFamily={FONT} fontSize={30} fontWeight={500} fill={C.mint} x={RX} y={-200} offset={[-1, 0]} opacity={0} text="" />
      <Txt ref={pSub} fontFamily={MONO} fontSize={28} fill={C.muted} x={RX} y={-152} offset={[-1, 0]} opacity={0} text="" />
      <Node ref={legend} opacity={0}>
        <Rect x={RX + 14} y={-60} width={28} height={28} fill={C.green} radius={3} />
        <Txt text="fitted model" fontFamily={FONT} fontSize={26} fill={C.mint} x={RX + 44} y={-60} offset={[-1, 0]} />
        <Rect x={RX + 14} y={-10} width={28} height={28} fill={C.dim} radius={3} />
        <Txt text="chance" fontFamily={FONT} fontSize={26} fill={C.mint} x={RX + 44} y={-10} offset={[-1, 0]} />
        <Txt text="(model’s own density in the bin)" fontFamily={FONT} fontSize={22} fill={C.muted} x={RX + 44} y={22} offset={[-1, 0]} />
        <Txt text={`mean over ${data.classic[0].subjects} subjects`} fontFamily={FONT} fontSize={22} fill={C.muted} x={RX} y={80} offset={[-1, 0]} />
      </Node>
    </>,
  );

  function* showPanel(p: Panel, d: number, first = false) {
    const raw = Math.max(...p.recall, ...p.chance) * 1.04;
    const mx = NICE.find(v => v >= raw) ?? 1;
    const label = p.label.charAt(0).toUpperCase() + p.label.slice(1);
    if (!first) yield* all(pLabel().opacity(0, 0.2), pSub().opacity(0, 0.2));
    pLabel().text(label);
    pSub().text(`n = ${p.n} regions`);
    yield* all(
      pLabel().opacity(1, 0.3),
      pSub().opacity(1, 0.3),
      ymax(mx, d, easeInOutCubic),
      ...chanceV.map((s, k) => s(p.chance[k], d, easeInOutCubic)),
      ...recallV.map((s, k) => s(first ? 0 : p.recall[k], d, easeInOutCubic)),
    );
  }

  // ======================================================== beat A: the question
  beat('A_question');
  yield* title().opacity(1, 0.6);
  const c68 = data.classic[0];
  yield* all(lin().opacity(1, 0.5), xlab().opacity(1, 0.5), legend().opacity(1, 0.5),
    cap.set('Of a subject’s real edges at each length, how many does the best fit place?'));
  yield* showPanel(c68, 0.9, true);
  yield* waitFor(0.6);
  yield* sequence(0.08, ...recallV.map((s, k) => s(c68.recall[k], 0.8, easeOutCubic)));
  yield* cap.set('Exactly as many as chance');
  yield* waitFor(1.5);

  // ======================================================== beat B: every model
  beat('B_every_model');
  yield* cap.set('Every rule, every resolution');
  for (const p of [data.classic[1], data.classic[2], ...data.rules]) {
    yield* showPanel(p, 0.8);
    yield* waitFor(0.9);
  }
  yield* waitFor(0.6);

  // ======================================================== beat C: other brains
  beat('C_other_brains');
  const logv = createRef<Node>();
  const one = createRef<Line>();
  view.add(
    <Node ref={logv} opacity={0}>
      {[1, 10, 100].map(t => (
        <Node>
          <Line points={[[X0, ly(t)], [X1, ly(t)]]} stroke={t === 1 ? C.mint : C.faint} lineWidth={t === 1 ? 2 : 1} lineDash={t === 1 ? [10, 8] : []} />
          <Txt text={`${t}×`} fontFamily={MONO} fontSize={24} fill={t === 1 ? C.mint : C.muted} x={X0 - 16} y={ly(t)} offset={[1, 0]} />
        </Node>
      ))}
      <Txt text="recall ÷ chance  (log scale)" fontFamily={FONT} fontSize={26} fill={C.mint} x={X0 - 110} y={BASE - H / 2} rotation={-90} />
      <Txt text="chance" fontFamily={FONT} fontSize={24} fill={C.mint} x={X1 - 4} y={ly(1) + 26} offset={[1, 0]} />
    </Node>,
  );
  const m = data.classic[2]; // classic rule, n = 1000
  const g = data.consensus[2]; // group consensus, n = 1000
  const modelDots: Circle[] = [];
  const gBars: Rect[] = [];
  const gLabs: Txt[] = [];
  for (let k = 0; k < NB; k++) {
    const r = m.recall[k] / m.chance[k];
    const rg = g.recall[k] / g.chance[k];
    const bar = (<Rect x={bx(k) + BW * 0.12} y={ly(1)} offset={[0, 1]} width={BW * 0.36} height={0} fill={C.mint} radius={[4, 4, 0, 0]} />) as Rect;
    gBars.push(bar);
    view.add(bar);
    const dot = (<Circle x={bx(k) - BW * 0.24} y={ly(r)} size={20} fill={C.green} stroke={C.bg} lineWidth={3} opacity={0} />) as Circle;
    modelDots.push(dot);
    view.add(dot);
    const lab = (<Txt fontFamily={MONO} fontSize={24} fill={C.mint} x={bx(k) + BW * 0.12} y={ly(rg) - 24} opacity={0}
      text={`${rg >= 10 ? rg.toFixed(0) : rg.toFixed(1)}×`} />) as Txt;
    gLabs.push(lab);
    view.add(lab);
  }
  const legend2 = createRef<Node>();
  view.add(
    <Node ref={legend2} opacity={0}>
      <Circle x={RX + 14} y={-60} size={22} fill={C.green} />
      <Txt text="fitted model (classic rule)" fontFamily={FONT} fontSize={26} fill={C.mint} x={RX + 44} y={-60} offset={[-1, 0]} />
      <Rect x={RX + 14} y={-10} width={24} height={28} fill={C.mint} radius={3} />
      <Txt text="other brains" fontFamily={FONT} fontSize={26} fill={C.mint} x={RX + 44} y={-10} offset={[-1, 0]} />
      <Txt text={`(edge frequency in the other ${g.subjects - 1})`} fontFamily={FONT} fontSize={22} fill={C.muted} x={RX + 44} y={22} offset={[-1, 0]} />
    </Node>,
  );
  yield* all(
    lin().opacity(0, 0.5), legend().opacity(0, 0.5),
    cap.set('Measured against chance, the models sit on the line'),
  );
  pLabel().text('n = 1000 regions');
  pSub().text('');
  yield* all(logv().opacity(1, 0.5), legend2().opacity(1, 0.5), pLabel().opacity(1, 0.3),
    sequence(0.05, ...modelDots.map(d => d.opacity(1, 0.4))));
  yield* waitFor(1.0);
  yield* cap.set('Other people’s brains predict the long-range edges far better');
  yield* sequence(0.1, ...gBars.map((b, k) => all(
    b.height(ly(1) - ly(g.recall[k] / g.chance[k]), 0.8, easeOutCubic),
    gLabs[k].opacity(1, 0.6),
  )));
  yield* waitFor(0.8);

  // long-range callout
  const call = createRef<Node>();
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  view.add(
    <Node ref={call} opacity={0}>
      <Txt text="pairs > 60 mm, top 25% predicted:" fontFamily={FONT} fontSize={24} fill={C.muted} x={RX} y={120} offset={[-1, 0]} />
      <Txt text={`model ${pct(m.long_top25_hit)}`} fontFamily={MONO} fontSize={28} fill={C.green} x={RX} y={160} offset={[-1, 0]} />
      <Txt text={`base rate ${pct(m.long_base_rate)}`} fontFamily={MONO} fontSize={28} fill={C.muted} x={RX} y={198} offset={[-1, 0]} />
      <Txt text={`other brains ${pct(g.long_top25_hit)}`} fontFamily={MONO} fontSize={28} fill={C.mint} x={RX} y={236} offset={[-1, 0]} />
      <Txt text="are real edges" fontFamily={FONT} fontSize={24} fill={C.muted} x={RX} y={274} offset={[-1, 0]} />
    </Node>,
  );
  yield* all(call().opacity(1, 0.6), cap.set('The models match the statistics, not the wiring'));
  yield* waitFor(3.0);
  beat('end');
});
