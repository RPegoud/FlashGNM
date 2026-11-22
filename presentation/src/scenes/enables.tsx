import {Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeOutCubic, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import talk from '../data/talk.json';
import dense from '../data/science_dense.json';

// Talk clip E: conclusion, what the speedup enables. Numbers: src/data/talk.json,
// src/data/science_dense.json.

const LS = talk.landscape;
const comma = (x: number) => Math.round(x).toLocaleString('en-US');
const D = dense.points.reduce((a, b) => (b.T > a.T ? b : a));

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '5', 'What this enables');
  const cap = caption(view);

  const items = [
    {big: `${(LS.ours_s_cohort_total / 3600).toFixed(1)} h`,
     head: 'Whole studies, every fit replicated',
     sub: `${LS.subjects} people at ${comma(LS.n)} regions, every grid point simulated ${LS.repeats}×, on one GPU`},
    {big: `${Math.round(D.setup_s + D.run_s)} s`,
     head: 'Dense connectomes',
     sub: `${comma(D.B)} simulated networks with ${comma(D.T)} connections each`},
    {big: 'per person',
     head: 'Honest uncertainty',
     sub: 'an error bar on every individual fit, not one lucky run'},
    {big: 'every rule',
     head: 'Fair model comparison',
     sub: 'rival wiring rules tested side by side, at every resolution'},
  ];

  // ======================================================== beat A: what it enables
  beat('A_enables');
  const rows: Node[] = items.map((it, k) => {
    const y = -270 + k * 160;
    const g = (
      <Node opacity={0} x={-40}>
        <Rect x={-820} y={y} width={6} height={110} radius={3} fill={C.green} />
        <Txt text={it.big} fontFamily={MONO} fontSize={60} fontWeight={700} fill={C.green} x={-780} y={y} offset={[-1, 0]} />
        <Txt text={it.head} fontFamily={FONT} fontSize={38} fontWeight={600} fill={C.mint} x={-260} y={y - 24} offset={[-1, 0]} />
        <Txt text={it.sub} fontFamily={FONT} fontSize={28} fill={C.muted} x={-260} y={y + 26} offset={[-1, 0]} />
      </Node>
    ) as Node;
    view.add(g);
    return g;
  });
  yield* all(title().opacity(1, 0.6), cap.set('Questions that used to be too expensive to ask'));
  for (const r of rows) {
    yield* all(r.opacity(1, 0.5), r.x(0, 0.5, easeOutCubic));
    yield* waitFor(1.6);
  }
  yield* waitFor(1.0);

  // ======================================================== beat B: close
  beat('B_close');
  const close = createRef<Node>();
  view.add(
    <Node ref={close} opacity={0}>
      <Txt text="Every brain. The whole landscape. Replicated." fontFamily={FONT} fontSize={64} fontWeight={600} fill={C.green} y={-40} />
      <Txt text={`${comma(LS.sims_per_subject)} simulations per person, in about a minute`} fontFamily={FONT} fontSize={34} fill={C.mint} y={50} />
      <Txt text="Thank you" fontFamily={FONT} fontSize={30} fill={C.muted} y={170} />
    </Node>,
  );
  yield* all(...rows.map(r => r.opacity(0, 0.5)), title().opacity(0, 0.5), cap.node().opacity(0, 0.5));
  yield* close().opacity(1, 0.8);
  yield* waitFor(3.0);
  beat('end');
});
