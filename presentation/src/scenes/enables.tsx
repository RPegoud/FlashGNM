import {Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeOutCubic, sequence, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import talk from '../data/talk.json';
import bench from '../data/bench.json';

// Talk clip E: one final TL;DR / next-steps frame (draft_talk_feedback_v2 §6).
// Numbers: src/data/bench.json (throughput), src/data/talk.json (cohort cost).

const LS = talk.landscape;
const H = bench.headline;
const comma = (x: number) => Math.round(x).toLocaleString('en-US');
const sig3 = (x: number) => Number(x.toPrecision(3)).toLocaleString('en-US');

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '6', 'TL;DR');
  const cap = caption(view);

  const head = createRef<Node>();
  view.add(
    <Node ref={head} opacity={0}>
      <Txt text={`~${sig3(H.vs_toolbox)}× higher simulation throughput`} fontFamily={FONT} fontSize={64} fontWeight={700} fill={C.green} x={-870} y={-320} offset={[-1, 0]} />
      <Txt text={`n = ${comma(H.n)}, B = ${comma(H.B)}, same H100`} fontFamily={MONO} fontSize={24} fill={C.muted} x={-870} y={-262} offset={[-1, 0]} />
    </Node>,
  );
  const sub = createRef<Txt>();
  view.add(<Txt ref={sub} fontFamily={FONT} fontSize={36} fill={C.mint} x={-870} y={-195} offset={[-1, 0]} opacity={0}
    text={`${LS.subjects} replicated fits at ${comma(LS.n)} regions: ${(LS.toolbox_s_cohort_total / 86400 / 365).toFixed(1)} years → ${(LS.ours_s_cohort_total / 3600).toFixed(1)} hours`} />);

  const results = [
    'Parameter correlations can arise from fitting uncertainty',
    'Single-run searches select lucky minima',
    `Long-range failure persists at ${comma(LS.n)} regions`,
  ];
  const next = [
    'PR to the GenerativeNetworkModels toolbox (in preparation)',
    'Dense probabilistic connectomes',
    'Per-subject uncertainty',
    'Better similarity measures, including DeltaCon',
  ];
  const col = (x: number, label: string, items: string[], numbered: boolean) => {
    const g = createRef<Node>();
    const rows: Node[] = [];
    view.add(<Node ref={g}><Txt text={label} fontFamily={FONT} fontSize={30} fontWeight={600} fill={C.green} x={x} y={-90} offset={[-1, 0]} opacity={0} /></Node>);
    items.forEach((t, k) => {
      const y = -20 + k * 82;
      const r = (
        <Node opacity={0}>
          {numbered
            ? <Txt text={`${k + 1}`} fontFamily={MONO} fontSize={30} fontWeight={700} fill={C.green} x={x} y={y} offset={[-1, 0]} />
            : <Rect x={x + 8} y={y} width={12} height={12} radius={2} fill={C.green} />}
          <Txt text={t} fontFamily={FONT} fontSize={30} fill={C.mint} x={x + 40} y={y} offset={[-1, 0]} width={760} textWrap />
        </Node>
      ) as Node;
      g().add(r);
      rows.push(r);
    });
    return {label: g().children()[0], rows};
  };
  const R = col(-870, 'Results', results, true);
  const N = col(60, 'Next', next, false);

  // ======================================================== single beat
  beat('A_tldr');
  yield* title().opacity(1, 0.5);
  yield* all(head().opacity(1, 0.6));
  yield* sub().opacity(1, 0.5);
  yield* waitFor(0.4);
  yield* R.label.opacity(1, 0.4);
  yield* sequence(0.35, ...R.rows.map(r => r.opacity(1, 0.4)));
  yield* waitFor(0.3);
  yield* N.label.opacity(1, 0.4);
  yield* sequence(0.3, ...N.rows.map(r => r.opacity(1, 0.4)));
  yield* cap.set('In contact with the toolbox maintainers');
  yield* waitFor(3.0);
  beat('end');
});
