import {Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeInOutCubic, easeOutCubic, sequence, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import talk from '../data/talk.json';

// Talk clip C: what the speedup buys. Memory per simulation and batch size, then the
// cohort-scale "what took X now runs in Y". Numbers: src/data/talk.json.

const M = talk.memory.per_n;
const LS = talk.landscape;
const comma = (x: number) => Math.round(x).toLocaleString('en-US');
const mb = (x: number) => (x >= 10 ? x.toFixed(0) : x.toFixed(1));

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '3', 'What the speedup buys');
  const cap = caption(view);

  // ======================================================== beat A: memory per simulation
  beat('A_memory');
  const X0 = -300, X1 = 600;
  const MAX = Math.ceil(Math.max(...M.map(m => m.toolbox.mb_per_sim)) / 50) * 50;
  const bw = (v: number) => (v / MAX) * (X1 - X0);
  const head = createRef<Txt>();
  view.add(<Txt ref={head} text="GPU memory per simulation" fontFamily={FONT} fontSize={34} fontWeight={600} fill={C.mint} x={-870} y={-330} offset={[-1, 0]} opacity={0} />);
  type Row = {g: Node; bar: Rect; lab: Txt; w: number};
  const rows: Row[] = [];
  M.forEach((m, gi) => {
    const y0 = -210 + gi * 230;
    view.add(<Txt text={`${comma(m.n)} regions`} fontFamily={FONT} fontSize={32} fill={C.mint} x={-870} y={y0 + 40} offset={[-1, 0]} opacity={0} ref={createRef()} />);
    [['Reference toolbox', m.toolbox.mb_per_sim, C.dim, C.muted], ['Our GPU kernel', m.ours.mb_per_sim, C.green, C.green]].forEach(([label, v, fill, tc], k) => {
      const y = y0 + k * 80;
      const bar = createRef<Rect>();
      const lab = createRef<Txt>();
      const g = (
        <Node opacity={0}>
          <Txt text={label as string} fontFamily={FONT} fontSize={28} fill={tc as string} x={X0 - 30} y={y} offset={[1, 0]} />
          <Rect ref={bar} x={X0} y={y} offset={[-1, 0]} width={0} height={48} radius={4} fill={fill as string} />
          <Txt ref={lab} text={`${mb(v as number)} MB`} fontFamily={MONO} fontSize={32} fill={tc as string} x={X0 + Math.max(bw(v as number), 3) + 20} y={y} offset={[-1, 0]} opacity={0} />
        </Node>
      ) as Node;
      view.add(g);
      rows.push({g, bar: bar(), lab: lab(), w: Math.max(bw(v as number), 3)});
    });
  });
  const groupLabels = view.children().filter(n => n instanceof Txt && (n as Txt).text().endsWith('regions')) as Txt[];

  yield* all(title().opacity(1, 0.6), head().opacity(1, 0.6), ...groupLabels.map(t => t.opacity(1, 0.6)),
    cap.set('Each simulation needs about 15× less GPU memory'));
  for (const r of rows) {
    yield* r.g.opacity(1, 0.2);
    yield* all(r.bar.width(r.w, 0.7, easeOutCubic));
    yield* r.lab.opacity(1, 0.3);
  }
  const m2 = M[M.length - 1];
  const note = createRef<Txt>();
  view.add(<Txt ref={note} fontFamily={FONT} fontSize={30} fill={C.mint} x={-870} y={290} offset={[-1, 0]} opacity={0}
    text={`At ${comma(m2.n)} regions, one 80 GB H100: the toolbox runs out of memory at ${comma(m2.toolbox.oom_at[0])} simulations; we run ${comma(m2.ours.B)}`} />);
  yield* note().opacity(1, 0.6);
  yield* cap.set('So far more simulations fit on one GPU at once');
  yield* waitFor(2.0);

  // ======================================================== beat B: a whole cohort
  beat('B_cohort');
  yield* all(head().opacity(0, 0.5), note().opacity(0, 0.5), ...groupLabels.map(t => t.opacity(0, 0.5)), ...rows.map(r => r.g.opacity(0, 0.5)));
  const c = createRef<Node>();
  const before = createRef<Txt>();
  const arrow = createRef<Txt>();
  const after = createRef<Txt>();
  const years = LS.toolbox_s_cohort_total / 86400 / 365;
  const hours = LS.ours_s_cohort_total / 3600;
  view.add(
    <Node ref={c}>
      <Txt fontFamily={FONT} fontSize={38} fill={C.mint} y={-170} opacity={0}
        text={`A whole study: ${LS.subjects} people × ${comma(LS.n)} regions × ${comma(LS.sims_per_subject)} networks each`} />
      <Txt ref={before} fontFamily={MONO} fontSize={120} fill={C.muted} x={-300} y={-10} opacity={0} text={`${years.toFixed(1)} years`} />
      <Txt ref={arrow} fontFamily={MONO} fontSize={110} fill={C.mint} x={110} y={-10} opacity={0} text="→" />
      <Txt ref={after} fontFamily={MONO} fontSize={120} fill={C.green} fontWeight={700} x={430} y={-10} opacity={0} text={`${hours.toFixed(1)} h`} />
      <Txt fontFamily={FONT} fontSize={26} fill={C.muted} y={140} opacity={0}
        text="reference toolbox (extrapolated, simulation only) vs our kernel (measured, simulation + scoring) · one H100" />
    </Node>,
  );
  const [line1, , , , foot] = c().children() as Txt[];
  yield* all(line1.opacity(1, 0.6), cap.set('What previously took over a year now runs over lunch'));
  yield* before().opacity(1, 0.6);
  yield* waitFor(0.5);
  yield* all(arrow().opacity(1, 0.4), arrow().x(130, 0.4, easeOutCubic));
  yield* all(after().opacity(1, 0.5), after().scale(1.08, 0.25).to(1, 0.25));
  yield* foot.opacity(1, 0.5);
  yield* waitFor(2.5);
  beat('end');
});
