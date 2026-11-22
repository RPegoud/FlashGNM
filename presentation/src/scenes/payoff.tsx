import {Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, createSignal, easeInOutCubic, easeOutCubic, sequence, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {TileGrid} from '../lib/tiles';
import talk from '../data/talk.json';

// Talk clip C: what a replicated study costs. 51 x 51 x 20 = 52,020 simulations per
// person; one person 6.8 days -> 76 s; 70 people 1.3 years -> 1.5 h. Numbers:
// src/data/talk.json (results/landscape_griffa_full, results/*_sim_baseline).

const LS = talk.landscape;
const comma = (x: number) => Math.round(x).toLocaleString('en-US');

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '4', 'What a replicated study costs');
  const cap = caption(view);

  // ======================================================== beat A: 51 x 51 x 20
  beat('A_count');
  const GX = -420, GY = 0, SZ = 380;
  const REP = LS.repeats;
  // 19 echoes behind the grid (plain outlined squares), then the grid itself on top
  const echoes: Rect[] = [];
  for (let r = REP - 1; r >= 1; r--) {
    const e = (<Rect x={GX + r * 9} y={GY - r * 9} size={SZ} radius={4} fill={'#0a1f12'} stroke={C.deep} lineWidth={2} opacity={0} />) as Rect;
    view.add(e);
    echoes.unshift(e);
  }
  const grid0 = new TileGrid({x: GX, y: GY});
  grid0.k = LS.grid[0];
  grid0.size = SZ;
  grid0.base = C.green;
  view.add(grid0);
  const lab = createRef<Node>();
  view.add(
    <Node ref={lab} opacity={0}>
      <Txt text={`${LS.grid[0]} values of η`} fontFamily={FONT} fontSize={28} fill={C.mint} x={GX} y={GY + SZ / 2 + 40} />
      <Txt text={`${LS.grid[1]} values of γ`} fontFamily={FONT} fontSize={28} fill={C.mint} x={GX - SZ / 2 - 40} y={GY} rotation={-90} />
    </Node>,
  );
  const eqRef = createRef<Node>();
  const t1 = createRef<Txt>(), t2 = createRef<Txt>(), t3 = createRef<Txt>(), t4 = createRef<Txt>();
  const EX = 120;
  view.add(
    <Node ref={eqRef}>
      <Txt ref={t1} text={`${LS.grid[0]} η values`} fontFamily={MONO} fontSize={48} fill={C.mint} x={EX} y={-150} offset={[-1, 0]} opacity={0} />
      <Txt ref={t2} text={`× ${LS.grid[1]} γ values`} fontFamily={MONO} fontSize={48} fill={C.mint} x={EX} y={-70} offset={[-1, 0]} opacity={0} />
      <Txt ref={t3} text={`× ${REP} repeats`} fontFamily={MONO} fontSize={48} fill={C.mint} x={EX} y={10} offset={[-1, 0]} opacity={0} />
      <Txt ref={t4} text={`= ${comma(LS.sims_per_subject)} simulations`} fontFamily={MONO} fontSize={56} fontWeight={700} fill={C.green} x={EX} y={110} offset={[-1, 0]} opacity={0} />
    </Node>,
  );
  const per = createRef<Txt>();
  view.add(<Txt ref={per} text="per person" fontFamily={FONT} fontSize={32} fill={C.mint} x={EX} y={170} offset={[-1, 0]} opacity={0} />);

  yield* title().opacity(1, 0.6);
  yield* all(grid0.reveal(1, 1.2, easeInOutCubic), lab().opacity(1, 0.5), t1().opacity(1, 0.4),
    cap.set('One fit searches a grid of parameter pairs'));
  yield* t2().opacity(1, 0.4);
  yield* waitFor(0.4);
  // echo the grid 20 times: one network is random, so each point is repeated
  const x20 = createRef<Txt>();
  view.add(<Txt ref={x20} text={`× ${REP}`} fontFamily={MONO} fontSize={40} fontWeight={700} fill={C.green}
    x={GX + (REP - 1) * 9 + SZ / 2 + 20} y={GY - (REP - 1) * 9 - SZ / 2 + 10} offset={[-1, 0]} opacity={0} />);
  yield* all(sequence(0.05, ...echoes.map((e, k) => e.opacity(1 - (0.7 * k) / (REP - 2), 0.15))), t3().opacity(1, 0.4), x20().opacity(1, 0.6),
    cap.set('Each point is simulated 20 times, because one network is random'));
  yield* all(t4().opacity(1, 0.5), t4().scale(1.05, 0.25).to(1, 0.25), per().opacity(1, 0.5));
  yield* waitFor(2.0);

  // ======================================================== beat B: one person
  beat('B_person');
  yield* all(grid0.opacity(0, 0.5), ...echoes.map(e => e.opacity(0, 0.5)), lab().opacity(0, 0.5), eqRef().opacity(0, 0.5), per().opacity(0, 0.5), x20().opacity(0, 0.5));
  const big = (y: number, before: string, after: string) => {
    const g = createRef<Node>();
    const b = createRef<Txt>(), a = createRef<Txt>(), arr = createRef<Txt>();
    view.add(
      <Node ref={g}>
        <Txt ref={b} text={before} fontFamily={MONO} fontSize={104} fill={C.muted} x={-90} y={y} offset={[1, 0]} opacity={0} />
        <Txt ref={arr} text="→" fontFamily={MONO} fontSize={96} fill={C.mint} x={0} y={y} opacity={0} />
        <Txt ref={a} text={after} fontFamily={MONO} fontSize={104} fontWeight={700} fill={C.green} x={90} y={y} offset={[-1, 0]} opacity={0} />
      </Node>,
    );
    return {g, b, a, arr};
  };
  const l1 = createRef<Txt>();
  view.add(<Txt ref={l1} text={`One person at ${comma(LS.n)} regions`} fontFamily={FONT} fontSize={36} fill={C.mint} y={-250} opacity={0} />);
  const r1 = big(-150, `${(LS.toolbox_s_per_subject / 86400).toFixed(1)} days`, `${Math.round(LS.ours_s_per_subject_median)} s`);
  yield* all(l1().opacity(1, 0.5), cap.set(`${comma(LS.sims_per_subject)} simulations: the reference toolbox vs our kernel, one H100`));
  yield* r1.b().opacity(1, 0.5);
  yield* waitFor(0.4);
  yield* r1.arr().opacity(1, 0.3);
  yield* all(r1.a().opacity(1, 0.5), r1.a().scale(1.06, 0.25).to(1, 0.25));
  yield* waitFor(1.8);

  // ======================================================== beat C: a whole study
  beat('C_cohort');
  const l2 = createRef<Txt>();
  view.add(<Txt ref={l2} text={`${LS.subjects} people`} fontFamily={FONT} fontSize={36} fill={C.mint} y={60} opacity={0} />);
  const years = LS.toolbox_s_cohort_total / 86400 / 365;
  const hours = LS.ours_s_cohort_total / 3600;
  const r2 = big(160, `${years.toFixed(1)} years`, `${hours.toFixed(1)} h`);
  const e2e = LS.toolbox_s_cohort_total / LS.ours_s_cohort_total;
  const foot = createRef<Txt>();
  view.add(<Txt ref={foot} fontFamily={FONT} fontSize={24} fill={C.muted} y={300} textAlign={'center'} lineHeight={36} opacity={0}
    text={`toolbox time extrapolated from timed steps, simulation only; ours measured, including evaluation\nend to end ≈ ${comma(Math.round(e2e / 100) * 100)}×, distinct from the simulation-throughput gain`} />);
  yield* all(l2().opacity(1, 0.5), cap.set('Zoom out to a whole study'));
  yield* r2.b().opacity(1, 0.5);
  yield* waitFor(0.4);
  yield* r2.arr().opacity(1, 0.3);
  yield* all(r2.a().opacity(1, 0.5), r2.a().scale(1.06, 0.25).to(1, 0.25));
  yield* foot().opacity(1, 0.5);
  yield* waitFor(2.5);
  beat('end');
});
