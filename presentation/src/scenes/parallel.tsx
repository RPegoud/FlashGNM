import {Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeInOutCubic, easeOutCubic, linear, sequence, useRandom, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {TileGrid} from '../lib/tiles';
import talk from '../data/talk.json';
import bench from '../data/bench.json';

// Talk clip B: why the GPU makes this possible. O(n^2) vs O(n) per step, then how many
// simulations fit side by side. Numbers: src/data/talk.json, src/data/bench.json.

const P = talk.parallel;
const comma = (x: number) => Math.round(x).toLocaleString('en-US');
const N = 14; // toy grid
const CS = 30;
const LX = -470, RX = 470, GY = -40;

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '2', 'Why the GPU makes it possible');
  const cap = caption(view);
  const rng = useRandom(3);

  // ---------------------------------------------------------------- two toy grids
  const mkGrid = (cx: number) => {
    const cells: Rect[] = [];
    const g = (<Node opacity={0} />) as Node;
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        const r = (<Rect x={cx + (j - (N - 1) / 2) * CS} y={GY + (i - (N - 1) / 2) * CS} size={CS - 4} radius={3} fill={C.faint} />) as Rect;
        cells.push(r);
        g.add(r);
      }
    view.add(g);
    return {g, cells};
  };
  const L = mkGrid(LX);
  const R = mkGrid(RX);
  const scan = createRef<Rect>();
  view.add(<Rect ref={scan} x={LX} y={GY - (N * CS) / 2} width={N * CS + 8} height={CS + 4} radius={4} stroke={C.mint} lineWidth={3} opacity={0} />);

  const H = (N * CS) / 2;
  const labels = createRef<Node>();
  view.add(
    <Node ref={labels} opacity={0}>
      <Txt text="Reference toolbox" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.muted} x={LX} y={GY - H - 110} />
      <Txt text="every step redoes all n² pairs" fontFamily={FONT} fontSize={28} fill={C.muted} x={LX} y={GY - H - 62} />
      <Txt text="Our GPU kernel" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.green} x={RX} y={GY - H - 110} />
      <Txt text="every step updates only what changed" fontFamily={FONT} fontSize={28} fill={C.mint} x={RX} y={GY - H - 62} />
    </Node>,
  );
  const cost = createRef<Node>();
  view.add(
    <Node ref={cost} opacity={0}>
      <Txt text={`work per step at n = ${comma(P.n)}`} fontFamily={FONT} fontSize={24} fill={C.muted} x={0} y={GY + H + 50} />
      <Txt text={`${comma(P.n * P.n)} pairs`} fontFamily={MONO} fontSize={48} fill={C.muted} x={LX} y={GY + H + 110} />
      <Txt text={`${P.rows_per_step} rows · ${comma(P.rows_per_step * P.n)}`} fontFamily={MONO} fontSize={48} fill={C.green} x={RX} y={GY + H + 110} />
      <Txt text="O(n²)" fontFamily={MONO} fontSize={28} fill={C.muted} x={LX} y={GY + H + 160} />
      <Txt text="O(n)" fontFamily={MONO} fontSize={28} fill={C.green} x={RX} y={GY + H + 160} />
    </Node>,
  );

  // ======================================================== beat A: one step, two ways
  beat('A_step');
  yield* title().opacity(1, 0.6);
  yield* all(L.g.opacity(1, 0.5), R.g.opacity(1, 0.5), labels().opacity(1, 0.5),
    cap.set('Each step adds one connection to the growing network'));
  yield* waitFor(0.4);
  for (let s = 0; s < 3; s++) {
    const i = rng.nextInt(0, N), j = (i + rng.nextInt(1, N)) % N;
    const rowCells = (r: number) => R.cells.slice(r * N, r * N + N);
    yield* all(
      // left: sweep every row, light every cell
      (function* () {
        scan().opacity(1);
        scan().y(GY - H + CS / 2 - CS);
        yield* all(scan().y(GY + H - CS / 2, 1.1, linear),
          sequence(1.1 / N, ...Array.from({length: N}, (_, r) => all(...L.cells.slice(r * N, r * N + N).map(c => c.fill(C.muted, 0.08))))));
        scan().opacity(0);
        yield* all(...L.cells.map(c => c.fill(C.faint, 0.35)));
      })(),
      // right: only rows i and j
      all(...[...rowCells(i), ...rowCells(j)].map(c => c.fill(C.green, 0.15).wait(0.6).to(C.faint, 0.35))),
    );
    if (s === 0) yield* all(cost().opacity(1, 0.5), cap.set('The reference redoes all n² pairs; we update a few rows'));
    yield* waitFor(0.3);
  }
  yield* waitFor(1.2);

  // ======================================================== beat B: side by side
  beat('B_side_by_side');
  const BOX = 420;
  const tl = new TileGrid({x: LX, y: GY + 10});
  tl.k = Math.round(Math.sqrt(P.toolbox.B));
  tl.size = BOX;
  tl.base = C.dim;
  const tr = new TileGrid({x: RX, y: GY + 10});
  tr.k = Math.round(Math.sqrt(P.ours_max.B));
  tr.size = BOX;
  tr.base = C.deep;
  view.add(tl);
  view.add(tr);
  const mem = createRef<Node>();
  view.add(
    <Node ref={mem} opacity={0}>
      <Rect x={LX} y={GY + 10} size={BOX + 16} stroke={C.dim} lineWidth={2} radius={6} />
      <Rect x={RX} y={GY + 10} size={BOX + 16} stroke={C.dim} lineWidth={2} radius={6} />
      <Txt text={`${comma(P.toolbox.B)} simulations`} fontFamily={MONO} fontSize={40} fill={C.muted} x={LX} y={GY + BOX / 2 + 70} />
      <Txt text={`${P.toolbox.peak_gb.toFixed(1)} GB of GPU memory`} fontFamily={FONT} fontSize={26} fill={C.muted} x={LX} y={GY + BOX / 2 + 118} />
      <Txt text={`${comma(P.ours_max.B)} simulations`} fontFamily={MONO} fontSize={40} fill={C.green} x={RX} y={GY + BOX / 2 + 70} />
      <Txt text={`${P.ours_max.peak_gb.toFixed(1)} GB of GPU memory`} fontFamily={FONT} fontSize={26} fill={C.mint} x={RX} y={GY + BOX / 2 + 118} />
    </Node>,
  );
  yield* all(L.g.opacity(0, 0.5), R.g.opacity(0, 0.5), cost().opacity(0, 0.5),
    cap.set('A GPU runs thousands of simulations side by side, if each one is small'));
  yield* mem().opacity(1, 0.5);
  yield* all(tl.reveal(1, 1.6, easeInOutCubic), tr.reveal(1, 3.0, easeInOutCubic));
  yield* cap.set(`Same memory, ${Math.round(P.ours_max.B / P.toolbox.B)}× more simulations at once (n = ${comma(P.n)}, one H100)`);
  yield* waitFor(2.0);

  // ======================================================== beat C: a sweep per launch
  beat('C_sweep');
  const sweepNote = createRef<Node>();
  view.add(
    <Node ref={sweepNote} opacity={0}>
      <Txt text="one (η, γ) for the whole batch" fontFamily={FONT} fontSize={26} fill={C.muted} x={LX} y={GY + BOX / 2 + 160} />
      <Txt text="each simulation its own (η, γ)" fontFamily={FONT} fontSize={26} fill={C.green} x={RX} y={GY + BOX / 2 + 160} />
    </Node>,
  );
  yield* all(sweepNote().opacity(1, 0.5),
    cap.set('Each simulation gets its own parameters: a whole sweep in one launch'));
  yield* tr.sweep(1, 1.5, easeInOutCubic);
  yield* waitFor(2.0);

  // ======================================================== beat D: payoff
  beat('D_payoff');
  const h = bench.headline;
  const pay = createRef<Node>();
  view.add(
    <Node ref={pay} opacity={0}>
      <Txt text={`~${Number(h.vs_toolbox.toPrecision(3)).toLocaleString('en-US')}×`} fontFamily={MONO} fontSize={150} fontWeight={700} fill={C.green} y={-60} />
      <Txt text="the reference toolbox’s throughput" fontFamily={FONT} fontSize={40} fill={C.mint} y={60} />
      <Txt text={`${comma(h.v2bi)} vs ~${Number(h.toolbox.toPrecision(2))} simulations / s · n = ${comma(h.n)}, B = ${comma(h.B)}, same H100`}
        fontFamily={MONO} fontSize={26} fill={C.muted} y={130} />
    </Node>,
  );
  yield* all(tl.opacity(0, 0.5), tr.opacity(0, 0.5), mem().opacity(0, 0.5), sweepNote().opacity(0, 0.5),
    cap.set('Less work per step × more simulations at once'));
  yield* all(pay().opacity(1, 0.6), pay().scale(0.96, 0).to(1, 0.6, easeOutCubic));
  yield* waitFor(2.5);
  beat('end');
});
