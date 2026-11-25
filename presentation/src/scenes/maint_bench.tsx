import {Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, createSignal, easeInOutCubic, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {TileGrid} from '../lib/tiles';
import data from '../data/bench.json';
import talk from '../data/talk.json';

// Maintainer video, scene 4: throughput (same rescaling bar chart as clip 6) and batch
// size at equal memory (same numbers as talk clip B). Numbers: src/data/bench.json,
// src/data/talk.json.

const comma = (x: number) => Math.round(x).toLocaleString('en-US');
function sims(x: number): string {
  if (x >= 100) return comma(x);
  if (x >= 1) return x.toFixed(1);
  return Number(x.toPrecision(2)).toString();
}
const sig3 = (x: number) => Number(x.toPrecision(3)).toLocaleString('en-US');
const H = data.headline;
const P = talk.parallel;

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '4', 'Benchmarks (one H100)');
  const cap = caption(view);

  // ======================================================== throughput
  beat('A_throughput');
  const gB = createRef<Node>();
  view.add(<Node ref={gB} />);
  const LX = -430, X0 = -400, X1 = 470;
  const xmax = createSignal(data.impls[0].sims_per_s);
  const bx = (v: number) => X0 + (v / xmax()) * (X1 - X0);
  const step = () => {
    const raw = xmax() / 5;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    return [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw) ?? 10 * p;
  };
  const tickLabel = (v: number) => {
    for (let dp = 0; dp <= 4; dp++)
      if (Math.abs(Number(v.toFixed(dp)) - v) < 1e-9) return v.toLocaleString('en-US', {minimumFractionDigits: dp, maximumFractionDigits: dp});
    return v.toPrecision(2);
  };
  const ROW0 = -235, ROWH = 82;
  const AXY = ROW0 + data.impls.length * ROWH - 20;
  const axis = createRef<Node>();
  gB().add(
    <Node ref={axis} opacity={0}>
      <Txt text={`n = ${comma(H.n)} nodes · B = ${comma(H.B)} simulations · T = ${comma(H.T)} edges`} fontFamily={FONT} fontSize={30} fill={C.muted} x={-870} y={-335} offset={[-1, 0]} />
      <Txt text="speedup vs toolbox" fontFamily={FONT} fontSize={26} fill={C.muted} x={870} y={ROW0 - 62} offset={[1, 0]} />
      <Line points={[[X0, AXY], [X1 + 20, AXY]]} stroke={C.muted} lineWidth={2} />
      {Array.from({length: 7}, (_, k) => (
        <Node x={() => bx(k * step())} opacity={() => (k * step() <= xmax() * 1.0001 ? 1 : 0)}>
          <Line points={[[0, ROW0 - 40], [0, AXY]]} stroke={C.faint} lineWidth={2} />
          <Line points={[[0, AXY], [0, AXY + 10]]} stroke={C.muted} lineWidth={2} />
          <Txt text={() => tickLabel(k * step())} fontFamily={MONO} fontSize={26} fill={C.muted} y={AXY + 34} />
        </Node>
      ))}
      <Txt text="simulations / s" fontFamily={FONT} fontSize={28} fill={C.mint} x={(X0 + X1) / 2} y={AXY + 82} />
      <Txt text={`~ extrapolated from ${data.probe_steps} timed steps`} fontFamily={FONT} fontSize={26} fill={C.muted} x={870} y={AXY + 82} offset={[1, 0]} />
    </Node>,
  );
  const rows = data.impls.map((d, r) => {
    const y = ROW0 + r * ROWH;
    const ours = !d.extrapolated;
    const best = r === data.impls.length - 1;
    const fill = best ? C.green : ours ? C.deep : C.dim;
    const grow = createSignal(0);
    const w = () => grow() * Math.max(3, bx(d.sims_per_s) - X0);
    const val = createRef<Txt>(), sp = createRef<Txt>();
    const group = (
      <Node y={y} opacity={0}>
        <Txt text={d.label} fontFamily={FONT} fontSize={30} fill={ours ? C.mint : C.muted} x={LX} offset={[1, 0]} />
        <Rect x={X0} offset={[-1, 0]} width={w} height={46} radius={4} fill={fill} stroke={ours ? fill : C.muted} lineWidth={ours ? 0 : 2} />
        <Txt ref={val} text={(d.extrapolated ? '~' : '') + sims(d.sims_per_s)} fontFamily={MONO} fontSize={30} fill={best ? C.green : C.mint} x={() => X0 + w() + 18} offset={[-1, 0]} opacity={0} />
        <Txt ref={sp} text={r === 0 ? '1×' : `~${sig3(d.speedup_vs_toolbox)}×`} fontFamily={MONO} fontSize={32} fill={best ? C.green : ours ? C.mint : C.muted} fontWeight={best ? 700 : 400} x={870} offset={[1, 0]} opacity={0} />
      </Node>
    ) as Node;
    gB().add(group);
    return {group, val, sp, grow, v: d.sims_per_s};
  });
  yield* title().opacity(1, 0.5);
  yield* all(axis().opacity(1, 0.5), cap.set('Every implementation on the same GPU; each new bar rescales the axis'));
  let shown = 0;
  for (const b of rows) {
    shown = Math.max(shown, b.v);
    yield* b.group.opacity(1, 0.2);
    yield* all(b.grow(1, 0.9, easeInOutCubic), xmax(shown, 0.9, easeInOutCubic));
    yield* all(b.val().opacity(1, 0.3), b.sp().opacity(1, 0.3));
    yield* waitFor(0.3);
  }
  yield* cap.set(`~${sig3(H.vs_toolbox)}× higher simulation throughput · n = ${comma(H.n)}, B = ${comma(H.B)}`);
  yield* waitFor(2.2);

  // ======================================================== batch size at equal memory
  beat('B_batch');
  yield* gB().opacity(0, 0.5);
  gB().remove();
  const BOX = 420, TX = -470, TY = -20;
  const tl = new TileGrid({x: TX, y: TY});
  tl.k = Math.round(Math.sqrt(P.toolbox.B));
  tl.size = BOX;
  tl.base = C.dim;
  const tr = new TileGrid({x: -TX, y: TY});
  tr.k = Math.round(Math.sqrt(P.ours_max.B));
  tr.size = BOX;
  tr.base = C.deep;
  view.add(tl);
  view.add(tr);
  const mem = createRef<Node>();
  view.add(
    <Node ref={mem} opacity={0}>
      <Txt text="Reference toolbox" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.muted} x={TX} y={TY - BOX / 2 - 60} />
      <Txt text="New kernel" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.green} x={-TX} y={TY - BOX / 2 - 60} />
      <Rect x={TX} y={TY} size={BOX + 16} stroke={C.dim} lineWidth={2} radius={6} />
      <Rect x={-TX} y={TY} size={BOX + 16} stroke={C.dim} lineWidth={2} radius={6} />
      <Txt text={`${comma(P.toolbox.B)} simulations`} fontFamily={MONO} fontSize={40} fill={C.muted} x={TX} y={TY + BOX / 2 + 60} />
      <Txt text={`${P.toolbox.peak_gb.toFixed(1)} GB peak GPU memory`} fontFamily={FONT} fontSize={26} fill={C.muted} x={TX} y={TY + BOX / 2 + 106} />
      <Txt text={`${comma(P.ours_max.B)} simulations`} fontFamily={MONO} fontSize={40} fill={C.green} x={-TX} y={TY + BOX / 2 + 60} />
      <Txt text={`${P.ours_max.peak_gb.toFixed(1)} GB peak GPU memory`} fontFamily={FONT} fontSize={26} fill={C.mint} x={-TX} y={TY + BOX / 2 + 106} />
    </Node>,
  );
  yield* all(mem().opacity(1, 0.5), cap.set(`Batch size at about the same memory, n = ${comma(P.n)}`));
  yield* all(tl.reveal(1, 1.6, easeInOutCubic), tr.reveal(1, 3.0, easeInOutCubic));
  yield* cap.set(`${Math.round(P.ours_max.B / P.toolbox.B)}× more simulations at once, each with its own (η, γ)`);
  yield* waitFor(2.5);
  beat('end');
});
