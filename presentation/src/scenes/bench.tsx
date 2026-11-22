import {Circle, Latex, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, createSignal, easeInOutCubic, easeOutCubic, sequence, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import data from '../data/bench.json';

// Every number on screen comes from src/data/bench.json, written by scripts/bench_export.py
// from results/26-11-21_*_sim_*/results.jsonl (H100). Nothing below is typed by hand.

type Cell = {status: string; sims_per_s?: number; peak_gb?: number | null; extrapolated?: boolean};

// ---------------------------------------------------------------- formatting
const comma = (x: number) => Math.round(x).toLocaleString('en-US');
/** sims/s: integers above 100, one decimal above 1, two significant digits below. */
function sims(x: number): string {
  if (x >= 100) return comma(x);
  if (x >= 1) return x.toFixed(1);
  return Number(x.toPrecision(2)).toString();
}
/** Speedups to three significant digits, with thousands separators. */
const sig3 = (x: number) => Number(x.toPrecision(3)).toLocaleString('en-US');
const tilde = (c: Cell) => (c.extrapolated ? '~' : '');
const log10 = Math.log10;

// Decade tick labels.
function decade(e: number): string {
  const v = Math.pow(10, e);
  if (v >= 1e6) return `${v / 1e6}M`;
  if (v >= 1e3) return `${v / 1e3}k`;
  return v >= 1 ? `${v}` : `${Number(v.toPrecision(1))}`;
}

const H = data.headline;
const SIZE_LINE = `n = ${data.impls_n} nodes   ·   B = ${data.impls_B} simulations   ·   T = ${comma(H.T)} edges`;

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '4', 'Benchmarks');
  const cap = caption(view);

  // ======================================================== beat A: what is measured
  beat('A_setup');
  const gA = createRef<Node>();
  const fA = createRef<Latex>();
  const rowsA: Node[] = [];
  view.add(<Node ref={gA} />);
  gA().add(
    <Latex
      ref={fA}
      tex="\text{simulations / s} \;=\; \frac{B}{t_{\text{setup}} + t_{\text{run}}}"
      fill={C.mint}
      fontSize={60}
      y={-235}
      opacity={0}
    />,
  );
  // n / T / distances table.
  const colX = (k: number) => -220 + k * 270;
  const tableRows: [string, string[]][] = [
    ['nodes  n', data.ns.map(n => String(n))],
    ['edges  T', data.ns.map(n => comma((data.T as Record<string, number>)[String(n)]))],
    ['distances', data.ns.map(n => ((data.dist as Record<string, string>)[String(n)].startsWith('griffa') ? 'Griffa' : 'synthetic'))],
  ];
  tableRows.forEach(([label, vals], r) => {
    const y = -70 + r * 70;
    const g = (
      <Node opacity={0} y={y}>
        <Txt text={label} fontFamily={FONT} fontSize={30} fill={C.muted} x={-420} offset={[1, 0]} />
        {vals.map((v, k) => (
          <Txt text={v} fontFamily={MONO} fontSize={34} fill={r === 0 ? C.green : C.mint} x={colX(k)} />
        ))}
      </Node>
    ) as Node;
    rowsA.push(g);
    gA().add(g);
  });
  const notesA = createRef<Node>();
  gA().add(
    <Node ref={notesA} opacity={0}>
      <Txt
        text={`batch  B = ${data.B_grid.join(', ')}  (v2: up to ${Math.max(...data.B_big)}),  one (η, γ) per simulation`}
        fontFamily={FONT}
        fontSize={30}
        fill={C.mint}
        y={170}
      />
      <Txt
        text={`${data.gpu}   ·   simulation only   ·   median of ${data.repeats} runs   ·   T = mean Griffa edge count at that n`}
        fontFamily={FONT}
        fontSize={26}
        fill={C.muted}
        y={240}
      />
    </Node>,
  );

  yield* title().opacity(1, 0.6);
  yield* all(cap.set('Whole simulations per second, at realistic sizes, on one H100'), fA().opacity(1, 0.6));
  yield* sequence(0.25, ...rowsA.map(g => g.opacity(1, 0.5)));
  yield* notesA().opacity(1, 0.6);
  yield* waitFor(2.0);

  // ======================================================== beat B: implementations at n = 1000
  beat('B_impls');
  // Remove the setup slide instantly so this beat's file starts on a clean frame (talk slide 4).
  gA().remove();

  const gB = createRef<Node>();
  view.add(<Node ref={gB} />);
  const LX = -430;
  const X0 = -400;
  const X1 = 470;
  // Linear axis that rescales as bars arrive: the largest bar shown so far spans the full
  // width, earlier bars shrink, ticks re-pick a round step for the current range.
  const xmax = createSignal(data.impls[0].sims_per_s);
  const bx = (v: number) => X0 + (v / xmax()) * (X1 - X0);
  const step = () => {
    const raw = xmax() / 5;
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    return ([1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw) ?? 10 * p);
  };
  const tickLabel = (v: number) => {
    for (let dp = 0; dp <= 4; dp++) {
      if (Math.abs(Number(v.toFixed(dp)) - v) < 1e-9) return v.toLocaleString('en-US', {minimumFractionDigits: dp, maximumFractionDigits: dp});
    }
    return v.toPrecision(2);
  };
  const NTICK = 7;
  const ROW0 = -235;
  const ROWH = 82;
  const AXY = ROW0 + data.impls.length * ROWH - 20;

  const axisB = createRef<Node>();
  gB().add(
    <Node ref={axisB} opacity={0}>
      <Txt text={SIZE_LINE + '   ·   H100'} fontFamily={FONT} fontSize={30} fill={C.muted} x={-870} y={-335} offset={[-1, 0]} />
      <Txt text="speedup vs toolbox" fontFamily={FONT} fontSize={26} fill={C.muted} x={870} y={ROW0 - 62} offset={[1, 0]} />
      <Line points={[[X0, AXY], [X1 + 20, AXY]]} stroke={C.muted} lineWidth={2} />
      {Array.from({length: NTICK}, (_, k) => (
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

  type BarRow = {group: Node; val: Txt; sp: Txt; grow: ReturnType<typeof createSignal<number>>; v: number};
  const barRows: BarRow[] = data.impls.map((d, r) => {
    const y = ROW0 + r * ROWH;
    const ours = !d.extrapolated;
    const best = r === data.impls.length - 1;
    const fill = best ? C.green : ours ? C.deep : C.dim;
    const grow = createSignal(0); // 0 -> 1 as the bar first appears
    // tiny bars keep a 3 px sliver once a much faster implementation sets the scale
    const w = () => grow() * Math.max(3, bx(d.sims_per_s) - X0);
    const val = createRef<Txt>();
    const sp = createRef<Txt>();
    const group = (
      <Node y={y} opacity={0}>
        <Txt text={d.label} fontFamily={FONT} fontSize={30} fill={ours ? C.mint : C.muted} x={LX} offset={[1, 0]} />
        <Rect x={X0} offset={[-1, 0]} width={w} height={46} radius={4} fill={fill} stroke={ours ? fill : C.muted} lineWidth={ours ? 0 : 2} />
        <Txt ref={val} text={tilde(d) + sims(d.sims_per_s)} fontFamily={MONO} fontSize={30} fill={best ? C.green : C.mint} x={() => X0 + w() + 18} offset={[-1, 0]} opacity={0} />
        <Txt ref={sp} text={r === 0 ? '1×' : `~${sig3(d.speedup_vs_toolbox)}×`} fontFamily={MONO} fontSize={32} fill={best ? C.green : ours ? C.mint : C.muted} fontWeight={best ? 700 : 400} x={870} offset={[1, 0]} opacity={0} />
      </Node>
    ) as Node;
    gB().add(group);
    return {group, val: val(), sp: sp(), grow, v: d.sims_per_s};
  });

  // Each new bar grows while the axis rescales to the largest bar shown so far.
  let shownMax = 0;
  function* growRow(b: BarRow, d = 0.9) {
    shownMax = Math.max(shownMax, b.v);
    yield* b.group.opacity(1, 0.2);
    yield* all(b.grow(1, d, easeInOutCubic), xmax(shownMax, d, easeInOutCubic));
    yield* all(b.val.opacity(1, 0.3), b.sp.opacity(1, 0.3));
  }

  yield* all(axisB().opacity(1, 0.5), cap.set(`n = ${data.impls_n}, B = ${String(data.impls_B)}: every implementation on the same H100`));
  yield* growRow(barRows[0], 0.6);
  yield* waitFor(0.4);
  for (const b of barRows.slice(1)) {
    yield* growRow(b);
    yield* waitFor(0.35);
  }
  yield* cap.set(`~${sig3(H.vs_toolbox)}× higher simulation throughput · n = ${comma(H.n)}, B = ${comma(H.B)}, same H100`);
  yield* waitFor(2.2);

  // ======================================================== beat C: scaling with n
  beat('C_scaling');
  yield* gB().opacity(0, 0.5);
  gB().remove();

  const gC = createRef<Node>();
  view.add(<Node ref={gC} />);
  const PX0 = -600;
  const PX1 = 290;
  const PY0 = 240; // 10^-2
  const PY1 = -270; // 10^6
  const YE0 = -2;
  const YE1 = 6;
  // Categorical x (one column per n): log-n spacing crowds 1000 and 2000.
  const cx = (n: number) => PX0 + (data.ns.indexOf(n) / (data.ns.length - 1)) * (PX1 - PX0);
  const cy = (v: number) => PY0 + ((log10(v) - YE0) / (YE1 - YE0)) * (PY1 - PY0);

  const axisC = createRef<Node>();
  gC().add(
    <Node ref={axisC} opacity={0}>
      <Line points={[[PX0 - 40, PY1 - 20], [PX0 - 40, PY0 + 20], [PX1 + 40, PY0 + 20]]} stroke={C.muted} lineWidth={2} />
      {Array.from({length: YE1 - YE0 + 1}, (_, k) => YE0 + k).map(e => (
        <Node>
          <Line points={[[PX0 - 40, cy(10 ** e)], [PX1 + 40, cy(10 ** e)]]} stroke={C.faint} lineWidth={2} />
          <Txt text={decade(e)} fontFamily={MONO} fontSize={26} fill={C.muted} x={PX0 - 56} y={cy(10 ** e)} offset={[1, 0]} />
        </Node>
      ))}
      {data.ns.map(n => (
        <Node>
          <Txt text={`n = ${n}`} fontFamily={MONO} fontSize={28} fill={C.mint} x={cx(n)} y={PY0 + 54} />
          <Txt text={`T = ${comma((data.T as Record<string, number>)[String(n)])}`} fontFamily={MONO} fontSize={26} fill={C.muted} x={cx(n)} y={PY0 + 90} />
        </Node>
      ))}
      <Txt text="simulations / s  (log scale)" fontFamily={FONT} fontSize={28} fill={C.mint} x={PX0 - 40} y={PY1 - 52} offset={[-1, 0]} />
      <Txt text={`network size n (nodes),  B = ${String(data.impls_B)}`} fontFamily={FONT} fontSize={28} fill={C.muted} x={(PX0 + PX1) / 2} y={PY0 + 132} />
    </Node>,
  );

  const seriesC = data.scaling.map(s => {
    const ok = s.points.filter((p: Cell & {n: number}) => p.status === 'ok') as (Cell & {n: number})[];
    const best = s.key === 'v2bi';
    const color = best ? C.green : C.muted;
    const line = createRef<Line>();
    const dots: Circle[] = [];
    const g = (
      <Node>
        <Line ref={line} points={ok.map(p => [cx(p.n), cy(p.sims_per_s!)] as [number, number])} stroke={color} lineWidth={best ? 6 : 4} lineDash={s.key === 'dense' ? [14, 10] : []} end={0} />
      </Node>
    ) as Node;
    for (const p of ok) {
      const c = (<Circle x={cx(p.n)} y={cy(p.sims_per_s!)} size={best ? 20 : 16} fill={color} scale={0} />) as Circle;
      dots.push(c);
      g.add(c);
    }
    const last = ok[ok.length - 1];
    const label = (
      <Txt text={s.label} fontFamily={FONT} fontSize={30} fill={color} fontWeight={best ? 600 : 400} x={cx(last.n) + 26} y={cy(last.sims_per_s!)} offset={[-1, 0]} opacity={0} />
    ) as Txt;
    g.add(label);
    gC().add(g);
    return {line: line(), dots, label};
  });

  // OOM markers at n = 2000 for the series that ran out of memory.
  const oomC = createRef<Node>();
  const oomNames = data.scaling.filter(s => s.points.some((p: Cell) => p.status === 'oom')).map(s => s.label.replace(' (reference)', ''));
  gC().add(
    <Node ref={oomC} opacity={0}>
      <Txt text="✕" fontFamily={FONT} fontSize={30} fill={C.muted} x={cx(2000)} y={PY0} />
      <Txt text={`${oomNames.join(', ')}: OOM`} fontFamily={FONT} fontSize={26} fill={C.muted} x={cx(2000) + 26} y={PY0} offset={[-1, 0]} />
    </Node>,
  );

  // Speedup row above the plot, one entry per n, with a dashed gap marker.
  const tb = data.scaling.find(s => s.key === 'toolbox')!.points as (Cell & {n: number})[];
  const v2 = data.scaling.find(s => s.key === 'v2bi')!.points as (Cell & {n: number})[];
  const spC: Node[] = [];
  const SPY = -372;
  const spLabel = (<Txt text="vs toolbox" fontFamily={FONT} fontSize={26} fill={C.muted} x={PX0 - 56} y={SPY} offset={[1, 0]} opacity={0} />) as Txt;
  spC.push(spLabel);
  gC().add(spLabel);
  data.speedup_n.forEach((s, k) => {
    const n = s.n;
    const g = (
      <Node opacity={0}>
        <Txt text={s.speedup ? `~${sig3(s.speedup)}×` : 'toolbox OOM'} fontFamily={MONO} fontSize={s.speedup ? 34 : 26} fill={s.speedup ? C.green : C.muted} fontWeight={600} x={cx(n)} y={SPY} />
        {s.speedup ? (
          <Line points={[[cx(n), cy(tb[k].sims_per_s!) - 14], [cx(n), cy(v2[k].sims_per_s!) + 14]]} stroke={C.green} lineWidth={2} lineDash={[8, 8]} opacity={0.7} />
        ) : null}
      </Node>
    ) as Node;
    spC.push(g);
    gC().add(g);
  });

  yield* all(axisC().opacity(1, 0.5), cap.set(`Scaling with network size, B = ${String(data.impls_B)}`));
  for (const s of seriesC) {
    yield* all(s.line.end(1, 1.0), sequence(0.25, ...s.dots.map(d => d.scale(1, 0.3))));
    yield* s.label.opacity(1, 0.3);
  }
  yield* oomC().opacity(1, 0.4);
  yield* sequence(0.2, ...spC.map(g => g.opacity(1, 0.4)));
  yield* cap.set('The gap grows with n; the toolbox runs out of memory at n = 2000');
  yield* waitFor(2.2);

  // ======================================================== beat D: large batches
  beat('D_batch');
  yield* gC().opacity(0, 0.5);
  gC().remove();

  const gD = createRef<Node>();
  view.add(<Node ref={gD} />);
  const QX0 = -600;
  const QX1 = 120;
  const QY0 = 230; // 10^2
  const QY1 = -270; // 10^6
  const Bs = data.B_big;
  const dx = (B: number) => QX0 + ((Math.log2(B) - Math.log2(Bs[0])) / (Math.log2(Bs[Bs.length - 1]) - Math.log2(Bs[0]))) * (QX1 - QX0);
  const dy = (v: number) => QY0 + ((log10(v) - 2) / 4) * (QY1 - QY0);

  const axisD = createRef<Node>();
  gD().add(
    <Node ref={axisD} opacity={0}>
      <Line points={[[QX0 - 40, QY1 - 20], [QX0 - 40, QY0 + 20], [QX1 + 40, QY0 + 20]]} stroke={C.muted} lineWidth={2} />
      {[2, 3, 4, 5, 6].map(e => (
        <Node>
          <Line points={[[QX0 - 40, dy(10 ** e)], [QX1 + 40, dy(10 ** e)]]} stroke={C.faint} lineWidth={2} />
          <Txt text={decade(e)} fontFamily={MONO} fontSize={26} fill={C.muted} x={QX0 - 56} y={dy(10 ** e)} offset={[1, 0]} />
        </Node>
      ))}
      {Bs.map(B => (
        <Txt text={String(B)} fontFamily={MONO} fontSize={26} fill={C.mint} x={dx(B)} y={QY0 + 54} />
      ))}
      <Txt text="simulations / s  (log scale)" fontFamily={FONT} fontSize={28} fill={C.mint} x={QX0 - 40} y={QY1 - 52} offset={[-1, 0]} />
      <Txt text="batch size B  (log scale),  v2 + batched init" fontFamily={FONT} fontSize={28} fill={C.muted} x={(QX0 + QX1) / 2} y={QY0 + 100} />
    </Node>,
  );

  const shades = [C.green, C.deep, C.mint, C.muted];
  const seriesD = data.bigB.map((s, k) => {
    const ok = s.points.filter(p => p.status === 'ok');
    const oom = s.points.filter(p => p.status !== 'ok');
    const color = shades[k];
    const line = createRef<Line>();
    const g = (
      <Node>
        <Line ref={line} points={ok.map(p => [dx(p.B), dy(p.sims_per_s!)] as [number, number])} stroke={color} lineWidth={k === 0 ? 6 : 4} end={0} />
      </Node>
    ) as Node;
    const dots = ok.map(p => {
      const c = (<Circle x={dx(p.B)} y={dy(p.sims_per_s!)} size={16} fill={color} scale={0} />) as Circle;
      g.add(c);
      return c;
    });
    const last = ok[ok.length - 1];
    const lastY = dy(last.sims_per_s!);
    const extra: Node[] = oom.map(p => (<Txt text="✕" fontFamily={FONT} fontSize={30} fill={C.muted} x={dx(p.B)} y={lastY} opacity={0} />) as Txt);
    extra.forEach(e => g.add(e));
    const gain = Math.max(...ok.map(p => p.sims_per_s!)) / ok[0].sims_per_s!;
    const labelX = dx(Bs[Bs.length - 1]) + 26;
    const label = (
      <Txt fontFamily={FONT} fontSize={30} fill={color} x={labelX} y={lastY} offset={[-1, 0]} opacity={0}
        text={`n = ${s.n}   ${gain >= 1.005 ? '×' + gain.toFixed(2) : 'flat'}${oom.length ? `, OOM ≥ ${oom[0].B}` : ''}`} />
    ) as Txt;
    g.add(label);
    gD().add(g);
    return {line: line(), dots, label, extra};
  });
  const gainNote = createRef<Txt>();
  gD().add(
    <Txt ref={gainNote} text={`× = best / B = ${Bs[0]}`} fontFamily={FONT} fontSize={26} fill={C.muted} x={dx(Bs[Bs.length - 1]) + 26} y={QY1 - 52} offset={[-1, 0]} opacity={0} />,
  );

  // Headline callout on the right.
  const best = data.best;
  const callout = createRef<Node>();
  gD().add(
    <Node ref={callout} opacity={0} x={700}>
      <Txt text={comma(best.sims_per_s)} fontFamily={MONO} fontWeight={700} fontSize={84} fill={C.green} y={-80} />
      <Txt text="simulations / s" fontFamily={FONT} fontSize={32} fill={C.mint} y={0} />
      <Txt text={`n = ${best.n},  B = ${best.B}`} fontFamily={MONO} fontSize={28} fill={C.muted} y={60} />
      <Txt text={`T = ${comma((data.T as Record<string, number>)[String(best.n)])} edges each`} fontFamily={MONO} fontSize={28} fill={C.muted} y={102} />
    </Node>,
  );

  yield* all(axisD().opacity(1, 0.5), cap.set('Kernel v2 with batched init: more simulations per launch'));
  yield* sequence(
    0.4,
    ...seriesD.map(s => all(s.line.end(1, 1.0), sequence(0.18, ...s.dots.map(d => d.scale(1, 0.3))))),
  );
  yield* all(...seriesD.map(s => s.label.opacity(1, 0.4)), ...seriesD.flatMap(s => s.extra.map(e => e.opacity(1, 0.4))), gainNote().opacity(1, 0.4));
  yield* all(callout().opacity(1, 0.6), cap.set(`Larger batches fill the GPU: ${comma(best.sims_per_s)} sims/s at n = ${best.n}`));
  yield* waitFor(2.2);

  // ======================================================== beat E: memory
  beat('E_memory');
  yield* gD().opacity(0, 0.5);
  gD().remove();

  const gE = createRef<Node>();
  view.add(<Node ref={gE} />);
  const MX0 = -400;
  const CAP = data.gpu_mem_gb;
  const MX1 = 260; // position of the GPU capacity
  const mx = (gb: number) => MX0 + (gb / CAP) * (MX1 - MX0);
  const MR0 = -225;
  const MRH = 82;
  const MAY = MR0 + data.memory.length * MRH - 20;

  const axisE = createRef<Node>();
  gE().add(
    <Node ref={axisE} opacity={0}>
      <Txt text={`n = ${String(data.mem_n)} nodes   ·   B = ${String(data.mem_B)} simulations   ·   peak GPU memory`} fontFamily={FONT} fontSize={30} fill={C.muted} x={-870} y={-335} offset={[-1, 0]} />
      <Line points={[[MX0, MAY], [MX1 + 20, MAY]]} stroke={C.muted} lineWidth={2} />
      {[0, 20, 40, 60, 80].map(g => (
        <Node>
          <Line points={[[mx(g), MR0 - 40], [mx(g), MAY]]} stroke={C.faint} lineWidth={2} />
          <Txt text={`${g}`} fontFamily={MONO} fontSize={26} fill={C.muted} x={mx(g)} y={MAY + 34} />
        </Node>
      ))}
      <Line points={[[mx(CAP), MR0 - 50], [mx(CAP), MAY]]} stroke={C.mint} lineWidth={2} lineDash={[8, 8]} />
      <Txt text={`H100: ${CAP.toFixed(0)} GB`} fontFamily={FONT} fontSize={26} fill={C.mint} x={mx(CAP)} y={MR0 - 72} />
      <Txt text="peak memory, GB  (linear)" fontFamily={FONT} fontSize={28} fill={C.mint} x={(MX0 + MX1) / 2} y={MAY + 82} />
    </Node>,
  );

  const memRows = data.memory.map((d, r) => {
    const y = MR0 + r * MRH;
    const ok = d.status === 'ok';
    const ours = !d.extrapolated && ok;
    const fill = d.key === 'v2' ? C.green : ours ? C.deep : C.dim;
    const w = ok ? mx(d.peak_gb!) - MX0 : mx(CAP) - MX0;
    const bar = createRef<Rect>();
    const val = createRef<Txt>();
    const group = (
      <Node y={y} opacity={0}>
        <Txt text={d.label} fontFamily={FONT} fontSize={30} fill={ours ? C.mint : C.muted} x={LX} offset={[1, 0]} />
        <Rect ref={bar} x={MX0} offset={[-1, 0]} width={0} height={46} radius={4} fill={ok ? fill : '#00000000'} stroke={ok ? fill : C.muted} lineWidth={ok ? 0 : 2} lineDash={ok ? [] : [10, 8]} />
        <Txt ref={val} text={ok ? `${d.peak_gb!.toFixed(1)} GB` : 'out of memory'} fontFamily={ok ? MONO : FONT} fontSize={30} fill={ok ? (d.key === 'v2' ? C.green : C.mint) : C.muted} x={ok ? MX0 + w + 18 : mx(CAP) + 18} offset={[-1, 0]} opacity={0} />
      </Node>
    ) as Node;
    gE().add(group);
    return {group, bar: bar(), val: val(), w, key: d.key};
  });

  // State model: bytes per node pair x n^2 x B.
  const sg = data.state_gb;
  const model = createRef<Node>();
  const protoRow = data.memory.findIndex(d => d.key === 'proto');
  const v2Row = data.memory.findIndex(d => d.key === 'v2');
  const v2biRow = data.memory.findIndex(d => d.key === 'v2bi');
  gE().add(
    <Node ref={model} opacity={0}>
      <Txt text="state = bytes per pair × n² × B" fontFamily={FONT} fontSize={28} fill={C.muted} x={870} y={MR0 - 72} offset={[1, 0]} />
      <Latex tex={`9\\,\\text{B} \\to ${sg.proto_9B.toFixed(1)}\\ \\text{GB}`} fill={C.mint} fontSize={36} x={870} y={MR0 + protoRow * MRH} offset={[1, 0]} />
      <Latex tex={`3\\,\\text{B} \\to ${sg.v2_3B.toFixed(1)}\\ \\text{GB}`} fill={C.green} fontSize={36} x={870} y={MR0 + v2Row * MRH} offset={[1, 0]} />
      <Txt text="+ init temporaries" fontFamily={FONT} fontSize={26} fill={C.muted} x={870} y={MR0 + v2biRow * MRH} offset={[1, 0]} />
    </Node>,
  );
  const bytesNote = createRef<Txt>();
  gE().add(
    <Txt ref={bytesNote} text="prototype: A (bool) + C (int32) + P (fp32)   →   v2: A (bool) + C (int16), P recomputed" fontFamily={FONT} fontSize={26} fill={C.muted} x={870} y={MAY + 82 + 46} offset={[1, 0]} opacity={0} />,
  );

  yield* all(axisE().opacity(1, 0.5), cap.set(`Memory at n = ${String(data.mem_n)}: the baselines do not fit on the GPU`));
  yield* sequence(0.25, ...memRows.slice(0, 3).map(m => all(m.group.opacity(1, 0.3), m.bar.width(m.w, 0.7, easeOutCubic), m.val.opacity(1, 0.7))));
  yield* waitFor(0.5);
  yield* sequence(0.35, ...memRows.slice(3).map(m => all(m.group.opacity(1, 0.3), m.bar.width(m.w, 0.7, easeOutCubic), m.val.opacity(1, 0.7))));
  yield* all(model().opacity(1, 0.6), bytesNote().opacity(1, 0.6), cap.set('Dropping the stored P: 9 → 3 bytes per node pair'));
  yield* waitFor(2.2);

  // ======================================================== beat F: headline
  beat('F_headline');
  yield* gE().opacity(0, 0.5);
  gE().remove();

  const gF = createRef<Node>();
  view.add(<Node ref={gF} opacity={0} />);
  const big = createRef<Txt>();
  gF().add(
    <>
      <Txt ref={big} text={`~${sig3(H.vs_toolbox)}×`} fontFamily={FONT} fontWeight={700} fontSize={190} fill={C.green} y={-150} scale={0.85} />
      <Txt text="more simulations per second than the toolbox" fontFamily={FONT} fontSize={42} fill={C.mint} y={-10} />
      <Txt text={`${sims(H.v2bi)} vs ~${sims(H.toolbox)} sims/s   ·   ~${sig3(H.vs_dense)}× vs the dense baseline (~${sims(H.dense)})`} fontFamily={MONO} fontSize={28} fill={C.mint} y={75} />
      <Txt text={`v2 + batched init   ·   ${SIZE_LINE}`} fontFamily={MONO} fontSize={26} fill={C.muted} y={135} />
      <Txt text={data.gpu + '   ·   same GPU, whole simulations, setup included'} fontFamily={MONO} fontSize={26} fill={C.muted} y={177} />
      <Txt text={`~ baselines extrapolated from ${data.probe_steps} timed steps; the toolbox runs all B simulations with one (η, γ), which favours it`} fontFamily={FONT} fontSize={26} fill={C.muted} y={250} />
    </>,
  );
  yield* all(gF().opacity(1, 0.7), big().scale(1, 0.9, easeOutCubic), cap.set('Same conditions, same GPU'));
  yield* waitFor(3.0);
  beat('end');
});
