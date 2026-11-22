import {Circle, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  createSignal,
  easeInOutCubic,
  easeOutCubic,
  sequence,
  SimpleSignal,
  Vector2,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import data from '../data/science_s1.json';

// S1 "Synthetic twins": real subjects' optima vs a null cohort grown at ONE (eta, gamma).
// Numbers: src/data/science_s1.json (scripts/science_export.py).

type Scale = typeof data['033'];

// Plot frame (screen px).
const PX0 = -700, PX1 = 160, PY0 = -330, PY1 = 300;

interface Axes {
  eta: [number, number];
  gamma: [number, number];
  etaTicks: number[];
  gammaTicks: number[];
}
const AX68: Axes = {eta: [-2.1, -1.4], gamma: [1.0, 4.3], etaTicks: [-2.0, -1.8, -1.6, -1.4], gammaTicks: [1, 2, 3, 4]};
const AX1000: Axes = {eta: [-1.45, -1.15], gamma: [0.85, 1.5], etaTicks: [-1.4, -1.3, -1.2], gammaTicks: [0.9, 1.1, 1.3, 1.5]};

function mapper(ax: Axes) {
  const sx = (PX1 - PX0) / (ax.eta[1] - ax.eta[0]);
  const sy = (PY1 - PY0) / (ax.gamma[1] - ax.gamma[0]);
  return {
    sx,
    sy,
    p: (e: number, g: number) => new Vector2(PX0 + (e - ax.eta[0]) * sx, PY1 - (g - ax.gamma[0]) * sy),
  };
}

/** Points that share a grid cell get a small sunflower offset (px) so all 70 are visible. */
function spread(eta: number[], gamma: number[]): Vector2[] {
  const seen = new Map<string, number>();
  return eta.map((e, k) => {
    const key = `${e.toFixed(2)},${gamma[k].toFixed(2)}`;
    const m = seen.get(key) ?? 0;
    seen.set(key, m + 1);
    if (m === 0) return Vector2.zero;
    const r = 7 * Math.sqrt(m);
    const a = m * 2.39996;
    return new Vector2(r * Math.cos(a), r * Math.sin(a));
  });
}

function axesNode(ax: Axes, ref: ReturnType<typeof createRef<Node>>) {
  const M = mapper(ax);
  return (
    <Node ref={ref} opacity={0}>
      <Line points={[[PX0, PY0 - 10], [PX0, PY1], [PX1 + 10, PY1]]} stroke={C.muted} lineWidth={2} />
      {ax.etaTicks.map(t => (
        <Node>
          <Line points={[[M.p(t, 0).x, PY1], [M.p(t, 0).x, PY1 + 10]]} stroke={C.muted} lineWidth={2} />
          <Txt text={t.toFixed(1)} fontFamily={MONO} fontSize={24} fill={C.muted} x={M.p(t, 0).x} y={PY1 + 34} />
        </Node>
      ))}
      {ax.gammaTicks.map(t => (
        <Node>
          <Line points={[[PX0 - 10, M.p(0, t).y], [PX0, M.p(0, t).y]]} stroke={C.muted} lineWidth={2} />
          <Line points={[[PX0, M.p(0, t).y], [PX1, M.p(0, t).y]]} stroke={C.faint} lineWidth={1} />
          <Txt text={t.toFixed(1)} fontFamily={MONO} fontSize={24} fill={C.muted} x={PX0 - 22} y={M.p(0, t).y} offset={[1, 0]} />
        </Node>
      ))}
      <Txt text="η  (distance exponent)" fontFamily={FONT} fontSize={28} fill={C.mint} x={(PX0 + PX1) / 2} y={PY1 + 78} />
      <Txt text="γ  (matching exponent)" fontFamily={FONT} fontSize={28} fill={C.mint} x={PX0 - 100} y={(PY0 + PY1) / 2} rotation={-90} />
    </Node>
  );
}

/** 8.4e-7 -> '8 × 10⁻⁷' */
function sci(p: number): string {
  const [m, e] = p.toExponential(0).split('e');
  const sup = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const ex = String(Math.abs(+e)).split('').map(ch => sup[+ch]).join('');
  return `${m} × 10${+e < 0 ? '⁻' : ''}${ex}`;
}

function counter(sig: SimpleSignal<number>, digits = 2, sign = true) {
  return () => {
    const v = sig();
    return (sign && v >= 0 ? '+' : '') + v.toFixed(digits);
  };
}

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '07', 'Synthetic twins');
  const cap = caption(view);

  const d68 = data['033'] as Scale;
  const d1000 = data['500'] as Scale;

  // ---------------------------------------------------------------- right panel
  const RX = 330; // left edge of the stats panel
  const corrReal = createSignal(0);
  const corrNull = createSignal(0);
  const sdReal = createSignal(0);
  const sdNull = createSignal(0);
  const panel = createRef<Node>();
  const rowReal = createRef<Node>();
  const rowNull = createRef<Node>();
  const sdHead = createRef<Txt>();
  const sdRowReal = createRef<Node>();
  const sdRowNull = createRef<Node>();
  view.add(
    <Node ref={panel} opacity={0}>
      <Txt text="corr(η, γ) across subjects" fontFamily={FONT} fontSize={28} fill={C.muted} x={RX} y={-300} offset={[-1, 0]} />
      <Node ref={rowReal}>
        <Circle x={RX + 12} y={-235} size={18} fill={C.green} />
        <Txt text="real subjects" fontFamily={FONT} fontSize={32} fill={C.mint} x={RX + 40} y={-235} offset={[-1, 0]} />
        <Txt text={counter(corrReal)} fontFamily={MONO} fontSize={56} fill={C.green} x={RX + 560} y={-235} offset={[1, 0]} />
      </Node>
      <Node ref={rowNull} opacity={0}>
        <Circle x={RX + 12} y={-160} size={18} fill={C.mint} />
        <Txt text="synthetic twins" fontFamily={FONT} fontSize={32} fill={C.mint} x={RX + 40} y={-160} offset={[-1, 0]} />
        <Txt text={counter(corrNull)} fontFamily={MONO} fontSize={56} fill={C.mint} x={RX + 560} y={-160} offset={[1, 0]} />
      </Node>
      <Txt ref={sdHead} text="spread of γ (SD)" fontFamily={FONT} fontSize={28} fill={C.muted} x={RX} y={-60} offset={[-1, 0]} opacity={0} />
      <Node ref={sdRowReal} opacity={0}>
        <Txt text="real" fontFamily={FONT} fontSize={30} fill={C.mint} x={RX + 40} y={0} offset={[-1, 0]} />
        <Txt text={counter(sdReal, 2, false)} fontFamily={MONO} fontSize={44} fill={C.green} x={RX + 560} y={0} offset={[1, 0]} />
      </Node>
      <Node ref={sdRowNull} opacity={0}>
        <Txt text="twins" fontFamily={FONT} fontSize={30} fill={C.mint} x={RX + 40} y={60} offset={[-1, 0]} />
        <Txt text={counter(sdNull, 2, false)} fontFamily={MONO} fontSize={44} fill={C.mint} x={RX + 560} y={60} offset={[1, 0]} />
      </Node>
    </Node>,
  );
  const note = createRef<Txt>();
  const note2 = createRef<Txt>();
  const note3 = createRef<Txt>();
  view.add(
    <>
      <Txt ref={note} fontFamily={FONT} fontSize={24} fill={C.muted} x={RX} y={160} offset={[-1, 0]} opacity={0}
        text={`n = ${d68.n} regions · ${d68.n_subjects} subjects`} />
      <Txt ref={note2} fontFamily={FONT} fontSize={24} fill={C.muted} x={RX} y={195} offset={[-1, 0]} opacity={0}
        text="each dot: best fit of a grid replicated 20 times" />
      <Txt ref={note3} fontFamily={FONT} fontSize={24} fill={C.muted} x={RX} y={230} offset={[-1, 0]} opacity={0}
        text={`optima lie on a ${d68.grid_step} grid; overlaps nudged apart`} />
    </>,
  );

  // ---------------------------------------------------------------- n = 68 plot
  const ax68 = createRef<Node>();
  view.add(axesNode(AX68, ax68));
  const M68 = mapper(AX68);
  const ellLayer = createRef<Node>();
  const realLayer = createRef<Node>();
  const nullLayer = createRef<Node>();
  view.add(
    <>
      <Node ref={ellLayer} />
      <Node ref={realLayer} />
      <Node ref={nullLayer} />
    </>,
  );

  const offR = spread(d68.real.eta, d68.real.gamma);
  const realDots: Circle[] = d68.real.eta.map((e, k) => {
    const c = (<Circle position={M68.p(e, d68.real.gamma[k]).add(offR[k])} size={14} fill={C.green} scale={0} />) as Circle;
    realLayer().add(c);
    return c;
  });

  // Bootstrap ellipses (1 SD of each subject's optimum over repeats), screen space.
  const ells: Line[] = d68.real_ellipses.map((el, k) => {
    const ctr = M68.p(d68.real.eta[k], d68.real.gamma[k]).add(offR[k]);
    const pts: Vector2[] = [];
    for (let s = 0; s <= 40; s++) {
      const t = (s / 40) * Math.PI * 2;
      const a = el.sd_major * Math.cos(t), b = el.sd_minor * Math.sin(t);
      const de = a * Math.cos(el.angle) - b * Math.sin(el.angle);
      const dg = a * Math.sin(el.angle) + b * Math.cos(el.angle);
      pts.push(ctr.add(new Vector2(de * M68.sx, -dg * M68.sy)));
    }
    const l = (<Line points={pts} stroke={C.deep} lineWidth={1.5} opacity={0} end={0} />) as Line;
    ellLayer().add(l);
    return l;
  });

  // ======================================================== beat A: real subjects
  beat('A_real');
  yield* title().opacity(1, 0.6);
  yield* all(ax68().opacity(1, 0.6), cap.set('70 real subjects: best-fit (η, γ) at 68 regions'));
  yield* all(
    sequence(0.025, ...realDots.map(c => c.scale(1, 0.35, easeOutCubic))),
    panel().opacity(1, 0.5),
    note().opacity(1, 0.5),
    note2().opacity(1, 0.5),
    note3().opacity(1, 0.5),
  );
  yield* corrReal(d68.corr_real, 1.2, easeInOutCubic);
  yield* waitFor(1.0);

  // ======================================================== beat B: uncertainty
  beat('B_uncertainty');
  yield* cap.set('Each subject’s own uncertainty runs along the same diagonal');
  yield* sequence(0.01, ...ells.map(l => all(l.opacity(0.55, 0.5), l.end(1, 0.6))));
  yield* waitFor(2.0);

  // ======================================================== beat C: the twins
  beat('C_twins');
  const [te, tg] = d68.true;
  const star = createRef<Node>();
  const starLab = createRef<Txt>();
  const sp = M68.p(te, tg);
  const starPts: Vector2[] = [];
  for (let s = 0; s < 10; s++) {
    const r = s % 2 === 0 ? 22 : 9;
    const a = -Math.PI / 2 + (s * Math.PI) / 5;
    starPts.push(new Vector2(r * Math.cos(a), r * Math.sin(a)));
  }
  view.add(
    <>
      <Node ref={star} position={sp} scale={0}>
        <Line points={starPts} closed fill={C.bg} stroke={C.mint} lineWidth={3} />
      </Node>
      <Txt ref={starLab} text={`one true (η, γ) = (${te}, ${tg})`} fontFamily={FONT} fontSize={28} fill={C.mint}
        x={sp.x + 36} y={sp.y - 40} offset={[-1, 0]} opacity={0} />
    </>,
  );
  yield* all(
    cap.set('70 synthetic subjects, all grown with the same parameters'),
    ...ells.map(l => l.opacity(0, 0.5)),
    ...realDots.map(c => c.opacity(0.25, 0.5)),
  );
  yield* all(star().scale(1, 0.5, easeOutCubic), starLab().opacity(1, 0.5));
  yield* waitFor(0.6);

  const offN = spread(d68.null.eta, d68.null.gamma);
  const nullDots: Circle[] = d68.null.eta.map((e, k) => {
    const c = (<Circle position={sp} size={14} fill={C.mint} opacity={0} />) as Circle;
    nullLayer().add(c);
    return c;
  });
  yield* all(rowNull().opacity(1, 0.4), starLab().opacity(0, 0.4));
  yield* all(
    sequence(0.015, ...nullDots.map((c, k) =>
      all(c.opacity(1, 0.2), c.position(M68.p(d68.null.eta[k], d68.null.gamma[k]).add(offN[k]), 1.0, easeOutCubic)))),
    corrNull(d68.corr_null, 1.8, easeInOutCubic),
  );
  yield* waitFor(0.5);
  yield* cap.set('Fitted the usual way, identical twins look just as “individual”');
  yield* all(sdHead().opacity(1, 0.4), sdRowReal().opacity(1, 0.4), sdRowNull().opacity(1, 0.4),
    sdReal(d68.sd_gamma_real, 1.0), sdNull(d68.sd_gamma_null, 1.0));
  yield* waitFor(2.5);

  // ======================================================== beat D: n = 1000
  beat('D_n1000');
  yield* all(
    ax68().opacity(0, 0.5), realLayer().opacity(0, 0.5), nullLayer().opacity(0, 0.5), star().opacity(0, 0.5),
    note().opacity(0, 0.5), note2().opacity(0, 0.5), note3().opacity(0, 0.5),
    cap.set('At 1000 regions the picture changes'),
  );
  const ax1000 = createRef<Node>();
  view.add(axesNode(AX1000, ax1000));
  const M1 = mapper(AX1000);
  const r1 = spread(d1000.real.eta, d1000.real.gamma);
  const n1 = spread(d1000.null.eta, d1000.null.gamma);
  const layer1 = createRef<Node>();
  view.add(<Node ref={layer1} />);
  const nd = d1000.null.eta.map((e, k) => {
    const c = (<Circle position={M1.p(e, d1000.null.gamma[k]).add(n1[k])} size={12} fill={C.mint} scale={0} />) as Circle;
    layer1().add(c);
    return c;
  });
  const rd = d1000.real.eta.map((e, k) => {
    const c = (<Circle position={M1.p(e, d1000.real.gamma[k]).add(r1[k])} size={12} fill={C.green} scale={0} />) as Circle;
    layer1().add(c);
    return c;
  });
  const [te1, tg1] = d1000.true;
  const star1 = createRef<Node>();
  view.add(
    <Node ref={star1} position={M1.p(te1, tg1)} scale={0}>
      <Line points={starPts} closed fill={C.bg} stroke={C.mint} lineWidth={3} />
    </Node>,
  );
  note().text(`n = ${d1000.n} regions · twins grown at (${te1}, ${tg1})`);
  note3().text('one twin parameter set per scale');
  note2().text(`F-test, γ variance real > twins: p = ${sci(d1000.F_p_gamma_real_gt_null)}`);
  yield* all(
    ax1000().opacity(1, 0.6),
    corrReal(d1000.corr_real, 0.8), corrNull(d1000.corr_null, 0.8),
    sdReal(d1000.sd_gamma_real, 0.8), sdNull(d1000.sd_gamma_null, 0.8),
  );
  yield* all(
    star1().scale(1, 0.4),
    sequence(0.01, ...nd.map(c => c.scale(1, 0.3, easeOutCubic))),
  );
  yield* sequence(0.015, ...rd.map(c => c.scale(1, 0.3, easeOutCubic)));
  yield* all(note().opacity(1, 0.5), note2().opacity(1, 0.5), note3().opacity(1, 0.5),
    cap.set('Real subjects now spread wider than their twins: here, individuals differ'));
  yield* waitFor(3.0);
  beat('end');
});
