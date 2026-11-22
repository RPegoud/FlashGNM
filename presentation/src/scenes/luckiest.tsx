import {Circle, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  createRef,
  createSignal,
  easeInOutCubic,
  easeOutCubic,
  sequence,
  Vector2,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {PointCloud} from '../lib/pointcloud';
import data from '../data/science_s2.json';

// S2 "Luckiest of 10,000": Oldham-style single-run search vs a replicated grid.
// Numbers: src/data/science_s2.json (scripts/science_export.py).

const PX0 = -760, PX1 = -140, PY0 = -330, PY1 = 300;
const [E0, E1] = data.box.eta;
const [G0, G1] = data.box.gamma;
const sx = (PX1 - PX0) / (E1 - E0);
const sy = (PY1 - PY0) / (G1 - G0);
const pp = (e: number, g: number) => new Vector2(PX0 + (e - E0) * sx, PY1 - (g - G0) * sy);

// Energy axis (right panel).
const QX0 = 80, QX1 = 740, QMAX = 0.3;
const qx = (e: number) => QX0 + (e / QMAX) * (QX1 - QX0);

function sci(p: number): string {
  const [m, e] = p.toExponential(0).split('e');
  const sup = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const ex = String(Math.abs(+e)).split('').map(ch => sup[+ch]).join('');
  return `${m} × 10${+e < 0 ? '⁻' : ''}${ex}`;
}

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '08', 'Luckiest of 10,000');
  const cap = caption(view);
  const L = data.n_levels;

  // ---------------------------------------------------------------- plane
  const axes = createRef<Node>();
  view.add(
    <Node ref={axes} opacity={0}>
      <Rect x={(PX0 + PX1) / 2} y={(PY0 + PY1) / 2} width={PX1 - PX0} height={PY1 - PY0} stroke={C.dim} lineWidth={2} />
      {[-2, -1.5, -1, -0.5, 0].map(t => (
        <Node>
          <Line points={[[pp(t, 0).x, PY1], [pp(t, 0).x, PY1 + 10]]} stroke={C.muted} lineWidth={2} />
          <Txt text={t.toFixed(1)} fontFamily={MONO} fontSize={24} fill={C.muted} x={pp(t, 0).x} y={PY1 + 34} />
        </Node>
      ))}
      {[-10, -5, 0, 5, 10].map(t => (
        <Node>
          <Line points={[[PX0 - 10, pp(0, t).y], [PX0, pp(0, t).y]]} stroke={C.muted} lineWidth={2} />
          <Txt text={`${t}`} fontFamily={MONO} fontSize={24} fill={C.muted} x={PX0 - 20} y={pp(0, t).y} offset={[1, 0]} />
        </Node>
      ))}
      <Txt text="η" fontFamily={FONT} fontSize={30} fill={C.mint} x={(PX0 + PX1) / 2} y={PY1 + 74} />
      <Txt text="γ" fontFamily={FONT} fontSize={30} fill={C.mint} x={PX0 - 80} y={(PY0 + PY1) / 2} />
    </Node>,
  );
  const cloud = new PointCloud({});
  cloud.setPoints(
    data.points.map(([e, g, l]) => {
      const v = pp(e, g);
      return {x: v.x, y: v.y, group: l, color: ramp(0.18 + (0.82 * l) / (L - 1))};
    }),
    L,
  );
  view.add(cloud);

  // ---------------------------------------------------------------- wave counter
  const wave = createSignal(0);
  const waveTxt = createRef<Txt>();
  const subTxt = createRef<Txt>();
  view.add(
    <>
      <Txt ref={waveTxt} fontFamily={MONO} fontSize={34} fill={C.mint} x={QX0} y={-290} offset={[-1, 0]} opacity={0}
        text={() => `wave ${Math.max(1, Math.ceil(wave()))} / ${L}   ·   ${Math.round(wave() * data.per_level).toLocaleString('en-US')} networks`} />
      <Txt ref={subTxt} fontFamily={FONT} fontSize={26} fill={C.muted} x={QX0} y={-240} offset={[-1, 0]} opacity={0}
        text={`subject ${data.subject} · additive rule · n = ${data.n} · α not shown`} />
    </>,
  );

  // ======================================================== beat A: the search
  beat('A_search');
  yield* title().opacity(1, 0.6);
  yield* all(axes().opacity(1, 0.6), cap.set(`Oldham-style search: ${L} waves of ${data.per_level.toLocaleString('en-US')} points, one network each`),
    waveTxt().opacity(1, 0.6), subTxt().opacity(1, 0.6));
  for (let l = 0; l < L; l++) {
    yield* all(cloud.reveal[l](1, 1.0, easeInOutCubic), wave(l + 1, 1.0, easeInOutCubic));
  }
  const legend = createRef<Txt>();
  view.add(<Txt ref={legend} fontFamily={FONT} fontSize={26} fill={C.muted} x={QX0} y={-195} offset={[-1, 0]} opacity={0}
    text="later waves (brighter) sample near the low-energy valley" />);
  yield* legend().opacity(1, 0.5);
  yield* waitFor(1.2);

  // ======================================================== beat B: the luckiest point
  beat('B_luckiest');
  const bp = pp(data.best.eta, data.best.gamma);
  const ring = createRef<Circle>();
  const bestDot = createRef<Circle>();
  const bestLab = createRef<Txt>();
  view.add(
    <>
      <Circle ref={ring} position={bp} size={0} stroke={C.green} lineWidth={4} />
      <Circle ref={bestDot} position={bp} size={0} fill={C.green} />
      <Txt ref={bestLab} fontFamily={MONO} fontSize={30} fill={C.green} x={bp.x - 34} y={bp.y - 44} offset={[1, 0]} opacity={0}
        text={`reported ${data.best.reported.toFixed(3)}`} />
    </>,
  );
  yield* all(
    cloud.dim(0.35, 0.6),
    legend().opacity(0, 0.4),
    cap.set('It reports the luckiest single network'),
  );
  yield* all(bestDot().size(16, 0.4, easeOutCubic), ring().size(70, 0.6, easeOutCubic), bestLab().opacity(1, 0.5));
  yield* ring().size(40, 0.4);
  yield* waitFor(1.5);

  // ======================================================== beat C: fresh networks
  beat('C_fresh');
  const panel = createRef<Node>();
  const rows = [
    {y: -80, label: 'reported by the search (1 network)'},
    {y: 40, label: `same point, ${data.n_fresh} fresh networks`},
    {y: 160, label: `replicated grid’s optimum, ${data.n_fresh} fresh networks`},
  ];
  view.add(
    <Node ref={panel} opacity={0}>
      <Line points={[[QX0, 230], [QX1, 230]]} stroke={C.muted} lineWidth={2} />
      {[0, 0.1, 0.2, 0.3].map(t => (
        <Node>
          <Line points={[[qx(t), 230], [qx(t), 240]]} stroke={C.muted} lineWidth={2} />
          <Txt text={t.toFixed(1)} fontFamily={MONO} fontSize={24} fill={C.muted} x={qx(t)} y={262} />
        </Node>
      ))}
      <Txt text="energy (max of 4 KS statistics, lower is better)" fontFamily={FONT} fontSize={24} fill={C.muted} x={QX0} y={300} offset={[-1, 0]} />
    </Node>,
  );
  const rowNodes = rows.map(r => {
    const t = (<Txt text={r.label} fontFamily={FONT} fontSize={28} fill={C.mint} x={QX0} y={r.y - 36} offset={[-1, 0]} opacity={0} />) as Txt;
    view.add(t);
    return t;
  });
  const mk = (e: number, sd: number, y: number, color: string) => {
    const g = createRef<Node>();
    const wh = createRef<Line>();
    const dot = createRef<Circle>();
    const lab = createRef<Txt>();
    view.add(
      <Node ref={g}>
        <Line ref={wh} points={[[qx(e), y], [qx(e), y]]} stroke={color} lineWidth={4} opacity={sd > 0 ? 1 : 0} />
        <Circle ref={dot} x={qx(e)} y={y} size={0} fill={color} />
        <Txt ref={lab} fontFamily={MONO} fontSize={28} fill={color} x={qx(e + sd) + 22} y={y} offset={[-1, 0]} opacity={0}
          text={sd > 0 ? `${e.toFixed(3)} ± ${sd.toFixed(3)}` : e.toFixed(3)} />
      </Node>,
    );
    return {
      node: g,
      show: () => all(dot().size(20, 0.4, easeOutCubic),
        wh().points([[qx(e - sd), y], [qx(e + sd), y]], 0.6, easeOutCubic), lab().opacity(1, 0.5)),
    };
  };
  const rRep = mk(data.best.reported, 0, rows[0].y, C.green);
  const rFresh = mk(data.heldout_search.mean, data.heldout_search.sd, rows[1].y, C.mint);
  const rGrid = mk(data.heldout_grid.mean, data.heldout_grid.sd, rows[2].y, C.deep);

  yield* all(waveTxt().opacity(0, 0.4), subTxt().y(-290, 0.4), panel().opacity(1, 0.5),
    cap.set(`Grow ${data.n_fresh} fresh networks at that same point`));
  yield* all(rowNodes[0].opacity(1, 0.4), rRep.show());
  yield* waitFor(0.5);
  yield* all(rowNodes[1].opacity(1, 0.4), rFresh.show());
  // optimism bracket
  const opt = data.heldout_search.mean - data.best.reported;
  const br = createRef<Line>();
  const brLab = createRef<Txt>();
  view.add(
    <>
      <Line ref={br} points={[[qx(data.best.reported), -40], [qx(data.heldout_search.mean), -40]]} stroke={C.green}
        lineWidth={3} lineDash={[8, 6]} endArrow arrowSize={12} end={0} />
      <Txt ref={brLab} fontFamily={MONO} fontSize={26} fill={C.green} x={qx(data.heldout_search.mean) + 18}
        y={-40} offset={[-1, 0]} opacity={0} text={`+${opt.toFixed(3)} optimism`} />
    </>,
  );
  yield* all(br().end(1, 0.6), brLab().opacity(1, 0.6));
  yield* cap.set('Its reported fit was luck: fresh networks score much worse');
  yield* waitFor(1.2);

  // grid optimum on the plane + row 3
  const [ge, gg] = data.heldout_grid.optimum;
  const gp = pp(ge, gg);
  const sq = createRef<Rect>();
  const sqLab = createRef<Txt>();
  view.add(
    <>
      <Rect ref={sq} position={gp} size={0} fill={C.bg} stroke={C.mint} lineWidth={3} rotation={45} />
      <Txt ref={sqLab} fontFamily={FONT} fontSize={26} fill={C.mint} x={gp.x - 34} y={gp.y + 44} offset={[1, 0]} opacity={0} text="replicated grid’s optimum" />
    </>,
  );
  yield* all(sq().size(22, 0.4), sqLab().opacity(1, 0.4), rowNodes[2].opacity(1, 0.4), rGrid.show(),
    cap.set('A grid replicated at every point picks a better one'));
  yield* waitFor(2.0);

  // ======================================================== beat D: cohort
  beat('D_cohort');
  const c = data.cohort;
  const big = createRef<Node>();
  view.add(
    <Node ref={big} opacity={0}>
      <Txt fontFamily={FONT} fontSize={28} fill={C.muted} x={QX0} y={-250} offset={[-1, 0]} text={`All ${c.n} subjects, same rule and resolution:`} />
      <Txt fontFamily={MONO} fontSize={64} fill={C.green} x={QX0} y={-170} offset={[-1, 0]} text={`${c.worse} / ${c.n}`} />
      <Txt fontFamily={FONT} fontSize={30} fill={C.mint} x={QX0 + 330} y={-170} offset={[-1, 0]} text="search worse than the grid" />
      <Txt fontFamily={FONT} fontSize={24} fill={C.muted} x={QX0 + 330} y={-130} offset={[-1, 0]} text={`Wilcoxon p = ${sci(c.wilcoxon_p)}`} />
    </Node>,
  );
  const big2 = createRef<Node>();
  view.add(
    <Node ref={big2} opacity={0}>
      <Txt fontFamily={MONO} fontSize={64} fill={C.green} x={QX0} y={-50} offset={[-1, 0]} text={`+${c.median_optimism.toFixed(3)}`} />
      <Txt fontFamily={FONT} fontSize={30} fill={C.mint} x={QX0 + 330} y={-50} offset={[-1, 0]} text="median optimism of the reported fit" />
      <Txt fontFamily={FONT} fontSize={24} fill={C.muted} x={QX0 + 330} y={-10} offset={[-1, 0]} text={`subject ${data.subject} is typical: +${c.subject_optimism.toFixed(3)}`} />
    </Node>,
  );
  yield* all(
    panel().opacity(0, 0.5), ...rowNodes.map(r => r.opacity(0, 0.5)), subTxt().opacity(0, 0.5),
    br().opacity(0, 0.5), brLab().opacity(0, 0.5),
    ...[rRep, rFresh, rGrid].map(r => r.node().opacity(0, 0.5)),
  );
  yield* all(big().opacity(1, 0.6), cap.set('Fitting with one network per point is misleading'));
  yield* waitFor(0.6);
  yield* big2().opacity(1, 0.6);
  yield* waitFor(3.0);
  beat('end');
});
