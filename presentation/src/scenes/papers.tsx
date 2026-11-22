import {Circle, Img, Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {all, createRef, easeInOutCubic, easeOutCubic, sequence, Vector2, waitFor} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import akarcaPng from '../assets/papers/akarca2021-01.png';
import oldham22Png from '../assets/papers/oldham2022-01.png';
import oldham25Png from '../assets/papers/oldham2025-01.png';
import s1 from '../data/science_s1.json';
import s2 from '../data/science_s2.json';
import s3 from '../data/science_s3.json';

// Talk clip D: three published analyses, re-run with replicated fits.
// Left: the paper's first page and venue. Right: one sentence + a compact chart of our
// result. Numbers: src/data/science_s{1,2,3}.json. Full versions: clips 07-09 (backup).

const comma = (x: number) => Math.round(x).toLocaleString('en-US');
const sg = (x: number) => (x >= 0 ? '+' : '') + x.toFixed(2);

const PAPERS = [
  {img: akarcaPng, venue: 'Nature Communications · 2021', who: 'Akarca et al.'},
  {img: oldham22Png, venue: 'Science Advances · 2022', who: 'Oldham et al.'},
  {img: oldham25Png, venue: 'Network Neuroscience · 2025', who: 'Oldham, Fornito & Ball'},
];

// right panel geometry
const RX0 = -180, RX1 = 860;
const CY = 170; // chart centre y
const CW = 760, CH = 300;

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const title = header(view, '5', 'Re-testing published findings');
  const cap = caption(view);

  // ---------------------------------------------------------------- cards
  const cards = PAPERS.map((p, k) => {
    const g = createRef<Node>();
    const img = createRef<Img>();
    view.add(
      <Node ref={g} x={-560 + k * 560} y={-20} opacity={0} scale={0.73}>
        <Img ref={img} src={p.img} width={520} radius={6} />
        <Txt text={p.venue} fontFamily={FONT} fontSize={30} fill={C.green} y={390} />
        <Txt text={p.who} fontFamily={FONT} fontSize={26} fill={C.muted} y={430} />
      </Node>,
    );
    return g;
  });

  // ======================================================== beat A: intro
  beat('A_intro');
  yield* title().opacity(1, 0.6);
  yield* all(sequence(0.25, ...cards.map(c => c().opacity(1, 0.6))),
    cap.set('Three published analyses, re-run on 70 people with 20 simulations per grid point'));
  yield* waitFor(2.0);

  // ---------------------------------------------------------------- shared right panel
  function panel(they: string, ours: string) {
    const g = createRef<Node>();
    view.add(
      <Node ref={g} opacity={0}>
        <Txt text={they} fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.mint} x={RX0} y={-340} offset={[-1, -1]} width={RX1 - RX0} textWrap lineHeight={48} />
        <Txt text={ours} fontFamily={FONT} fontSize={30} fill={C.muted} x={RX0} y={-220} offset={[-1, -1]} width={RX1 - RX0} textWrap lineHeight={42} />
      </Node>,
    );
    return g;
  }
  function* focus(k: number) {
    yield* all(...cards.map((c, j) => (j === k
      ? all(c().position(new Vector2(-630, -40), 0.8, easeInOutCubic), c().scale(0.85, 0.8, easeInOutCubic), c().opacity(1, 0.4))
      : c().opacity(0, 0.4))));
  }

  // ======================================================== beat B: Akarca 2021
  beat('B_akarca');
  const d = s1['033'];
  const p1 = panel(
    'Can the parameter correlation appear without individual differences?',
    '70 synthetic “twins”, generated from one shared parameter pair, fitted like real people',
  );
  // mini scatter
  const E0 = -2.1, E1 = -1.4, G0 = 1.0, G1 = 4.3;
  const mp = (e: number, g: number) => new Vector2(RX0 + 40 + ((e - E0) / (E1 - E0)) * (CW - 260), CY + CH / 2 - ((g - G0) / (G1 - G0)) * CH);
  const sc = createRef<Node>();
  const spread = (eta: number[], gamma: number[]) => {
    const seen = new Map<string, number>();
    return eta.map((e, k) => {
      const key = `${e.toFixed(2)},${gamma[k].toFixed(2)}`;
      const m = seen.get(key) ?? 0;
      seen.set(key, m + 1);
      return m === 0 ? Vector2.zero : new Vector2(5 * Math.sqrt(m) * Math.cos(m * 2.4), 5 * Math.sqrt(m) * Math.sin(m * 2.4));
    });
  };
  const offR = spread(d.real.eta, d.real.gamma);
  const offN = spread(d.null.eta, d.null.gamma);
  view.add(
    <Node ref={sc} opacity={0}>
      <Line points={[[RX0 + 40, CY - CH / 2 - 10], [RX0 + 40, CY + CH / 2], [RX0 + CW - 200, CY + CH / 2]]} stroke={C.muted} lineWidth={2} />
      <Txt text="η" fontFamily={FONT} fontSize={26} fill={C.muted} x={RX0 + CW - 180} y={CY + CH / 2} />
      <Txt text="γ" fontFamily={FONT} fontSize={26} fill={C.muted} x={RX0 + 40} y={CY - CH / 2 - 34} />
    </Node>,
  );
  const realDots = d.real.eta.map((e, k) => {
    const c = (<Circle position={mp(e, d.real.gamma[k]).add(offR[k])} size={10} fill={C.green} scale={0} />) as Circle;
    view.add(c);
    return c;
  });
  const twinDots = d.null.eta.map((e, k) => {
    const c = (<Circle position={mp(d.true[0], d.true[1])} size={10} fill={C.mint} opacity={0} />) as Circle;
    view.add(c);
    return c;
  });
  const leg1 = createRef<Node>();
  const LX = RX0 + CW - 120;
  view.add(
    <Node ref={leg1} opacity={0}>
      <Circle x={LX} y={CY - 90} size={16} fill={C.green} />
      <Txt text={`real  ${sg(d.corr_real)}`} fontFamily={MONO} fontSize={30} fill={C.green} x={LX + 24} y={CY - 90} offset={[-1, 0]} />
      <Circle x={LX} y={CY - 40} size={16} fill={C.mint} />
      <Txt text={`twins ${sg(d.corr_null)}`} fontFamily={MONO} fontSize={30} fill={C.mint} x={LX + 24} y={CY - 40} offset={[-1, 0]} />
      <Txt text="corr(η, γ)" fontFamily={FONT} fontSize={24} fill={C.muted} x={LX - 10} y={CY - 140} offset={[-1, 0]} />
    </Node>,
  );
  yield* all(focus(0), cap.set('Akarca et al. 2021: a null test'));
  yield* p1().opacity(1, 0.6);
  yield* all(sc().opacity(1, 0.4), sequence(0.01, ...realDots.map(c => c.scale(1, 0.3, easeOutCubic))));
  yield* sequence(0.01, ...twinDots.map((c, k) => all(c.opacity(1, 0.15), c.position(mp(d.null.eta[k], d.null.gamma[k]).add(offN[k]), 0.8, easeOutCubic))));
  yield* leg1().opacity(1, 0.5);
  yield* cap.set(`At ${d.n} regions, fitting noise reproduces a strong η–γ correlation`);
  yield* waitFor(1.6);
  yield* cap.set(`Only at ${comma(s1['500'].n)} regions do real people vary more than the twins`);
  yield* waitFor(2.0);

  // ======================================================== beat C: Oldham 2022
  beat('C_oldham2022');
  yield* all(p1().opacity(0, 0.4), sc().opacity(0, 0.4), leg1().opacity(0, 0.4),
    ...realDots.map(c => c.opacity(0, 0.4)), ...twinDots.map(c => c.opacity(0, 0.4)));
  const p2 = panel(
    'A single-run search selects the luckiest simulation.',
    `Generate ${s2.n_fresh} fresh networks there, and the apparent minimum disappears.`,
  );
  const QX0 = RX0 + 430, QX1 = RX1 - 150, QM = 0.3;
  const qx = (e: number) => QX0 + (e / QM) * (QX1 - QX0);
  const bars = [
    {lab: 'selected single run', v: s2.best.reported, sd: 0, col: C.green},
    {lab: `${s2.n_fresh} fresh runs, same parameters`, v: s2.heldout_search.mean, sd: s2.heldout_search.sd, col: C.mint},
    {lab: 'best replicated fit', v: s2.heldout_grid.mean, sd: s2.heldout_grid.sd, col: C.deep},
  ];
  const ch2 = createRef<Node>();
  view.add(
    <Node ref={ch2} opacity={0}>
      <Line points={[[QX0, CY + 140], [QX1, CY + 140]]} stroke={C.muted} lineWidth={2} />
      {[0, 0.1, 0.2, 0.3].map(t => (
        <Node>
          <Line points={[[qx(t), CY + 140], [qx(t), CY + 150]]} stroke={C.muted} lineWidth={2} />
          <Txt text={t.toFixed(1)} fontFamily={MONO} fontSize={22} fill={C.muted} x={qx(t)} y={CY + 172} />
        </Node>
      ))}
      <Txt text="fit error (lower is better)" fontFamily={FONT} fontSize={22} fill={C.muted} x={(QX0 + QX1) / 2} y={CY + 205} />
    </Node>,
  );
  const barNodes = bars.map((b, k) => {
    const y = CY - 60 + k * 70;
    const bar = createRef<Rect>();
    const g = (
      <Node opacity={0}>
        <Txt text={b.lab} fontFamily={FONT} fontSize={26} fill={b.col === C.deep ? C.mint : b.col} x={QX0 - 20} y={y} offset={[1, 0]} />
        <Rect ref={bar} x={QX0} y={y} offset={[-1, 0]} width={0} height={36} radius={4} fill={b.col} />
        <Txt text={b.v.toFixed(3)} fontFamily={MONO} fontSize={26} fill={b.col === C.deep ? C.mint : b.col} x={qx(b.v) + 16} y={y} offset={[-1, 0]} />
      </Node>
    ) as Node;
    view.add(g);
    return {g, bar: bar(), w: qx(b.v) - QX0};
  });
  yield* all(cards[0]().opacity(0, 0.4), focus(1), cap.set('Oldham et al. 2022: the winner’s curse'));
  yield* p2().opacity(1, 0.6);
  yield* ch2().opacity(1, 0.4);
  for (const b of barNodes) {
    yield* b.g.opacity(1, 0.2);
    yield* b.bar.width(b.w, 0.6, easeOutCubic);
  }
  yield* cap.set(`For ${s2.cohort.worse}/${s2.cohort.n} people, the replicated grid finds a better expected fit`);
  yield* waitFor(2.0);

  // ======================================================== beat D: Oldham 2025
  beat('D_oldham2025');
  yield* all(p2().opacity(0, 0.4), ch2().opacity(0, 0.4), ...barNodes.map(b => b.g.opacity(0, 0.4)));
  const m = s3.classic[2], g = s3.consensus[2];
  const p3 = panel(
    'Their main analysis used 200 left-hemisphere regions, partly to reduce compute.',
    `We extended the test to ${comma(m.n)} regions, whole brain.`,
  );
  const NB = s3.bins_mm.length - 1;
  const BX0 = RX0 + 80, BX1 = RX1 - 40;
  const bw = (BX1 - BX0) / NB;
  const bxc = (k: number) => BX0 + bw * (k + 0.5);
  const LG0 = Math.log10(0.5), LG1 = Math.log10(150);
  const BASE = CY + 150, BH = 300;
  const ly = (r: number) => BASE - ((Math.log10(Math.max(r, 0.5)) - LG0) / (LG1 - LG0)) * BH;
  const ch3 = createRef<Node>();
  view.add(
    <Node ref={ch3} opacity={0}>
      {[1, 10, 100].map(t => (
        <Node>
          <Line points={[[BX0, ly(t)], [BX1, ly(t)]]} stroke={t === 1 ? C.mint : C.faint} lineWidth={t === 1 ? 2 : 1} lineDash={t === 1 ? [10, 8] : []} />
          <Txt text={t === 1 ? 'chance' : `${t}×`} fontFamily={MONO} fontSize={22} fill={t === 1 ? C.mint : C.muted} x={BX0 - 12} y={ly(t)} offset={[1, 0]} />
        </Node>
      ))}
      {Array.from({length: NB}, (_, k) => {
        const lo = s3.bins_mm[k], hi = s3.bins_mm[k + 1];
        return <Txt text={hi === null ? `>${lo}` : `${lo}–${hi}`} fontFamily={MONO} fontSize={20} fill={C.muted} x={bxc(k)} y={BASE + 50} />;
      })}
      <Txt text="connection length (mm)" fontFamily={FONT} fontSize={22} fill={C.muted} x={(BX0 + BX1) / 2} y={BASE + 82} />
    </Node>,
  );
  const gb = Array.from({length: NB}, (_, k) => {
    const r = (<Rect x={bxc(k) + bw * 0.12} y={ly(1)} offset={[0, 1]} width={bw * 0.38} height={0} fill={C.mint} radius={[3, 3, 0, 0]} />) as Rect;
    view.add(r);
    return r;
  });
  const md = Array.from({length: NB}, (_, k) => {
    const c = (<Circle x={bxc(k) - bw * 0.22} y={ly(m.recall[k] / m.chance[k])} size={16} fill={C.green} stroke={C.bg} lineWidth={3} opacity={0} />) as Circle;
    view.add(c);
    return c;
  });
  const leg3 = createRef<Node>();
  view.add(
    <Node ref={leg3} opacity={0}>
      <Circle x={BX0 + 10} y={CY - 150} size={16} fill={C.green} />
      <Txt text="best-fitting model" fontFamily={FONT} fontSize={24} fill={C.green} x={BX0 + 30} y={CY - 150} offset={[-1, 0]} />
      <Rect x={BX0 + 290} y={CY - 150} width={16} height={20} fill={C.mint} />
      <Txt text="other people’s brains" fontFamily={FONT} fontSize={24} fill={C.mint} x={BX0 + 310} y={CY - 150} offset={[-1, 0]} />
    </Node>,
  );
  yield* all(cards[1]().opacity(0, 0.4), focus(2), cap.set('Oldham, Fornito & Ball 2025: long-range connections'));
  yield* p3().opacity(1, 0.6);
  yield* all(ch3().opacity(1, 0.4), leg3().opacity(1, 0.4), sequence(0.05, ...md.map(c => c.opacity(1, 0.3))));
  yield* waitFor(0.6);
  yield* sequence(0.08, ...gb.map((b, k) => b.height(ly(1) - ly(g.recall[k] / g.chance[k]), 0.6, easeOutCubic)));
  yield* cap.set(`At chance at every distance; other people’s brains predict the wiring up to ${Math.round(Math.max(...g.recall.map((r, k) => r / g.chance[k])))}× better`);
  yield* waitFor(2.5);
  beat('end');
});
