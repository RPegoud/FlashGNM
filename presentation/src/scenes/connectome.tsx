import {Line, Node, Rect, Txt, makeScene2D} from '@motion-canvas/2d';
import {
  all,
  cancel,
  createRef,
  easeInOutCubic,
  easeOutCubic,
  linear,
  sequence,
  ThreadGenerator,
  waitFor,
} from '@motion-canvas/core';
import {C, FONT, MONO} from '../lib/theme';
import {beat, caption, header} from '../lib/ui';
import {AdjMatrix, Brain3D} from '../lib/brain3d';
import talk from '../data/talk.json';
import con from '../data/connectome.json';

// Talk clip A: opener + what a connectome is and why we study it.
// Numbers: src/data/talk.json, src/data/connectome.json (scripts/talk_export.py).

const L = talk.landscape;
const days = (s: number) => (s / 86400).toFixed(1);
const comma = (x: number) => Math.round(x).toLocaleString('en-US');

export default makeScene2D(function* (view) {
  view.fill(C.bg);
  const cap = caption(view);

  // ======================================================== beat A: opener
  beat('A_open');
  const open = createRef<Node>();
  const before = createRef<Txt>();
  const arrow = createRef<Txt>();
  const after = createRef<Txt>();
  view.add(
    <Node ref={open}>
      <Txt fontFamily={FONT} fontSize={40} fill={C.mint} y={-150} opacity={0} ref={createRef()}
        text={`Fitting a brain model to one person at ${comma(L.n)} regions`} />
      <Txt ref={before} fontFamily={MONO} fontSize={120} fill={C.muted} x={-275} y={0} opacity={0} text={`${days(L.toolbox_s_per_subject)} days`} />
      <Txt ref={arrow} fontFamily={MONO} fontSize={110} fill={C.mint} x={125} y={0} opacity={0} text="→" />
      <Txt ref={after} fontFamily={MONO} fontSize={120} fill={C.green} fontWeight={700} x={415} y={0} opacity={0} text={`${Math.round(L.ours_s_per_subject_median)} s`} />
      <Txt fontFamily={FONT} fontSize={26} fill={C.muted} y={150} opacity={0}
        text={`${comma(L.sims_per_subject)} simulated networks · reference toolbox (extrapolated) vs our GPU kernel (measured) · one H100`} />
    </Node>,
  );
  const [line1, , , , foot] = open().children() as Txt[];
  yield* line1.opacity(1, 0.6);
  yield* before().opacity(1, 0.6);
  yield* waitFor(0.6);
  yield* all(arrow().opacity(1, 0.4), arrow().x(145, 0.4, easeOutCubic));
  yield* all(after().opacity(1, 0.5), after().scale(1.08, 0.25).to(1, 0.25));
  yield* foot.opacity(1, 0.5);
  yield* waitFor(2.0);

  // ======================================================== beat B: regions
  beat('B_regions');
  yield* open().opacity(0, 0.6);
  open().remove();
  const title = header(view, '1', 'Brain connectomes');

  const brain = new Brain3D({y: 20});
  brain.setData(con.xyz, con.hemi, con.edges);
  brain.edgeColor = C.green;
  brain.edgeAlpha(0.08);
  brain.nodeColors = [C.mint, C.green];
  brain.pitch(0.12);
  brain.mm(4.0);
  view.add(brain);
  const spin = yield (function* (): ThreadGenerator {
    for (;;) {
      brain.yaw(brain.yaw() + 0.005);
      yield;
    }
  })();

  const count = createRef<Txt>();
  view.add(<Txt ref={count} fontFamily={MONO} fontSize={30} fill={C.muted} x={870} y={-420} offset={[1, 0]} opacity={0} text="" />);

  yield* all(title().opacity(1, 0.6), cap.set(`MRI divides the brain into regions: here ${comma(con.n)}`));
  count().text(`${comma(con.n)} regions`);
  yield* all(brain.nodeReveal(1, 2.5, easeInOutCubic), count().opacity(1, 0.5));
  yield* waitFor(1.5);

  // ======================================================== beat C: fibres
  beat('C_fibres');
  yield* cap.set('Diffusion MRI traces the white-matter fibres between them');
  yield* all(brain.edgeReveal(1, 5, linear));
  count().text(`${comma(con.n)} regions · ${comma(con.n_edges)} connections`);
  yield* cap.set('Regions + fibres: a connectome, the brain’s wiring diagram');
  yield* waitFor(2.0);

  // ======================================================== beat D: matrix
  beat('D_matrix');
  const mat = new AdjMatrix({x: 470, y: 0, opacity: 1});
  mat.n = con.n;
  mat.edges = con.edges;
  mat.size = 540;
  view.add(mat);
  const firstHemi = con.hemi.filter(h => h === con.hemi[0]).length;
  const frame = createRef<Node>();
  const S = 540, MX = 470;
  const split = -S / 2 + (firstHemi / con.n) * S;
  view.add(
    <Node ref={frame} opacity={0}>
      <Rect x={MX} width={S + 8} height={S + 8} stroke={C.dim} lineWidth={2} />
      <Line points={[[MX + split, -S / 2], [MX + split, S / 2]]} stroke={C.muted} lineWidth={1.5} lineDash={[6, 6]} />
      <Line points={[[MX - S / 2, split], [MX + S / 2, split]]} stroke={C.muted} lineWidth={1.5} lineDash={[6, 6]} />
      <Txt text="region j →" fontFamily={FONT} fontSize={24} fill={C.muted} x={MX} y={-S / 2 - 34} />
      <Txt text="region i →" fontFamily={FONT} fontSize={24} fill={C.muted} x={MX - S / 2 - 34} y={0} rotation={-90} />
      <Txt text="right" fontFamily={FONT} fontSize={22} fill={C.muted} x={MX + (-S / 2 + split) / 2} y={S / 2 + 30} />
      <Txt text="left" fontFamily={FONT} fontSize={22} fill={C.muted} x={MX + (split + S / 2) / 2} y={S / 2 + 30} />
    </Node>,
  );
  yield* all(brain.x(-460, 1.0, easeInOutCubic), brain.mm(2.5, 1.0, easeInOutCubic), count().opacity(0, 0.4),
    cap.set(`The same network as a ${comma(con.n)} × ${comma(con.n)} matrix: a dot per connection`));
  yield* all(frame().opacity(1, 0.5), mat.reveal(1, 1.6, easeInOutCubic));
  yield* waitFor(2.0);

  // ======================================================== beat E: why
  beat('E_why');
  const why = createRef<Node>();
  const items = [
    'Wiring differs between people, through development and in disorders',
    'Generative models ask which simple rules grow a brain-like network',
    `Fitting one person means simulating ${comma(L.sims_per_subject)} networks`,
  ];
  const WX = 40;
  view.add(
    <Node ref={why}>
      <Txt text="Why study them?" fontFamily={FONT} fontSize={36} fontWeight={600} fill={C.green} x={WX} y={-210} offset={[-1, 0]} opacity={0} />
      {items.map((t, k) => (
        <Node opacity={0}>
          <Rect x={WX + 8} y={-110 + k * 110} width={12} height={12} radius={2} fill={C.green} />
          <Txt text={t} fontFamily={FONT} fontSize={32} fill={C.mint} x={WX + 36} y={-110 + k * 110} offset={[-1, 0]} width={760} textWrap />
        </Node>
      ))}
    </Node>,
  );
  yield* all(frame().opacity(0, 0.5), mat.opacity(0, 0.5), cap.set('Why connectomes matter'));
  mat.remove();
  const [whyTitle, ...whyItems] = why().children();
  yield* whyTitle.opacity(1, 0.5);
  for (const it of whyItems) {
    yield* it.opacity(1, 0.6);
    yield* waitFor(1.4);
  }
  yield* cap.set('That is the computational bottleneck we remove');
  yield* waitFor(2.0);
  cancel(spin);
  beat('end');
});
