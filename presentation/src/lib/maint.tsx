import {Latex, Node, Rect, Txt} from '@motion-canvas/2d';
import {all, createRef, createSignal} from '@motion-canvas/core';
import {C, FONT, MONO, ramp} from './theme';
import {Mat} from './gnm';

// Shared layout for the maintainer video: matrix in the centre-left, algorithm steps
// (LaTeX) top-right, a short description bottom-right.

export const CS = 22; // matrix cell size
export const MX = -390, MY = 20; // matrix centre
export const PX = 250; // left edge of the right column

/** P as a heatmap (plus an optional row-sum column). Colours are sqrt-scaled to max P. */
export function matrix(view: Node, P: Mat, withR = false, R?: number[]) {
  const n = P.length;
  const pmax = Math.max(...P.flat());
  const shade = (v: number) => ramp(0.12 + 0.88 * Math.sqrt(v / pmax));
  const g = (<Node opacity={0} />) as Node;
  const cells: Rect[][] = [];
  const pos = (i: number, j: number) => [MX + (j - (n - 1) / 2) * CS, MY + (i - (n - 1) / 2) * CS] as const;
  for (let i = 0; i < n; i++) {
    cells.push([]);
    for (let j = 0; j < n; j++) {
      const [x, y] = pos(i, j);
      const r = (<Rect x={x} y={y} size={CS - 3} radius={2} fill={shade(P[i][j])} stroke={C.mint} lineWidth={0} />) as Rect;
      cells[i].push(r);
      g.add(r);
    }
  }
  const H = (n * CS) / 2;
  g.add(<Txt text="P" fontFamily={FONT} fontSize={28} fill={C.muted} x={MX} y={MY - H - 30} />);
  const rCells: Rect[] = [];
  if (withR && R) {
    const rmax = Math.max(...R);
    const RX = MX + H + 34;
    g.add(<Txt text="R" fontFamily={FONT} fontSize={28} fill={C.muted} x={RX} y={MY - H - 30} />);
    R.forEach((v, i) => {
      const r = (<Rect x={RX} y={MY + (i - (n - 1) / 2) * CS} size={CS - 3} radius={2} fill={ramp(0.15 + 0.85 * (v / rmax))} stroke={C.mint} lineWidth={0} />) as Rect;
      rCells.push(r);
      g.add(r);
    });
  }
  view.add(g);
  return {g, cells, rCells, shade, H, n};
}

/** Numbered LaTeX steps with a cost tag each; focus(k) highlights step k. */
export function steps(view: Node, lines: {tex: string; cost: string}[], y0 = -330, dy = 54, fontSize = 30) {
  const g = (<Node opacity={0} />) as Node;
  const cursor = createRef<Rect>();
  g.add(<Rect ref={cursor} x={PX - 14} y={y0} width={640} height={dy - 6} radius={8} fill={C.faint} stroke={C.deep} lineWidth={2} offset={[-1, 0]} opacity={0} />);
  const tex: Latex[] = [];
  const costs: Txt[] = [];
  lines.forEach((l, k) => {
    const y = y0 + k * dy;
    g.add(<Txt text={`${k + 1}`} fontFamily={MONO} fontSize={24} fill={C.muted} x={PX} y={y} offset={[-1, 0]} />);
    const t = (<Latex tex={l.tex} fill={C.mint} fontSize={fontSize} x={PX + 36} y={y} offset={[-1, 0]} opacity={0.35} />) as Latex;
    const c = (<Txt text={l.cost} fontFamily={MONO} fontSize={24} fill={C.muted} x={PX + 610} y={y} offset={[1, 0]} opacity={0.35} />) as Txt;
    tex.push(t);
    costs.push(c);
    g.add(t);
    g.add(c);
  });
  view.add(g);
  function* focus(k: number, d = 0.3) {
    yield* all(
      cursor().opacity(1, d),
      cursor().y(y0 + k * dy, d),
      ...tex.map((t, j) => t.opacity(j === k ? 1 : j < k ? 0.55 : 0.35, d)),
      ...costs.map((c, j) => all(c.opacity(j === k ? 1 : j < k ? 0.55 : 0.35, d), c.fill(j === k ? (/[²³]/.test(lines[j].cost) ? C.mint : C.green) : C.muted, d))),
    );
  }
  return {g, focus};
}

/** Bottom-right description box with a cross-fading setter. */
export function description(view: Node, y = 250) {
  const t = createRef<Txt>();
  view.add(<Txt ref={t} text="" fontFamily={FONT} fontSize={30} fill={C.mint} x={PX} y={y} offset={[-1, -1]} width={620} textWrap lineHeight={42} opacity={0} />);
  function* set(text: string, d = 0.4) {
    if (t().opacity() > 0) yield* t().opacity(0, d / 2);
    t().text(text);
    yield* t().opacity(1, d / 2);
  }
  return {node: t, set};
}

/** A signal-driven counter under the matrix ("entries touched"). */
export function counter(view: Node, label: string, y: number) {
  const v = createSignal(0);
  const txt = createSignal('');
  const g = (<Node opacity={0} />) as Node;
  g.add(<Txt text={label} fontFamily={FONT} fontSize={24} fill={C.muted} x={MX} y={y} />);
  g.add(<Txt text={() => txt() || Math.round(v()).toLocaleString('en-US')} fontFamily={MONO} fontSize={44} fill={C.mint} x={MX} y={y + 46} />);
  view.add(g);
  return {g, v, txt};
}
