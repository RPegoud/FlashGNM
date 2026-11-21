import {Node, Rect, Txt} from '@motion-canvas/2d';
import {createRef, useLogger, useTime, waitFor} from '@motion-canvas/core';
import {C, FONT} from './theme';

/** Marks a beat boundary; scripts/render.mjs splits the rendered video at these. */
export function beat(name: string) {
  useLogger().info(`BEAT ${name} ${useTime().toFixed(4)}`);
}

/** Top-left section header: green index + mint title. */
export function header(view: Node, index: string, title: string) {
  const group = createRef<Node>();
  view.add(
    <Node ref={group} opacity={0}>
      <Rect layout direction="row" gap={22} alignItems="center" x={-960 + 90} y={-540 + 80} offset={[-1, 0]}>
        <Txt text={index} fontFamily={FONT} fontWeight={600} fontSize={30} fill={C.green} />
        <Rect width={2} height={30} fill={C.dim} />
        <Txt text={title} fontFamily={FONT} fontWeight={500} fontSize={30} fill={C.mint} />
      </Rect>
    </Node>,
  );
  return group;
}

/** Bottom caption with a cross-fade setter. */
export function caption(view: Node) {
  const t = createRef<Txt>();
  view.add(
    <Txt
      ref={t}
      x={-960 + 90}
      y={540 - 90}
      offset={[-1, 0]}
      fontFamily={FONT}
      fontSize={38}
      fill={C.mint}
      text=""
      opacity={0}
    />,
  );
  function* set(text: string, d = 0.5) {
    if (t().opacity() > 0) yield* t().opacity(0, d / 2);
    t().text(text);
    yield* t().opacity(1, d / 2);
  }
  return {node: t, set};
}

export {waitFor};
