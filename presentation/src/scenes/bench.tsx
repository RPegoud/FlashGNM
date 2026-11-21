import {makeScene2D, Txt} from '@motion-canvas/2d';
import {waitFor} from '@motion-canvas/core';
import {C, FONT} from '../lib/theme';
import {beat} from '../lib/ui';

// Placeholder: replace with the real clip.
export default makeScene2D(function* (view) {
  view.fill(C.bg);
  beat('A_placeholder');
  view.add(<Txt text="clip6_bench (todo)" fontFamily={FONT} fill={C.mint} />);
  yield* waitFor(1);
  beat('end');
});
