import {makeProject} from '@motion-canvas/core';
import './lib/fonts';
import wiring from './scenes/wiring?scene';

export default makeProject({
  scenes: [wiring],
});
