import {makeProject} from '@motion-canvas/core';
import './lib/fonts';
import scene from './scenes/bench?scene';

export default makeProject({
  scenes: [scene],
});
