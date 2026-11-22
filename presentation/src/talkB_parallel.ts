import {makeProject} from '@motion-canvas/core';
import './lib/fonts';
import scene from './scenes/parallel?scene';

export default makeProject({
  scenes: [scene],
});
