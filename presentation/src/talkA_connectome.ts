import {makeProject} from '@motion-canvas/core';
import './lib/fonts';
import scene from './scenes/connectome?scene';

export default makeProject({
  scenes: [scene],
});
