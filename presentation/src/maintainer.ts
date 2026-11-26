import {makeProject} from '@motion-canvas/core';
import './lib/fonts';
import reference from './scenes/maint_reference?scene';
import product from './scenes/maint_product?scene';
import kernel from './scenes/maint_kernel?scene';
import bench from './scenes/maint_bench?scene';

// One video for the toolbox maintainers: reference vs new kernel, then benchmarks.
export default makeProject({
  scenes: [reference, product, kernel, bench],
});
