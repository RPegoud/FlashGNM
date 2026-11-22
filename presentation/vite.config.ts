import {defineConfig} from 'vite';
import motionCanvas from '@motion-canvas/vite-plugin';
import ffmpeg from '@motion-canvas/ffmpeg';

// One Motion Canvas project per clip; each is served at /<name> in the editor.
export default defineConfig({
  plugins: [
    motionCanvas({
      project: [
        './src/clip1_wiring.ts',
        './src/clip2_naive.ts',
        './src/clip3_chain.ts',
        './src/clip4_matrices.ts',
        './src/clip5_kernel.ts',
        './src/clip6_bench.ts',
        './src/clip7_twins.ts',
        './src/clip8_luckiest.ts',
        './src/clip9_chance.ts',
      ],
    }),
    ffmpeg(),
  ],
});
