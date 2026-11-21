import {defineConfig} from 'vite';
import motionCanvas from '@motion-canvas/vite-plugin';
import ffmpeg from '@motion-canvas/ffmpeg';

// One Motion Canvas project per clip; each is served at /<name> in the editor.
export default defineConfig({
  plugins: [
    motionCanvas({
      project: ['./src/clip1_wiring.ts'],
    }),
    ffmpeg(),
  ],
});
