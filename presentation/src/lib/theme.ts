// Palette taken from modal_logo.png: black background, two greens, mint text.
export const C = {
  bg: '#000000',
  green: '#7FEE64', // highlight
  deep: '#14B85A', // secondary / fills
  mint: '#DDFFDC', // text, equations
  muted: '#7E9A84', // secondary text
  dim: '#22362A', // inactive elements
  faint: '#13201A', // very low-emphasis fills
};

export const FONT = 'Inter';
export const MONO = 'JetBrains Mono';

/** Lerp from dim to bright green for a value in [0, 1] (heatmaps, probabilities). */
export function ramp(t: number): string {
  t = Math.max(0, Math.min(1, t));
  const a = [0x13, 0x20, 0x1a];
  const m = [0x14, 0xb8, 0x5a];
  const b = [0x7f, 0xee, 0x64];
  const [lo, hi, u] = t < 0.6 ? [a, m, t / 0.6] : [m, b, (t - 0.6) / 0.4];
  const c = lo.map((x, k) => Math.round(x + (hi[k] - x) * u));
  return '#' + c.map(x => x.toString(16).padStart(2, '0')).join('');
}
