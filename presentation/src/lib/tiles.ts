import {Node, NodeProps} from '@motion-canvas/2d';
import {createSignal} from '@motion-canvas/core';
import {ramp} from './theme';

const hex = (c: string) => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));
function mix(a: string, b: string, t: number): string {
  const A = hex(a), B = hex(b);
  return `rgb(${A.map((x, k) => Math.round(x + (B[k] - x) * t)).join(',')})`;
}

/**
 * k x k tiles in a square of `size` px, one per simulation, drawn on the canvas.
 * `reveal` fills tiles in reading order; `sweep` crossfades from `base` to a colour that
 * varies across the grid (one parameter pair per tile).
 */
export class TileGrid extends Node {
  public k = 32;
  public size = 400;
  public base = '#14B85A';
  public readonly reveal = createSignal(0);
  public readonly sweep = createSignal(0);

  public constructor(props: NodeProps) {
    super(props);
  }

  protected draw(context: CanvasRenderingContext2D) {
    const k = this.k, c = this.size / k, h = this.size / 2;
    const gap = c > 2.5 ? Math.max(1, c * 0.12) : c * 0.15; // keep tiles distinct
    const shown = Math.floor(this.reveal() * k * k);
    const sw = this.sweep();
    context.save();
    for (let t = 0; t < shown; t++) {
      const x = t % k, y = Math.floor(t / k);
      context.fillStyle = sw > 0 ? mix(this.base, ramp(0.25 + 0.75 * ((x + (k - 1 - y)) / (2 * (k - 1)))), sw) : this.base;
      context.fillRect(-h + x * c, -h + y * c, c - gap, c - gap);
    }
    context.restore();
    super.draw(context);
  }
}
