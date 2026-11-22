import {Node, NodeProps} from '@motion-canvas/2d';
import {SimpleSignal, createSignal} from '@motion-canvas/core';

export interface CloudPoint {
  x: number; // screen px (local)
  y: number;
  group: number; // reveal group (e.g. search level)
  color: string;
}

/**
 * Thousands of points drawn straight onto the canvas (one Circle node per point would be
 * far too slow). Group g shows its first reveal[g] fraction of points; `dim` fades all.
 */
export class PointCloud extends Node {
  public points: CloudPoint[] = [];
  public reveal: SimpleSignal<number>[] = [];
  public readonly dim = createSignal(1);
  public radius = 2.5;

  public constructor(props: NodeProps) {
    super(props);
  }

  public setPoints(points: CloudPoint[], groups: number) {
    this.points = points;
    this.reveal = Array.from({length: groups}, () => createSignal(0));
  }

  protected draw(context: CanvasRenderingContext2D) {
    const counts = new Map<number, number>();
    const totals = new Map<number, number>();
    for (const p of this.points) totals.set(p.group, (totals.get(p.group) ?? 0) + 1);
    context.save();
    context.globalAlpha *= this.dim();
    for (const p of this.points) {
      const k = counts.get(p.group) ?? 0;
      counts.set(p.group, k + 1);
      const frac = this.reveal[p.group]?.() ?? 0;
      if (k >= frac * (totals.get(p.group) ?? 0)) continue;
      context.fillStyle = p.color;
      context.fillRect(p.x - this.radius, p.y - this.radius, 2 * this.radius, 2 * this.radius);
    }
    context.restore();
    super.draw(context);
  }
}
