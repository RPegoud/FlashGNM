import {Node, NodeProps} from '@motion-canvas/2d';
import {createSignal} from '@motion-canvas/core';

/**
 * A connectome drawn straight onto the canvas: region centroids (MNI mm, centred) rotated
 * about the vertical axis, perspective-projected; edges batched into one path.
 */
export class Brain3D extends Node {
  public xyz: number[][] = [];
  public hemi: number[] = [];
  public edges: number[][] = [];
  private order: number[] = []; // node reveal order

  public readonly yaw = createSignal(0);
  public readonly pitch = createSignal(0.3);
  public readonly mm = createSignal(3); // px per mm
  public readonly nodeReveal = createSignal(0);
  public readonly edgeReveal = createSignal(0);
  public readonly edgeAlpha = createSignal(0.16);
  public nodeColors: [string, string] = ['#DDFFDC', '#7FEE64'];
  public edgeColor = '#14B85A';

  public constructor(props: NodeProps) {
    super(props);
  }

  public setData(xyz: number[][], hemi: number[], edges: number[][]) {
    this.xyz = xyz;
    this.hemi = hemi;
    this.edges = edges;
    // deterministic shuffle for the node reveal
    this.order = xyz.map((_, k) => k);
    let s = 12345;
    for (let k = this.order.length - 1; k > 0; k--) {
      s = (s * 1103515245 + 12345) % 2147483648;
      const r = s % (k + 1);
      [this.order[k], this.order[r]] = [this.order[r], this.order[k]];
    }
  }

  private project(): {X: Float32Array; Y: Float32Array; f: Float32Array} {
    const n = this.xyz.length;
    const X = new Float32Array(n), Y = new Float32Array(n), f = new Float32Array(n);
    const cy = Math.cos(this.yaw()), sy = Math.sin(this.yaw());
    const cp = Math.cos(this.pitch()), sp = Math.sin(this.pitch());
    const s = this.mm();
    const Dz = 450; // viewer distance, mm
    for (let k = 0; k < n; k++) {
      const [x, y, z] = this.xyz[k];
      const x1 = x * cy - y * sy;
      const y1 = x * sy + y * cy;
      const y2 = y1 * cp - z * sp; // depth (towards viewer is negative)
      const z2 = y1 * sp + z * cp;
      const p = Dz / (Dz + y2);
      X[k] = x1 * s * p;
      Y[k] = -z2 * s * p;
      f[k] = p;
    }
    return {X, Y, f};
  }

  protected draw(context: CanvasRenderingContext2D) {
    if (this.xyz.length) {
      const {X, Y, f} = this.project();
      const ne = Math.floor(this.edgeReveal() * this.edges.length);
      if (ne > 0) {
        context.save();
        context.globalAlpha *= this.edgeAlpha();
        context.strokeStyle = this.edgeColor;
        context.lineWidth = 1;
        // Stroke in small batches: within one path overlaps paint once, across paths the
        // alpha accumulates, so dense tracts read brighter.
        const BATCH = 48;
        for (let e0 = 0; e0 < ne; e0 += BATCH) {
          context.beginPath();
          for (let e = e0; e < Math.min(ne, e0 + BATCH); e++) {
            const [a, b] = this.edges[e];
            context.moveTo(X[a], Y[a]);
            context.lineTo(X[b], Y[b]);
          }
          context.stroke();
        }
        context.restore();
      }
      const nn = Math.floor(this.nodeReveal() * this.order.length);
      context.save();
      for (let t = 0; t < nn; t++) {
        const k = this.order[t];
        const r = 2.0 * f[k];
        context.fillStyle = this.nodeColors[this.hemi[k] ? 1 : 0];
        context.globalAlpha = Math.max(0.35, Math.min(1, (f[k] - 0.85) * 3.5));
        context.fillRect(X[k] - r, Y[k] - r, 2 * r, 2 * r);
      }
      context.restore();
    }
    super.draw(context);
  }
}

/** n x n adjacency matrix as dots, both triangles; `reveal` sweeps rows top to bottom. */
export class AdjMatrix extends Node {
  public n = 0;
  public edges: number[][] = [];
  public size = 500;
  public color = '#7FEE64';
  public readonly reveal = createSignal(0);

  public constructor(props: NodeProps) {
    super(props);
  }

  protected draw(context: CanvasRenderingContext2D) {
    if (this.n) {
      const c = this.size / this.n;
      const d = Math.max(c, 1.1);
      const h = this.size / 2;
      const rows = this.reveal() * this.n;
      context.save();
      context.fillStyle = this.color;
      for (const [i, j] of this.edges) {
        if (i < rows) context.fillRect(-h + j * c, -h + i * c, d, d);
        if (j < rows) context.fillRect(-h + i * c, -h + j * c, d, d);
      }
      context.restore();
    }
    super.draw(context);
  }
}
