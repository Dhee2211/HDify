import test from "node:test";
import assert from "node:assert/strict";
import {
  FFHQ_TEMPLATE, decodeYuNet, iou, nms, similarity, applyTransform, invertTransform, squareBounds,
} from "../face.js";

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} vs ${b}`);

test("similarity recovers a known rotation, scale and translation", () => {
  const known = { a: 1.2 * Math.cos(0.3), b: 1.2 * Math.sin(0.3), tx: 40, ty: -15 };
  const src = FFHQ_TEMPLATE.map((p) => applyTransform(invertTransform(known), p));
  const est = similarity(src, FFHQ_TEMPLATE);
  close(est.a, known.a); close(est.b, known.b); close(est.tx, known.tx, 1e-4); close(est.ty, known.ty, 1e-4);
});

test("invertTransform round-trips points", () => {
  const t = { a: 0.8, b: -0.35, tx: 12, ty: 90 };
  const p = [123, 456];
  const back = applyTransform(invertTransform(t), applyTransform(t, p));
  close(back[0], p[0], 1e-9); close(back[1], p[1], 1e-9);
});

test("iou and nms keep the best of overlapping boxes", () => {
  const a = { x: 0, y: 0, w: 100, h: 100, score: 0.9 };
  const b = { x: 5, y: 5, w: 100, h: 100, score: 0.8 };
  const c = { x: 300, y: 300, w: 50, h: 50, score: 0.7 };
  close(iou(a, a), 1);
  assert.equal(iou(a, c), 0);
  const kept = nms([b, c, a]);
  assert.deepEqual(kept.map((f) => f.score), [0.9, 0.7]);
});

test("decodeYuNet decodes one confident cell", () => {
  const size = 640;
  const outs = {};
  for (const s of [8, 16, 32]) {
    const n = (size / s) ** 2;
    outs[`cls_${s}`] = new Float32Array(n);
    outs[`obj_${s}`] = new Float32Array(n);
    outs[`bbox_${s}`] = new Float32Array(n * 4);
    outs[`kps_${s}`] = new Float32Array(n * 10);
  }
  // stride 16, row 10, col 20
  const cols = size / 16, i = 10 * cols + 20;
  outs.cls_16[i] = 0.9; outs.obj_16[i] = 0.9;
  outs.bbox_16.set([0.5, 0.5, Math.log(4), Math.log(5)], i * 4);
  const faces = decodeYuNet((name) => outs[name]);
  assert.equal(faces.length, 1);
  const f = faces[0];
  close(f.score, 0.9, 1e-6);
  close(f.w, 64, 1e-3); close(f.h, 80, 1e-3);
  close(f.x + f.w / 2, (20 + 0.5) * 16); close(f.y + f.h / 2, (10 + 0.5) * 16);
  assert.equal(f.landmarks.length, 5);
});

test("squareBounds clamps to the image", () => {
  const t = { a: 1, b: 0, tx: -100, ty: -100 };
  assert.deepEqual(squareBounds(t, 300, 300), { x: 0, y: 0, w: 300, h: 300 });
});
