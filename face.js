// Pure face helpers (no DOM, unit-tested in test/face.test.mjs).

export const DETECT_SIZE = 640; // YuNet runs on a fixed 640 x 640 input
export const FACE_SIZE = 512; //   RestoreFormer++ works on 512 x 512 aligned faces

/** FFHQ 512 five-point template: eye, eye, nose, mouth corner, mouth corner (image left first). */
export const FFHQ_TEMPLATE = [
  [192.98138, 239.94708],
  [318.90277, 240.1936],
  [256.63416, 314.01935],
  [201.26117, 371.41043],
  [313.08905, 371.15118],
];

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/**
 * Decode YuNet's 12 outputs (cls, obj, bbox, kps for strides 8/16/32) into faces.
 * `get(name)` returns the flat Float32Array for an output. Coordinates are in the 640 x 640 input space.
 */
export function decodeYuNet(get, size = DETECT_SIZE, minScore = 0.6) {
  const faces = [];
  for (const stride of [8, 16, 32]) {
    const cls = get(`cls_${stride}`);
    const obj = get(`obj_${stride}`);
    const bbox = get(`bbox_${stride}`);
    const kps = get(`kps_${stride}`);
    const cols = size / stride;
    const rows = size / stride;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        const score = Math.sqrt(clamp01(cls[i]) * clamp01(obj[i]));
        if (score < minScore) continue;
        const cx = (c + bbox[i * 4]) * stride;
        const cy = (r + bbox[i * 4 + 1]) * stride;
        const w = Math.exp(bbox[i * 4 + 2]) * stride;
        const h = Math.exp(bbox[i * 4 + 3]) * stride;
        const landmarks = [];
        for (let n = 0; n < 5; n++) {
          landmarks.push([(kps[i * 10 + 2 * n] + c) * stride, (kps[i * 10 + 2 * n + 1] + r) * stride]);
        }
        faces.push({ x: cx - w / 2, y: cy - h / 2, w, h, score, landmarks });
      }
    }
  }
  return faces;
}

export function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Non-maximum suppression: keep the best-scoring face of each overlapping group. */
export function nms(faces, threshold = 0.3, limit = 10) {
  const sorted = [...faces].sort((a, b) => b.score - a.score);
  const kept = [];
  for (const f of sorted) {
    if (kept.every((k) => iou(k, f) < threshold)) kept.push(f);
    if (kept.length >= limit) break;
  }
  return kept;
}

/** Scale a face from detector space back to image space. */
export function scaleFace(f, k) {
  return {
    x: f.x * k, y: f.y * k, w: f.w * k, h: f.h * k, score: f.score,
    landmarks: f.landmarks.map(([x, y]) => [x * k, y * k]),
  };
}

/**
 * Least-squares similarity transform (rotation + uniform scale + translation) from `src` to `dst` points.
 * Maps (x, y) to (a*x - b*y + tx, b*x + a*y + ty).
 */
export function similarity(src, dst) {
  const n = src.length;
  let mx = 0, my = 0, mu = 0, mv = 0;
  for (let i = 0; i < n; i++) {
    mx += src[i][0]; my += src[i][1]; mu += dst[i][0]; mv += dst[i][1];
  }
  mx /= n; my /= n; mu /= n; mv /= n;
  let na = 0, nb = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - mx, y = src[i][1] - my;
    const u = dst[i][0] - mu, v = dst[i][1] - mv;
    na += x * u + y * v;
    nb += x * v - y * u;
    den += x * x + y * y;
  }
  const a = na / den;
  const b = nb / den;
  return { a, b, tx: mu - (a * mx - b * my), ty: mv - (b * mx + a * my) };
}

export function applyTransform(t, [x, y]) {
  return [t.a * x - t.b * y + t.tx, t.b * x + t.a * y + t.ty];
}

export function invertTransform(t) {
  const den = t.a * t.a + t.b * t.b;
  const a = t.a / den;
  const b = -t.b / den;
  return { a, b, tx: -(a * t.tx - b * t.ty), ty: -(b * t.tx + a * t.ty) };
}

/** Bounding box of the 512 x 512 face square after a transform, clamped to a w x h image. */
export function squareBounds(t, w, h, size = FACE_SIZE) {
  const pts = [[0, 0], [size, 0], [0, size], [size, size]].map((p) => applyTransform(t, p));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs)));
  const y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const x1 = Math.min(w, Math.ceil(Math.max(...xs)));
  const y1 = Math.min(h, Math.ceil(Math.max(...ys)));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}
