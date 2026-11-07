import test from "node:test";
import assert from "node:assert/strict";
import { reflect, planTiles, planScale } from "../tile.js";

test("reflect mirrors out-of-range indices", () => {
  assert.equal(reflect(-1, 10), 1);
  assert.equal(reflect(-3, 10), 3);
  assert.equal(reflect(10, 10), 8);
  assert.equal(reflect(12, 10), 6);
  assert.equal(reflect(5, 10), 5);
  assert.equal(reflect(-5, 1), 0);
});

test("reflect stays in range for tiny images and large offsets", () => {
  for (let n = 1; n < 8; n++) {
    for (let i = -50; i < 50; i++) {
      const r = reflect(i, n);
      assert.ok(r >= 0 && r < n, `reflect(${i}, ${n}) = ${r}`);
    }
  }
});

test("planTiles covers every pixel exactly once", () => {
  const w = 500, h = 301, core = 224;
  const seen = new Uint8Array(w * h);
  for (const t of planTiles(w, h, core)) {
    for (let y = t.y; y < t.y + t.ch; y++)
      for (let x = t.x; x < t.x + t.cw; x++) seen[y * w + x]++;
  }
  assert.ok(seen.every((v) => v === 1));
});

test("planScale picks the smallest model factor that reaches the target", () => {
  assert.deepEqual(planScale(100, 50, "2"), { model: 2, final: null });
  assert.equal(planScale(1200, 800, "hd").model, 2);
  assert.deepEqual(planScale(1200, 800, "hd").final, { w: 1920, h: 1280 });
  assert.equal(planScale(400, 300, "hd").model, 4);
  assert.equal(planScale(3000, 2000, "hd").alreadyLarger, true);
});
