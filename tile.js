// Pure tiling helpers (no DOM, unit-tested in test/tile.test.mjs).

/** Mirror an index into [0, n) so tiles at image borders see plausible context. */
export function reflect(i, n) {
  if (n <= 1) return 0;
  const period = 2 * (n - 1);
  const m = ((i % period) + period) % period;
  return m < n ? m : period - m;
}

/**
 * Split a w x h image into non-overlapping "core" blocks of size `core`.
 * Each block is later processed with `pad` pixels of extra context on every side.
 */
export function planTiles(w, h, core) {
  const tiles = [];
  const nx = Math.ceil(w / core);
  const ny = Math.ceil(h / core);
  for (let ty = 0; ty < ny; ty++) {
    for (let tx = 0; tx < nx; tx++) {
      const x = tx * core;
      const y = ty * core;
      tiles.push({ x, y, cw: Math.min(core, w - x), ch: Math.min(core, h - y) });
    }
  }
  return tiles;
}

/** Scale needed (and the model factor to use) to reach a target long edge. */
export function planScale(w, h, mode) {
  if (mode === "2" || mode === "4") return { model: Number(mode), final: null };
  const target = mode === "hd" ? 1920 : 3840;
  const long = Math.max(w, h);
  if (long >= target) return { model: 0, final: null, alreadyLarger: true };
  const need = target / long;
  const model = need <= 2 ? 2 : 4;
  const scale = target / long;
  return {
    model,
    final: { w: Math.round(w * scale), h: Math.round(h * scale) },
  };
}
