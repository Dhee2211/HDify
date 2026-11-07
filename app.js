import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs";
import { reflect, planTiles, planScale } from "./tile.js";

// ---------------------------------------------------------------- config
const HF = "https://huggingface.co/SceneWorks/real-esrgan-onnx/resolve/09f741bac80a246b407da3ee902bf5f3291b602f";
const MODELS = {
  2: { url: `${HF}/real_esrgan_x2.onnx`, sha256: "7115ba92e8a1bfa63d68558ef006ef3d91273a068d321b1439f8bb1c9179002c", bytes: 67073434 },
  4: { url: `${HF}/real_esrgan_x4.onnx`, sha256: "5c586662929cbc686c1a5c38d9c060dbdb4ea5863a1f7672b8c0761e6b89c033", bytes: 67051616 },
};
const TILE = 256; // input tile (incl. context). Keeps GPU buffers under default limits.
const PAD = 16; //   context pixels discarded on each side of a tile
const CORE = TILE - 2 * PAD;
const MAX_OUTPUT_PIXELS = 64e6;
const CACHE_NAME = "hdify-models-v1";

// ---------------------------------------------------------------- dom
const $ = (id) => document.getElementById(id);
const els = {
  drop: $("drop"), file: $("file"), workspace: $("workspace"), backend: $("backend"),
  scale: $("scale"), run: $("run"), cancel: $("cancel"), progress: $("progress"),
  bar: $("bar"), status: $("status"), compare: $("compare"), orig: $("orig"),
  result: $("result"), slider: $("slider"), actions: $("actions"), download: $("download"),
  reset: $("reset"), info: $("info"), error: $("error"),
  strength: $("strength"), strengthValue: $("strengthValue"),
};

let source = null; // { canvas, w, h, name }
let sessions = {}; // factor -> InferenceSession
let usingGPU = null;
let cancelled = false;
let busy = false;
let resultCanvas = null; // what gets downloaded
let enhanced = null; //     full-strength model output
let base = null; //         plain high-quality resize (built lazily for blending)
let blended = null; //      mix of base and enhanced

// ---------------------------------------------------------------- helpers
const fmtMB = (b) => (b / 1048576).toFixed(0) + " MB";
const setStatus = (t) => (els.status.textContent = t);
const setBar = (f) => (els.bar.style.width = `${Math.round(f * 100)}%`);
const showError = (msg) => { els.error.textContent = msg; els.error.hidden = !msg; };
// Yield to the event loop so the progress bar can repaint. Uses a timer, not
// requestAnimationFrame, which browsers pause while the tab is in the background.
const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

async function hasWebGPU() {
  try { return !!(navigator.gpu && (await navigator.gpu.requestAdapter())); } catch { return false; }
}

async function sha256Hex(buf) {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

// ---------------------------------------------------------------- model loading
async function fetchModel(factor) {
  const m = MODELS[factor];
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* private mode etc. */ }

  const hit = cache && (await cache.match(m.url));
  if (hit) {
    setStatus("Loading model from cache…");
    return await hit.arrayBuffer();
  }

  setStatus(`Downloading model (${fmtMB(m.bytes)}, one time)…`);
  const res = await fetch(m.url);
  if (!res.ok) throw new Error(`Model download failed (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || m.bytes;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    setBar(got / total);
    setStatus(`Downloading model… ${fmtMB(got)} / ${fmtMB(total)}`);
  }
  const buf = await new Blob(chunks).arrayBuffer();

  setStatus("Verifying model…");
  if ((await sha256Hex(buf)) !== m.sha256) throw new Error("Model checksum mismatch. Download was corrupted.");

  if (cache) {
    try { await cache.put(m.url, new Response(buf)); } catch { /* quota */ }
  }
  return buf;
}

async function getSession(factor) {
  if (sessions[factor]) return sessions[factor];
  const buf = await fetchModel(factor);
  setStatus("Starting engine…");
  setBar(0);
  const gpu = await hasWebGPU();
  const opts = { graphOptimizationLevel: "all" };
  let s;
  if (gpu) {
    try {
      s = await ort.InferenceSession.create(buf, { ...opts, executionProviders: ["webgpu"] });
      usingGPU = true;
    } catch (e) {
      console.warn("WebGPU session failed, falling back to WASM:", e);
    }
  }
  if (!s) {
    s = await ort.InferenceSession.create(buf, { ...opts, executionProviders: ["wasm"] });
    usingGPU = false;
  }
  updateBackendBadge();
  sessions[factor] = s;
  return s;
}

function updateBackendBadge() {
  els.backend.textContent = usingGPU ? "WebGPU" : "CPU (slow)";
  els.backend.dataset.mode = usingGPU ? "gpu" : "cpu";
}

// ---------------------------------------------------------------- upscaling
function readTile(src, w, h, x0, y0) {
  const plane = TILE * TILE;
  const out = new Float32Array(3 * plane);
  for (let j = 0; j < TILE; j++) {
    const sy = reflect(y0 + j, h);
    for (let i = 0; i < TILE; i++) {
      const sx = reflect(x0 + i, w);
      const p = (sy * w + sx) * 4;
      const o = j * TILE + i;
      out[o] = src[p] / 255;
      out[plane + o] = src[p + 1] / 255;
      out[2 * plane + o] = src[p + 2] / 255;
    }
  }
  return out;
}

function writeTile(dst, W, data, s, tile) {
  const S = TILE * s;
  const plane = S * S;
  for (let j = 0; j < tile.ch * s; j++) {
    const row = (tile.y * s + j) * W + tile.x * s;
    const srow = (PAD * s + j) * S + PAD * s;
    for (let i = 0; i < tile.cw * s; i++) {
      const d = (row + i) * 4;
      const o = srow + i;
      dst[d] = data[o] * 255;
      dst[d + 1] = data[plane + o] * 255;
      dst[d + 2] = data[2 * plane + o] * 255;
      dst[d + 3] = 255;
    }
  }
}

async function upscale(factor) {
  const { w, h } = source;
  const s = factor;
  const W = w * s, H = h * s;
  const session = await getSession(factor);
  const srcData = source.canvas.getContext("2d").getImageData(0, 0, w, h).data;
  const dst = new Uint8ClampedArray(W * H * 4);
  const tiles = planTiles(w, h, CORE);
  const inName = session.inputNames[0];
  const outName = session.outputNames[0];
  const t0 = performance.now();

  for (let n = 0; n < tiles.length; n++) {
    if (cancelled) throw new Error("cancelled");
    const t = tiles[n];
    const input = new ort.Tensor("float32", readTile(srcData, w, h, t.x - PAD, t.y - PAD), [1, 3, TILE, TILE]);
    const outputs = await session.run({ [inName]: input });
    const data = await outputs[outName].getData();
    writeTile(dst, W, data, s, t);
    input.dispose?.();
    outputs[outName].dispose?.();

    const done = n + 1;
    const elapsed = (performance.now() - t0) / 1000;
    const eta = Math.max(0, (elapsed / done) * (tiles.length - done));
    setBar(done / tiles.length);
    setStatus(`Enhancing… tile ${done} of ${tiles.length} · ${Math.ceil(eta)}s left`);
    await yieldToUI();
  }

  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.getContext("2d").putImageData(new ImageData(dst, W, H), 0, 0);
  return { canvas, seconds: (performance.now() - t0) / 1000 };
}

// ---------------------------------------------------------------- UI flow
async function loadImage(file) {
  showError("");
  if (!file || !file.type.startsWith("image/")) return showError("Please choose an image file (JPG, PNG, WebP).");
  try {
    const bmp = await createImageBitmap(file, { imageOrientation: "from-image" });
    const canvas = document.createElement("canvas");
    canvas.width = bmp.width;
    canvas.height = bmp.height;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff"; // flatten transparency
    ctx.fillRect(0, 0, bmp.width, bmp.height);
    ctx.drawImage(bmp, 0, 0);
    source = { canvas, w: bmp.width, h: bmp.height, name: file.name.replace(/\.[^.]+$/, "") };
  } catch {
    return showError("Could not read that image. Try a JPG, PNG or WebP file.");
  }
  resultCanvas = null;
  els.drop.hidden = true;
  els.workspace.hidden = false;
  els.actions.hidden = true;
  els.progress.hidden = true;
  showPreview(source.canvas, null);
  els.info.textContent = `${source.w} × ${source.h}`;
  els.run.disabled = false;
}

function showPreview(origCanvas, resCanvas) {
  const target = resCanvas || origCanvas;
  els.compare.style.aspectRatio = `${target.width} / ${target.height}`;
  for (const [el, c] of [[els.orig, origCanvas], [els.result, target]]) {
    el.width = Math.min(c.width, 2400);
    el.height = Math.round((el.width / c.width) * c.height);
    el.getContext("2d").drawImage(c, 0, 0, el.width, el.height);
  }
  els.compare.dataset.hasResult = resCanvas ? "1" : "";
  setSlider(resCanvas ? 50 : 100);
}

function setSlider(v) {
  els.slider.value = v;
  els.compare.style.setProperty("--pos", `${v}%`);
}

async function run() {
  if (!source || busy) return;
  showError("");
  const plan = planScale(source.w, source.h, els.scale.value);
  if (plan.alreadyLarger) return showError("This image is already larger than that target. Pick 2× or 4× instead.");
  if (source.w * source.h * plan.model ** 2 > MAX_OUTPUT_PIXELS) {
    return showError("The result would be too large for the browser. Try 2×, or a smaller photo.");
  }

  busy = true;
  cancelled = false;
  els.run.hidden = true;
  els.cancel.hidden = false;
  els.progress.hidden = false;
  els.actions.hidden = true;
  setBar(0);
  try {
    const { canvas, seconds } = await upscale(plan.model);
    let final = canvas;
    if (plan.final) {
      final = document.createElement("canvas");
      final.width = plan.final.w;
      final.height = plan.final.h;
      const ctx = final.getContext("2d");
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(canvas, 0, 0, final.width, final.height);
    }
    enhanced = final;
    base = null;
    blended = null;
    resultCanvas = final;
    els.strength.value = 100;
    els.strengthValue.textContent = "100%";
    showPreview(source.canvas, final);
    els.info.textContent = `${source.w} × ${source.h}  →  ${final.width} × ${final.height}  ·  ${seconds.toFixed(1)}s`;
    els.progress.hidden = true;
    els.actions.hidden = false;
  } catch (e) {
    els.progress.hidden = true;
    if (e.message !== "cancelled") {
      console.error(e);
      showError(`Something went wrong: ${e.message}`);
    }
  } finally {
    busy = false;
    els.run.hidden = false;
    els.cancel.hidden = true;
  }
}

/** Blend the enhanced image with a plain resize. 100 = full enhancement, 0 = plain resize. */
function applyStrength(percent) {
  if (!enhanced) return;
  els.strengthValue.textContent = `${percent}%`;
  if (percent >= 100) {
    resultCanvas = enhanced;
  } else {
    const { width: w, height: h } = enhanced;
    if (!base) {
      base = document.createElement("canvas");
      base.width = w;
      base.height = h;
      const bctx = base.getContext("2d");
      bctx.imageSmoothingQuality = "high";
      bctx.drawImage(source.canvas, 0, 0, w, h);
    }
    if (!blended) {
      blended = document.createElement("canvas");
      blended.width = w;
      blended.height = h;
    }
    const ctx = blended.getContext("2d");
    ctx.globalAlpha = 1;
    ctx.drawImage(base, 0, 0);
    ctx.globalAlpha = percent / 100;
    ctx.drawImage(enhanced, 0, 0);
    ctx.globalAlpha = 1;
    resultCanvas = blended;
  }
  const el = els.result;
  el.getContext("2d").drawImage(resultCanvas, 0, 0, el.width, el.height);
}

function download() {
  if (!resultCanvas) return;
  resultCanvas.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${source.name}-hd.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
  }, "image/png");
}

function reset() {
  source = null;
  resultCanvas = enhanced = base = blended = null;
  showError("");
  els.workspace.hidden = true;
  els.drop.hidden = false;
  els.file.value = "";
}

// ---------------------------------------------------------------- events
els.file.addEventListener("change", (e) => loadImage(e.target.files[0]));
els.run.addEventListener("click", run);
els.cancel.addEventListener("click", () => { cancelled = true; });
els.download.addEventListener("click", download);
els.reset.addEventListener("click", reset);
els.slider.addEventListener("input", (e) => setSlider(e.target.value));
els.strength.addEventListener("input", (e) => applyStrength(Number(e.target.value)));
for (const t of ["dragenter", "dragover"]) {
  document.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.add("over"); });
}
for (const t of ["dragleave", "drop"]) {
  document.addEventListener(t, (e) => { e.preventDefault(); els.drop.classList.remove("over"); });
}
document.addEventListener("drop", (e) => loadImage(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => {
  const f = [...(e.clipboardData?.files || [])].find((x) => x.type.startsWith("image/"));
  if (f) loadImage(f);
});

hasWebGPU().then((gpu) => {
  usingGPU = gpu;
  updateBackendBadge();
  if (!gpu) showError("WebGPU isn't available in this browser, so enhancing will run on the CPU and be slow. Chrome, Edge or Safari 26 work best.");
});
