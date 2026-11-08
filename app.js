import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs";
import { reflect, planTiles, planScale } from "./tile.js";
import {
  DETECT_SIZE, FACE_SIZE, FFHQ_TEMPLATE, decodeYuNet, nms, scaleFace,
  similarity, invertTransform, squareBounds,
} from "./face.js";

// ---------------------------------------------------------------- config
const HF = "https://huggingface.co/SceneWorks/real-esrgan-onnx/resolve/09f741bac80a246b407da3ee902bf5f3291b602f";
const MODELS = {
  2: { url: `${HF}/real_esrgan_x2.onnx`, sha256: "7115ba92e8a1bfa63d68558ef006ef3d91273a068d321b1439f8bb1c9179002c", bytes: 67073434 },
  4: { url: `${HF}/real_esrgan_x4.onnx`, sha256: "5c586662929cbc686c1a5c38d9c060dbdb4ea5863a1f7672b8c0761e6b89c033", bytes: 67051616 },
};
const FACE_MODELS = {
  detect: {
    name: "face detector",
    url: "https://huggingface.co/opencv/face_detection_yunet/resolve/3cc26e7f1014a5ee5d74a42acee58bafc9d0a310/face_detection_yunet_2023mar.onnx",
    sha256: "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    bytes: 232589,
  },
  restore: {
    name: "face restoration model",
    url: "https://huggingface.co/Saimon8420/restoreformer-pp-web/resolve/9e5912e04026135bc1a7c8557a2da66f521a7b8e/restoreformer_pp_int8w.onnx",
    sha256: "4b3983dba15b8dd26db1bc94be57558ca4d783424ca6f3715676ab53450bcb6c",
    bytes: 74375477,
  },
};
const MAX_FACES = 10;
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
  strength: $("strength"), strengthValue: $("strengthValue"), faces: $("faces"),
};

let source = null; // { canvas, w, h, name }
let sessions = {}; // model key -> InferenceSession
let usingGPU = null;
let cancelled = false;
let busy = false;
let resultCanvas = null; // what gets downloaded
let enhanced = null; //     full-strength model output
let base = null; //         plain high-quality resize (built lazily for blending)
let blended = null; //      mix of base and enhanced

// ---------------------------------------------------------------- helpers
const fmtMB = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0) + " MB";
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
async function fetchModel(m) {
  const label = m.name || "model";
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* private mode etc. */ }

  const hit = cache && (await cache.match(m.url));
  if (hit) {
    setStatus(`Loading ${label} from cache…`);
    return await hit.arrayBuffer();
  }

  setStatus(`Downloading ${label} (${fmtMB(m.bytes)}, one time)…`);
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
    setStatus(`Downloading ${label}… ${fmtMB(got)} / ${fmtMB(total)}`);
  }
  const buf = await new Blob(chunks).arrayBuffer();

  setStatus(`Verifying ${label}…`);
  if ((await sha256Hex(buf)) !== m.sha256) throw new Error(`The ${label} download was corrupted (checksum mismatch). Please try again.`);

  if (cache) {
    try { await cache.put(m.url, new Response(buf)); } catch { /* quota */ }
  }
  return buf;
}

async function getSession(key, model = MODELS[key]) {
  if (sessions[key]) return sessions[key];
  const buf = await fetchModel(model);
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
  sessions[key] = s;
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

// ---------------------------------------------------------------- faces
/** Find faces in the source photo with YuNet. Returns faces in source-image pixels. */
async function detectFaces() {
  const session = await getSession("detect", FACE_MODELS.detect);
  const k = DETECT_SIZE / Math.max(source.w, source.h);
  const c = document.createElement("canvas");
  c.width = c.height = DETECT_SIZE;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, DETECT_SIZE, DETECT_SIZE);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source.canvas, 0, 0, Math.round(source.w * k), Math.round(source.h * k));
  const px = ctx.getImageData(0, 0, DETECT_SIZE, DETECT_SIZE).data;
  const plane = DETECT_SIZE * DETECT_SIZE;
  const input = new Float32Array(3 * plane); // BGR, 0..255
  for (let i = 0; i < plane; i++) {
    input[i] = px[i * 4 + 2];
    input[plane + i] = px[i * 4 + 1];
    input[2 * plane + i] = px[i * 4];
  }
  const tensor = new ort.Tensor("float32", input, [1, 3, DETECT_SIZE, DETECT_SIZE]);
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const data = {};
  for (const name of session.outputNames) data[name] = await outputs[name].getData();
  const found = nms(decodeYuNet((name) => data[name]), 0.3, MAX_FACES);
  return found.map((f) => scaleFace(f, 1 / k));
}

/** Warp the face into the 512 x 512 template, restore it, and return a 512 canvas. */
async function restoreOne(session, canvas, t) {
  const crop = document.createElement("canvas");
  crop.width = crop.height = FACE_SIZE;
  const cctx = crop.getContext("2d", { willReadFrequently: true });
  cctx.imageSmoothingQuality = "high";
  cctx.setTransform(t.a, t.b, -t.b, t.a, t.tx, t.ty);
  cctx.drawImage(canvas, 0, 0);
  const px = cctx.getImageData(0, 0, FACE_SIZE, FACE_SIZE).data;
  const plane = FACE_SIZE * FACE_SIZE;
  const input = new Float32Array(3 * plane); // RGB, -1..1
  for (let i = 0; i < plane; i++) {
    input[i] = px[i * 4] / 127.5 - 1;
    input[plane + i] = px[i * 4 + 1] / 127.5 - 1;
    input[2 * plane + i] = px[i * 4 + 2] / 127.5 - 1;
  }
  const tensor = new ort.Tensor("float32", input, [1, 3, FACE_SIZE, FACE_SIZE]);
  const outName = session.outputNames[0];
  const outputs = await session.run({ [session.inputNames[0]]: tensor }, [outName]);
  const out = await outputs[outName].getData();
  const img = new ImageData(FACE_SIZE, FACE_SIZE);
  for (let i = 0; i < plane; i++) {
    img.data[i * 4] = (out[i] + 1) * 127.5;
    img.data[i * 4 + 1] = (out[plane + i] + 1) * 127.5;
    img.data[i * 4 + 2] = (out[2 * plane + i] + 1) * 127.5;
    img.data[i * 4 + 3] = 255;
  }
  tensor.dispose?.();
  outputs[outName].dispose?.();
  const res = document.createElement("canvas");
  res.width = res.height = FACE_SIZE;
  res.getContext("2d").putImageData(img, 0, 0);
  return res;
}

let faceMask = null;
function getFaceMask() {
  if (faceMask) return faceMask;
  faceMask = document.createElement("canvas");
  faceMask.width = faceMask.height = FACE_SIZE;
  const ctx = faceMask.getContext("2d");
  ctx.translate(FACE_SIZE / 2, 268);
  ctx.scale(1, 1.18);
  const g = ctx.createRadialGradient(0, 0, 0, 0, 0, 210);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.62, "rgba(255,255,255,1)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(-FACE_SIZE, -FACE_SIZE, FACE_SIZE * 2, FACE_SIZE * 2);
  return faceMask;
}

/** Blend a restored 512 face back into the canvas with a soft elliptical mask. */
function pasteFace(canvas, restored, t) {
  const inv = invertTransform(t);
  const box = squareBounds(inv, canvas.width, canvas.height);
  if (!box.w || !box.h) return;
  const layer = document.createElement("canvas");
  layer.width = box.w;
  layer.height = box.h;
  const ctx = layer.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.setTransform(inv.a, inv.b, -inv.b, inv.a, inv.tx - box.x, inv.ty - box.y);
  ctx.drawImage(restored, 0, 0);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(getFaceMask(), 0, 0);
  canvas.getContext("2d").drawImage(layer, box.x, box.y);
}

/** Detect faces and restore them in place on the upscaled canvas. Returns how many were restored. */
async function restoreFaces(canvas, scale) {
  setStatus("Looking for faces…");
  setBar(0);
  const found = await detectFaces();
  if (!found.length) return 0;
  if (cancelled) throw new Error("cancelled");
  const session = await getSession("restore", FACE_MODELS.restore);
  for (let n = 0; n < found.length; n++) {
    if (cancelled) throw new Error("cancelled");
    setBar(n / found.length);
    setStatus(`Restoring face ${n + 1} of ${found.length}…`);
    await yieldToUI();
    const lm = found[n].landmarks.map(([x, y]) => [x * scale, y * scale]);
    const t = similarity(lm, FFHQ_TEMPLATE);
    const restored = await restoreOne(session, canvas, t);
    pasteFace(canvas, restored, t);
  }
  return found.length;
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
    const started = performance.now();
    const { canvas } = await upscale(plan.model);
    let faceCount = 0;
    if (els.faces.checked) {
      try {
        faceCount = await restoreFaces(canvas, plan.model);
      } catch (e) {
        if (e.message === "cancelled") throw e;
        console.error(e);
        showError(`Face restoration was skipped: ${e.message}`);
      }
    }
    const seconds = (performance.now() - started) / 1000;
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
    const faceNote = els.faces.checked ? (faceCount ? `  ·  ${faceCount} face${faceCount > 1 ? "s" : ""} restored` : "  ·  no faces found") : "";
    els.info.textContent = `${source.w} × ${source.h}  →  ${final.width} × ${final.height}  ·  ${seconds.toFixed(1)}s${faceNote}`;
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

// Examples gallery: compare sliders and "Try this photo"
for (const fig of document.querySelectorAll(".ex")) {
  fig.querySelector("input").addEventListener("input", (e) => fig.style.setProperty("--pos", `${e.target.value}%`));
}
for (const btn of document.querySelectorAll("[data-example]")) {
  btn.addEventListener("click", async () => {
    if (busy) return;
    const name = btn.dataset.example;
    try {
      const blob = await (await fetch(`examples/${name}-input.jpg`)).blob();
      reset();
      await loadImage(new File([blob], `${name}.jpg`, { type: "image/jpeg" }));
      els.faces.checked = btn.dataset.faces === "1";
      els.scale.value = "4";
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch {
      showError("Could not load the example photo.");
    }
  });
}

hasWebGPU().then((gpu) => {
  usingGPU = gpu;
  updateBackendBadge();
  if (!gpu) showError("WebGPU isn't available in this browser, so enhancing will run on the CPU and be slow. Chrome, Edge or Safari 26 work best.");
});
