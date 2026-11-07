<div align="center">

# HDify

**Turn photos into HD, right in your browser.**

Private, free and open source. Powered by WebGPU. Nothing is uploaded.

**[Try it now: dhee2211.github.io/HDify](https://dhee2211.github.io/HDify/)**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![WebGPU](https://img.shields.io/badge/WebGPU-enabled-4f46e5)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

</div>

## Features

- **Sharper and bigger.** Enhance photos 2× or 4×, or target Full HD or 4K.
- **Private by design.** Your photos are processed on your own device and never leave it.
- **Fast on the GPU.** Uses WebGPU through ONNX Runtime Web, with a slower CPU fallback.
- **Before and after.** Drag the slider to compare.
- **No install, no backend.** A static site with no build step and no dependencies to install.

## How it works

HDify runs the [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) super-resolution model in your browser.
Large photos are split into 256 px tiles (with overlap to hide seams), enhanced one by one, and stitched back together.

The model (about 67 MB) downloads from Hugging Face the first time you enhance a photo. Its SHA-256 checksum is verified and it is cached in your browser, so later runs start instantly.

## Run it

You need a recent browser with WebGPU (Chrome or Edge 113+, or Safari 26+) and Python 3 or any static file server.

```bash
git clone https://github.com/Dhee2211/HDify.git
cd HDify
npm start        # or: python3 -m http.server 8000
```

Open http://localhost:8000 and drop in a photo.

> ES modules don't load from `file://`, so open the app through a local server rather than double-clicking `index.html`.

## Development

```bash
npm test         # unit tests for the tiling logic (Node 18+)
```

| File | Role |
| --- | --- |
| `index.html`, `style.css` | Interface |
| `app.js` | Model loading, WebGPU inference, tile pipeline, UI logic |
| `tile.js` | Pure tiling and scale-planning helpers (unit-tested) |
| `test/` | Tests |

## Limits

- Output is capped at about 64 megapixels to stay within browser memory.
- Transparent images are flattened onto white.
- Without WebGPU it falls back to the CPU, which is much slower.

## Roadmap ideas

- Face restoration with a permissively licensed model
- Batch mode
- Photo-specific model options (denoise strength, anime model)
- JPEG and WebP export
- Installable PWA with offline support

## Contributing

Contributions are welcome.

- [Request a feature](../../issues/new?template=feature_request.yml)
- [Report a bug](../../issues/new?template=bug_report.yml)
- [Read the contributing guide](CONTRIBUTING.md)

## Licenses

HDify is released under the [MIT License](LICENSE). Third-party components and their licenses are listed in [NOTICE.md](NOTICE.md).
