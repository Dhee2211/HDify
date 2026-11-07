# Contributing to HDify

Thanks for helping! HDify is intentionally small: a static site, no build step, no backend.

## Ground rules

1. **Privacy first.** Photos must never leave the user's device. No uploads, no analytics.
2. **Licenses matter.** Only add models and libraries with permissive licenses (MIT, BSD, Apache-2.0). Do not add non-commercial models, including CodeFormer. Add new entries to `NOTICE.md`.
3. **Stay light.** Avoid dependencies and build tooling unless there is a strong reason.
4. **Verify model downloads.** Every model must be pinned to a specific revision and checked with a SHA-256 hash.

## Setup

```bash
git clone https://github.com/Dhee2211/HDify.git
cd HDify
npm start     # serves http://localhost:8000
npm test      # runs the unit tests
```

## Pull requests

- Open an issue first for anything larger than a small fix.
- Keep PRs focused, and describe how you tested (browser, GPU, image size).
- Put logic that doesn't need the DOM in `tile.js` and cover it with a test.

## Security

Report anything that could expose user photos privately through GitHub's *Security → Report a vulnerability*.
