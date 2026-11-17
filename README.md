# img2three library

Procedural Three.js models rebuilt from photos by `img2three` runs, collected in one place, with a
gallery page.

```
models/<slug>/
  createObjectModel.ts   the factory: create<Name>Model(options) -> THREE.Group (+ helper .ts files)
  assets/                images the factory loads (URLs are relative to the library root)
  spec.json              the ObjectSculptSpec the code was generated from (source of truth)
  RESULT.md              the agent's report: what matches, what does not, every override
  reference.*            the photo; preview-comparison.png: photo vs render
  model.json             source run, export name, fidelity, turns, API-equivalent cost, library patches
models/index.ts          generated registry of every model (namespaced imports)
gallery/                 the gallery page source; index.html is the page
tools/publish.py         copy a finished run in
```

## Publishing

From WSL, automatically after a run:

```
img2three C:\bilder\lampa.jpg -n "Desk lamp" --into C:\repos\img2three-library
img2three publish C:\repos\img2three-runs\<run> --into C:\repos\img2three-library   # an earlier run
```

or by hand on Windows: `python tools/publish.py <run-dir> [--slug s] [--title t]`, then `npm run bundle`.

Rules: only finished runs (`state.json` status `complete` and a `RESULT.md`) are accepted. A different
run of the same subject never overwrites an existing model; it gets `<slug>-2`, `-3`, and so on.
`--force` only re-publishes the same source run.

## Gallery

`npm run bundle`, then serve the folder (textures load over http):
`python -m http.server 8770 --bind 127.0.0.1` and open http://127.0.0.1:8770/.
Pick a model, drag to orbit, scroll to zoom, use the explode slider; the photo is shown beside it.

## Using a model in your own app

Copy `models/<slug>/` (the .ts files and `assets/`), keep the asset URLs reachable from your page (or
rewrite the `models/<slug>/assets/...` strings), and call:

```ts
import { createClawHammerModel } from './models/claw-hammer/createObjectModel';
scene.add(createClawHammerModel({ castShadow: true }));
```

`npm run typecheck` runs strict `tsc` over the gallery and every published model.

## Reference image credits

| Model | Photo | License |
|---|---|---|
| `enamel-mug` | "Coffee Mug (Unsplash).jpg", Wikimedia Commons | CC0 |
| `claw-hammer` | "Claw-hammer.jpg", Wikimedia Commons | Public domain |

The models were rebuilt as procedural code with the
[img2threejs](https://github.com/img2threejs/img2threejs) skill (Apache-2.0).
