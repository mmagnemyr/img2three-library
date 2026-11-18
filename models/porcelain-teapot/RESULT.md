# RESULT — Porcelain teapot (img2threejs, generic profile)

**Status:** complete. `forge/next.py` reports `status=complete`, with 0 of the 3-per-pass / 6-total correction
budget used. All 8 build passes (blockout → structural → form-refinement → material → surface → lighting →
interaction → optimization) were reviewed with `continue`.
**Final fidelity estimate: 0.78–0.80** on the skill's scale, where 0.75 = "object reads correctly, local details
approximate". The decoration and the reference-view match are strong. Form details and the hidden sides are approximate.

## Artifacts
- Factory: `src/createObjectModel.ts` (`createPorcelainTeapotModel`). It is generated from `object-sculpt-spec.json`
  and was not hand-patched. `npm run typecheck` is clean.
- Final renders: `renders/final/reference.png`, `renders/final/orbit-000.png`, `orbit-090.png`, `orbit-180.png`,
  `orbit-270.png` (plus `*-z2.2.png` close framings, `orbits-sheet-z2.2.png`, `comparison-reference.png`,
  `zoom-pairs.png` (2× crops: reference | render), `exploded.png`).
- Textures loaded at runtime from `textures/` (path `../textures/...`, relative to `viewer/index.html`). If the factory is
  used elsewhere, copy the folder or change the URLs in the spec.
- Per-pass evidence: `renders/<pass>/` (tier1, multi-angle, turntable, self-intersection, attachment, interior-difference,
  comparison sheets, zoom pairs). Intake and material evidence: `evidence/`, `material-evidence/`, `material-analysis.json`.
- Reproduce: `scripts/fit_shape.py` / `fit_paths.py` (silhouette fit) → `scripts/bake_projection.py` + `make_maps.py`
  + `make_surface_maps.py` → `scripts/rebuild_spec.sh` → `generate_threejs_factory.py` → `scripts/run_pass_gates.sh <pass>`.

## How it was built
- **Camera** fitted by forward silhouette projection, using a pure-Pillow rasterizer with the viewer's camera maths
  (`scripts/fitlib.py`): fov 15°, distance 10.3, pitch 15°, yaw 0. Pitch was fixed from the perspective spread between the rim
  ellipse (~12°) and the foot ellipse (~19°). A free search drove it to 27° by compensating for profile errors, so it was not used.
- **Geometry:** lathe body (140-point profile), lathe lid, knob and gold cap; `tapered-sweep` spout (40 stations,
  r 0.22→0.13, open tip); `tube` handle (r 0.092) in the spout–handle plane. Handle yaw 0 was checked per lesson 10: both
  roots sit on the body silhouette edge. Bend radius ≥ tube radius was enforced, giving 0 self-intersections.
- **Decoration = the reference's own pixels** (projection route). The photo is baked into the LatheGeometry UV space of
  body and lid through the fitted camera, gated by facing, the photo mask, and depth-aware spout/handle occlusion.
  Shading is normalised against the local glaze white (clamp 0.9–1.15, lesson 16). The spout flower spray and the handle
  stain are `decal` components made from reference crops. The handle gold band is a decal projected from above.
- **Materials:** glaze is dielectric with roughness 0.10, clearcoat 0.6, and is textureless by evidence. Painted body and
  lid use the projected albedo plus enamel relief (height/normal from a paint mask, bump 0.0015) and roughness of
  0.14 for glaze vs 0.22 for enamel. Gold is metal, #9A7236, roughness 0.30. Material pipeline wired to `ceramic.glazed`
  and `metal.gold`; `material_gate` passes on band, glaze, red ground and gold crops.
- **Lighting** (lightingFromPhoto):
  - key light [-3,8,7] at 1.7 with a soft PCF shadow; fill 1.3; lavender bounce from below 0.6; back-right rim 0.5
  - hemisphere 1.1 with the cloth as ground colour; environment 0.45; ACES exposure 0.6, calibrated on glaze white
- **Action-ready:** semantic pivots, 7 sockets verified in world space with 0 error, colliders, destruction groups
  (vessel/lid/spout/handle), detachable lid; `pick()` and `explode()` verified in the browser.

## What matches (numbers from the final build)
- Silhouette: Tier-1 IoU **0.941** against the white-isolated reference, aspect delta 0.005, scale delta 0.005. The fit
  model's IoU is 0.958.
- Orbits: multi-angle area ratios 0.82–1.00, not degenerate. Turntable is clean: the only enclosed holes are the handle
  opening at 0° and 180°.
- Interior difference (inside the silhouette, by height band): lid/knob 0.187, shoulder band 0.135, panel 0.119,
  foot 0.081. Blockout was 0.29 / 0.26 / 0.23 / 0.18.
- Region means, reference → render: glaze white 203–213 → 214; red ground 163/62/46 → 165/48/40; right flank
  60/55/48 → 59/54/48; spout 179/190/203 → 187/192/201.
- Identity features are all present and placed: harlequin shoulder band, foot band, figural panel, chrysanthemum
  ground, lid chrysanthemum, gold knob cap, collar gold line, gallery gold line, spout lip ring, handle gold band,
  bamboo nodes, spout flower spray, handle stain.
- Budget: 72,615 triangles, 14 meshes, 26 draw calls (target 120k / 160).

## What does not match
- **Hidden sides are speculative.** The rear and the 90°/270° sides carry a rotated or quarter-turn copy of the visible
  panel, blended by visibility. They show repeated figures, some smearing, and the photo's dark right-side shading.
  One view cannot show the real rear decoration.
- **Lid and collar front:** the photo's lid front edge projects about 0.05 units lower than the model's. As a result, the
  lowest lid leaves are cut off and the white lip reads slightly taller than in the reference. This is the
  pitch/profile trade-off of a single-view fit.
- **Spout tip** is a round open cut with a gold ring, not the reference's slanted oval lip. A flared variant was tried
  and reverted because it read as a trumpet. The spout's root end is also open, visible only in the exploded view.
- **Bamboo nodes** are subtle (0.008 proud). The reference shows more distinct rings.
- The lower painted band is still darker than the photo (88 vs 106), because the cloth bounce is only partly reproduced.
- The stain decal reads slightly grey compared with the reference's warm brown.

## Manual overrides and deviations, with reasons
1. **`analyze_texture.py` classes rejected** (lessons 4 and 14; read-only, no `--in-place`). It called the glaze metal
   (1.0) and the gold dielectric (0.0). Classes come from my reading of the reference; see `material-evidence/README.md`.
2. **`apply_material_analysis.py` values partly restored** (`scripts/apply_materials.py`). It replaced the
   projected-photo `referencePbr` with a crop tile, reset roughness to registry priors (0.26 / 0.22), and added maps to
   textureless materials. The wiring metadata is kept; the measured values are restored.
3. **`delight_albedo.py` not used as albedo** (lesson 16). It was run for evidence only. The bake normalises against
   glaze white instead. The first clamp (0.85–1.45) lifted the red ground 1.45–1.95×; it is now 0.9–1.15.
4. **Isolation colour.** Magenta is read as foreground by the segmenter (saturated). #808080 merged with the darker
   calibrated shading in the material pass. Tier 1 therefore uses white isolation (`evidence/ref_iso_ffffff.png`).
5. **Turntable `--allow-holes`.** The only holes are the handle opening; the strict run is kept as `turntable-strict.json`.
6. **Tier-1 per-part colour ΔE is a whole-render k-means check.** Parts under 1% of the frame (gold trim, decals) get
   no cluster of their own, so the result moved with lighting changes. Gold and flower recipes use
   `extract_part_color_recipe.py` dominants, the tool of record, with class kept as metal. Final max ΔE is 18.9.
7. **Attachment anchors** are declared only where the anchor is not the root (knob→lid, lip→spout, nodes→handle).
   Lid, spout and handle are welded to the vessel under the embedding contract. The spout `maxOffset` is 1.3, matching
   its measured root-to-tip chord of 1.21.
8. **Exposure and lights were calibrated during the material pass** by sweeps against measured reference regions. Without
   that, the material comparisons could not separate albedo from lighting.

## Process issues (reported honestly)
- **Blockout `pipeline-sync` is archived as `pending`** in `state.json`. I ran `orchestrate_passes.py sync` before marking
  `ai-review-recorded`, which advanced the pass. The sync did run, but the step cannot be re-marked honestly (lesson 21).
  Every later pass used review → mark → sync → mark and archived clean.
- **A phantom InstancedMesh** (24 default boxes, 1,728 triangles, hidden inside the body) was emitted from blockout until
  the optimization pass. It came from the map-only `harlequin-diamonds` repetition system. It was removed by naming
  `body` as the realising component. It was never visible in any render. Generator observation: `realization:
  "map-only"` / `buildsGeometry: false` does not stop instancing.
- `check_part_coverage.py` reports `part-not-specified: body` as a note: 13 specified, 14 built, 0 errors. The checker
  treats the root component as the model.
- FPS was not measured (headless SwiftShader is not representative).
