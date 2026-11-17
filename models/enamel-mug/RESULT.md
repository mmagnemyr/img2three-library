# RESULT — Enamel mug (reference/reference.jpg)

**Status: complete.** `forge/next.py` → `status=complete`; all 8 build passes reviewed `continue`
(blockout → structural → form → material → surface → lighting → interaction → optimization).
Correction budget used: **0 / 6** (every fix was deterministic tuning before that pass's review was recorded).

**Final fidelity estimate: 0.80** ("strong procedural match for real-time use"). Last AI-vision score 0.87.
This is one photo, so the back of the mug, its interior and the far half of the sunburst mark are inferred.

## Final renders — `renders/final/`
| File | What |
|---|---|
| `reference.png` | fitted reference camera (yaw 0, pitch 1°, fov 2.707°, dist 25, target [0.157, 0.415, 0]) |
| `orbit-000.png`, `orbit-090.png`, `orbit-180.png`, `orbit-270.png` | 4 orbits at 12° elevation (`orbits-sheet.png` = 2×2) |
| `*-iso.png` | same views on #808080 with no table (used by the silhouette gates) |
| `comparison.png` | reference crop (left) vs render (right) |
| `explode.png`, `interactive.png` | explode layout (amount 1) and the interactive orbit viewer |
| `stats.json`, `tones.json` | triangles/draw calls; matched-region tone probe |

The code is in `src/createObjectModel.ts` (generated, then patched), `src/decalTexture.ts` (baked print) and
`src/mugInteraction.ts` (explode and pick). Rebuild with `npm run bundle`.

## What matches (measured against the isolated reference, 840×600 view)
- **Silhouette and proportions:** Tier-1 IoU **0.910**, aspect Δ 0.000, scale Δ 0.000.
  - Height is 437 vs 436 px, body width 431 vs 432 px, and the handle's outer edge is at 734 vs 734 px.
  - The handle's lower joint is at row 467 vs 467.
  - Model: body Ø 0.85, H 0.845 (1 unit ≈ 10 cm), H/D 0.99.
- **Black rolled rim bead** (torus R 0.448, r 0.0165): it overhangs the wall by 4.7 % of D per side.
  - The dark band covers photo rows 245–255 vs 245–254.
- **D-ring strap handle:** a closed D whose straight side lies on the wall.
  - The handle plane is yawed **26° toward the camera** (azimuth 64°). Evidence: the wall strip sits inside the body silhouette at 958–976 px with no bulge past the limb (`analysis/handle4x.png`).
  - It is built from the traced centreline, de-projected by 1/sin 64°.
- **Skull decal with "DECAF! / PANDCO" lettering:** the reference's own pixels, projected onto a 0.0012-proud shell.
  - The ink core reads 41.6 vs 45.6 luma.
  - Material comparator ΔE00 is 3.2.
- **Sunburst mark** on the left limb: chevron rows 375–396 vs 374–395; vertical strokes 339–350 / 418–429 vs 336–352 / 419–431.
- **Enamel finish and lighting:**
  - White glaze: ΔE00 3.1; lit wall 223.6 vs 228.4 luma; shaded right limb 167.6 vs 168.6 (cool tint reproduced).
  - Shaded band under the bead: present.
  - Black bead: ΔE00 6.0.
- **Interior difference** (inside the silhouette): whole **0.055**. By band: rim 0.138, upper wall 0.050, decal 0.054, base 0.041. At blockout these were 0.185 whole, 0.449 rim and 0.250 decal.
- **Structure gates:**
  - Self-intersection: 0 vertices inside (5727 sampled).
  - Turntable: 4/4 azimuths. Holes are allowed because the handle loop is a real through-hole.
  - Multi-angle: no collapsed views.
  - Attachment anchors: pass.
  - Part coverage: 0 errors, 0 warnings.
  - All 6 parts can be clicked and exploded; decal and chip clicks resolve to the body. Sockets, colliders and destruction groups are present.
- **Budget:** 10,864 triangles (budget 60k).
  - The model's main pass has 8 draws (budget 12).
  - The viewer frame totals 13 draws: 8 main + 4 shadow-map + the viewer's table.

## What does not match (with numbers)
- **Rim highlight:** the reference bead has a narrow, bright crown streak on its right half; the render's highlight is even and dimmer. The rim-region mean is 48.9 vs 73.1 luma.
- **Rim "frown":** in the reference, the rim's top edge drops 4–7 px toward both ends, and this is not modelled. A camera looking down 2.3° would explain it, but it would make the dark band 17 px thick against 10 px (measured, `renders/lighting-pass`). It is most likely lens distortion or an out-of-round rim.
- **Photo background:** the bokeh wood wall and the table reflection are not reproduced. The viewer uses a flat background and table plane, which is outside the model's scope.
- **Base chip:** it is a simplified dark sliver; the reference's is an irregular streak with a hook.
- **Handle top-right arc:** a few pixels tighter than the reference's broad curve.
- **Inferred, not observed:** the far half of the sunburst (redrawn from its visible half, confidence 0.4–0.5), the back of the mug (plain enamel), the interior (white enamel), and the strap's section (round, Ø 0.049).

## Manual overrides and patches (and why)
1. **`tools/patch_factory.py`** is a re-runnable patch applied after every generator run:
   - **DEF-5:** the tube with an attachment was emitted as a straight cylinder. The patch restores the authored `tubePath` and offsets the mesh by −endpoint.start so the pivot stays at the socket (CLAUDE.md lesson 3).
   - **Decal shell:** a closed attachment cylinder is replaced by an open segment (θ −120..60°, y 0.08–0.76). It carries the projected ink and is flagged `explodeWithParent` (lesson 6).
   - **`applyCavityOcclusion`:** a vertex-colour AO bake for the cup interior (0.5 at the lip down to 0.22 deep) and the outer lip flare under the bead (0.5). The environment and hemisphere fill have no occlusion, so they lit the interior white and erased the under-bead shading.
   - **`castShadow = false`** for the embedded weld pads and the chip.
   - **`attachInteraction(root)`** wires in explode and picking from `src/mugInteraction.ts`.
2. **`analyze_texture.py` misclassified finishes** (lesson 4), so I did not use `--in-place` and overrode the values with reasons (`material-evidence/material-evidence.json`):
   - White enamel came back as brushed-steel with metalness 1.0 → overridden to a dielectric glaze (metalness 0, roughness 0.3, clearcoat 0.5).
   - Black enamel and ink came back as roughness 0.9 "worn-composite" → overridden to gloss (0.35 / 0.32 with clearcoat).
   - PBR-evidence confidences were 0.716, 0.86 and 0.805, all ≥ 0.7.
3. **`textureless` is declared with evidence** on all four materials (lesson 5); the decal boundary is an alpha mask, not a PBR map.
4. **De-lighting:** the box-blur `delight_albedo.py` output (confidence 0.598) lifted the large ink masses, so it was not used for the print. `tools/bake_decal.py` instead normalises each column against its own enamel white.
5. **Ink and bead albedo are compensated for lighting and grading.** The final values are ink (62,62,70) and bead #323238. The photo's blacks are lifted (raw bead 39–50), and the environment had to drop to 0.3 to get the limb falloff. `material_feedback.py` proposed #4B4C54 (the lit-crop mean), which was rejected as baked lighting.
6. **Handle split:** the D is a C-loop tube plus a separate `handle-wall-strip` capsule. One closed tube cannot turn the 24° hairpin at the lower weld without folding through itself; the first self-intersection run found 31 inside vertices.
7. **`turntable_gate.py --allow-holes`:** the only enclosed-background regions are the handle loop (bounding boxes checked at 0° and 180°).
8. **Workspace substitutes for missing skill pieces:**
   - `runtime/scripts/export_mesh_geometry.mjs` is absent from the checkout, so `tools/export_meshes.py` replaces it. It supplies vertex normals; without them the gate steps away from the centroid, which is wrong for a concave cup.
   - The material pipeline/gate (`material_gate.py`) applies only to the multi-region route. This run used the single-crop route, so `material_comparator.py` provided the evidence instead.
9. **Camera fit history** (all values are recorded in `referenceCamera.method`):
   - Distance 8.45 stretched the height by 4%, so it moved to 25.
   - Pitch went 2° → 1° with height-dependent de-projection. Later I tested a near-orthographic camera at 2.3° and rejected it by measurement. The material-pass review's explanation was corrected in the lighting-pass review.
10. **Late fix after the final pass review:**
    - **Problem:** the handle's sockets and collider were authored in body space but resolved in the handle node's frame. `handle-root` ended up at (0.78, 1.27, 0.38), outside the model.
    - **Fix:** I re-authored them in the node frame.
    - **Verification:** the reference render is pixel-identical before and after (diff bbox `None`), so the recorded reviews still hold.

## Reproduce
`tools/author_spec.py` (spec from measurements) → `tools/bake_decal.py` → `tools/build.sh <pass> <dir>`
(generator + patch + bundle + captures) → `tools/gates.sh` / `tools/interior.sh` → reviews in `object-sculpt-spec.json`
(`reviewHistory`). Per-pass evidence is in `renders/<pass>/`; intake material is in `analysis/`, `detail-inventory/`,
`material-evidence/` and `projection/`; checklist evidence is in `evidence/`.

## Notes for the img2three CLAUDE.md (new lessons from this run)
- `append_review.py --action continue` requires `long-axis` and `thickness-axis` viewpoints in `--review-viewpoints-json`. Capture them for real (`tools/capture_axes.py`) rather than just adding labels.
- `self_intersection.py` needs per-vertex normals for vessels, and coarse concave fillets cause false positives from its masked-incident-triangle parity check. Subdivide the fillet.
- Blockout Tier 1 needs `--map-stripped-render` (`tools/capture_mapstripped.py`).
- A handle that looks like it's in pure profile can still be yawed. Check whether the wall-side strip sits inside or outside the body silhouette.
