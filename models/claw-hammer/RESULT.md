# RESULT — Claw hammer (reference/reference.jpg)

**Status:** complete. `forge/next.py` reports `status=complete`, all 8 build passes reviewed `continue`,
0 correction loops used (0/3 per pass, 0/6 total). Profile `generic`.

**Final fidelity estimate: 0.81** (on the skill's scale, between "object reads correctly, local details approximate"
and "strong procedural match for real-time use"). This is a procedural approximation built from a single
photo: the hidden far side is inferred from symmetry and absolute scale is assumed.

## Final renders (`renders/final/`)
- `reference.png`: the fitted reference camera view
- `orbit-000.png`, `orbit-090.png`, `orbit-180.png`, `orbit-270.png`: the viewer's 4 orbits (12° elevation)
- `tight-000…270.png`: the same orbits at 2.2× focal length (the viewer's orbit framing leaves the thin hammer small)
- `comparison.png`: reference | render side by side; `ref-overlay.png`: silhouette diff (red = reference only, cyan = render only)
- Factory: `src/createObjectModel.ts` (`createClawHammerModel`); viewer bundle: `viewer/bundle.js`

## What matches (measured)
| Check | Value |
|---|---|
| Silhouette IoU, reference view (Tier 1, `diagnose_render.py`) | **0.911** (aspect Δ 0.0015, scale Δ 0.017); independent mask IoU 0.913 |
| Camera fit (`authoring/fit_camera.py`, Nelder–Mead on IoU) | pitch 40.3°, fov 19.1°, distance 0.716 m, hammer yaw −19.7° on the table |
| Interior appearance difference vs photo (`interior_difference.py`) | 0.093 (blockout was 0.194) |
| Grip colour, mid grip | render #e85f29 vs reference #ec591b (R 0.98, G 1.07) |
| Cheek and face steel luminance | 0.97 and 1.02 of the reference |
| Per-part colour ΔE (Tier 1) | max 10.5 (striking face); grip 8.8, claw 6.3 |
| Grip groove pitch / slant (`authoring/groove_slant.py`) | 21.5 px vs 21.8 px; same slant direction, similar slope |
| Off-axis gates | multi-angle no collapse; turntable 4/4 azimuths, no holes; self-intersection clean; attachment clean |
| Structure | `check_part_coverage.py`: 0 errors / 0 warnings; 12 named meshes; every mesh picks to its part (browser) |
| Budget | 91,256 triangles (budget 120k), 12 draw calls (budget 24) |

Identity features present: a curved two-prong claw with an open V slot (dark forging-scale lateral faces,
lighter edges); a round chamfered striking face on a short waisted neck; a one-piece steel head and shank with
a camera-projected stamp (monogram + "8-62633 16oz TAIWAN DK") and the white scuff patch; and an orange grip
with 24 carved S-chevron grooves, a smooth zone after the collar, the logo-panel band, the mould parting line,
the lanyard hole and the flared butt.

## What does not match (numbers)
- **Claw lateral faces are still too light:** render #38393b vs reference #211e1f (about 1.8× brighter).
  The claw curvature in the side plane is foreshortened in the single view (±10° uncertainty).
- **The shank scuff patch renders about 11 % bright** (#e5e5e5 vs #cfcdce). These are the photo's own pixels
  re-lit by the scene lights, even with a 0.82 tint.
- **The grip butt has too much blue:** B channel 2.2× the reference (#e85e26 vs #e84c11). ACES desaturates
  bright saturated orange; R and G are within 1.0/1.24.
- **The steel's tonal range is narrower than the photo.** There are no studio softbox reflection streaks; the
  brushing is procedural, not the photo's real grinding marks.
- **Silhouette residuals:** the face disc is about 3 % large toward the lower left, the throat gap between
  claw and shank is slightly narrower, and the butt ends about 7 px further right in the reference view.
- **Small details:** the panel's embossed logo is barely visible (it is low-contrast in the photo too). The
  parting-line notches are absent. The grip and cheek sections are superellipses (an approximation of the
  real rounded-rectangle sections). The stamp is softer than the photo.
- **Assumed or inferred:** overall length is assumed at about 0.32 m (no ruler in the photo); the far side
  mirrors the visible side; the shank cross-section is inferred from shading.

## Manual overrides (with reasons)
1. **`analyze_texture.py` was run read-only** (CLAUDE.md lessons 4/14). It classified bare steel as
   "worn-composite", metalness 0, roughness 0.9, and the grip as "painted-metal". Rejected; the values come
   from the crops instead (`material-evidence/README.md`; each material records `analyzeTextureReadOnly`).
2. **`extract_part_color_recipe.py` got the class right (metal/rubber), but its confidence capped at
   0.60–0.69.** Accepted with `--allow-low-confidence`; the reason is recorded per recipe (`agentOverrideNote`).
   The claw recipe was re-extracted at the material pass. Its first crop was mostly slot shadow
   (rgba(13,8,11)), which made Tier-1 ΔE 41.5; with the new crop it is 6.3.
3. **`textureless` declarations, with evidence:** the grip (solid moulded TPE, whose crop variation is groove
   shading) and the three projected-decal carrier materials.
4. **Steel maps:** the 40×32 px reference PBR crops tiled as a visible weave. P8 (below) replaces them with
   seeded procedural longitudinal brushing. Its palette and roughness come from the same crops, and the
   `referencePbr` evidence stays in the spec.
5. **Lights calibrated against the photo** by measuring region colours before review: key 1.1 (overhead),
   fill 0.4, hemisphere 0.35, environment 1.0, exposure 1.0, white sweep, no table.
6. **Tier 1 runs on the reference's own white background.** The steel is the same grey as the iso #808080
   background, so iso segmentation lost 22.9 % of the render. Details: `evidence/review-contract-read.md`.
7. **The multi-angle and turntable gates use tight orbit captures** (`authoring/capture_tight_orbits.py`).
   From the material pass on, the turntable gate runs on flat silhouettes, because a near-white specular
   highlight was being read as a 973-px "hole".
8. **Decal shells are tagged open** (`geometry.parameters.openEnded`), so `self_intersection.py` skips them;
   the skip is listed in each `meshes.json`. That gate samples about 18.4k of 52.9k vertices (stride), so it is
   not exhaustive.
9. **`detailInventory.mapsTo` was normalised** from `component/feature` to bare ids. `check_part_coverage.py`
   only matches bare ids; strict validation still passes.

## Factory patch (`tools/patch_factory.py`, re-run after every generation; all parameters in `spec.factoryPatch`)
The generator lacks these code paths, so the factory is patched; the spec stays the source of truth.

- P1: superellipse sweep sections (`stations[].squareness`)
- P2: grip rib grooves carved into a densified grip sweep, the logo-panel inset, the lanyard recess, the
  parting line, and groove cavity vertex AO. The generator's repetition emitter is radial-only, so
  `repetitionSystems[grip-rib-grooves]` has count 0 there and its `carved` block drives the patch.
- P3: post-build hook.
- P4: Catmull-Rom densified sweeps (removes the kinks between stations).
- P5: spline claw outline with a 0.4 mm bevel.
- P6: three camera-projected decals of reference pixels on conforming shells: the cheek stamp, the shank
  scuff, and the grip hole interior + logo (lesson 6). They are built by `authoring/make_stamp_ink.py`,
  `make_scuff.py` and `make_grip_marks.py`.
- P7: binds the scene environment to each material, because three r170 ignores `material.envMapIntensity`
  unless `envMap` is set (without this the grip rendered #ff8954).
- P8: procedural brushed steel.
- P9: analytic cavity AO (claw slot and lateral faces, throat, neck waist; lesson 12).
- P10: `sculptRuntime.parts/explode/pick/partOf`, with one shared part definition and decals riding their
  parts.

The spec is re-authored by `authoring/build_spec.py` and `spec_meta.py`; `authoring/rebuild.sh` runs
generate → patch → bundle.

## Process notes / deviations (`evidence/process-notes.md`)
- **structural-pass:** every gate ran (evidence is in `renders/structural-pass/`), but its checklist steps
  were not `state.py mark`-ed before the review was appended. `state.json` therefore archives that pass with
  steps `pending`. The state file was not hand-edited. Every other pass is fully marked.
- **No type check was run:** there is no `tsc` in the workspace. esbuild bundles without type-checking, and
  the browser runs show no page errors.
- **Emission target:** none was requested (`emit_target.py` is a no-op). There are no plugin gates under the
  generic profile.
