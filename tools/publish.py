"""Publish a finished img2three run folder into this library.

usage: python tools/publish.py <run-dir> [--slug <slug>] [--title <title>] [--force]

Copies the run's factory sources (src/*.ts) to models/<slug>/, copies every image the factory
loads by relative URL into models/<slug>/assets/ and rewrites those URLs so they resolve from the
library root (where the gallery page lives), stores the spec, RESULT.md and the reference image,
writes models/<slug>/model.json (factory export, source run, numbers from the run) and regenerates
models/index.ts. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

LIB = Path(__file__).resolve().parents[1]
MODELS = LIB / "models"
# a quoted relative URL to an image, as a texture load in the factory ("../evidence/x.png", "projection/y.png")
ASSET_URL = re.compile(r"""(["'])((?:\.\./|\./)?[A-Za-z0-9_\-./]+\.(?:png|jpe?g|webp))\1""")

# a relative module import: `from '../evidence/x.json'` or `import '../y'`
REL_IMPORT = re.compile(r"""(\bfrom\s+|\bimport\s+)(["'])(\.\.?/[^"']+)\2""")


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def run_numbers(run: Path) -> dict:
    out: dict = {}
    # run.jsonl, plus run.<n>.jsonl from earlier sessions when the run was resumed: sum them all
    turns = cost = ms = 0.0
    sessions = 0
    for log in sorted(run.glob("run*.jsonl")):
        last = None
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if '"type":"result"' in line.replace(" ", ""):
                try:
                    last = json.loads(line)
                except ValueError:
                    continue
        if last:
            sessions += 1
            turns += last.get("num_turns") or 0
            cost += last.get("total_cost_usd") or 0
            ms += last.get("duration_ms") or 0
    if sessions:
        out = {"turns": int(turns), "apiEquivalentCostUsd": round(cost, 2), "durationMinutes": round(ms / 60000)}
        if sessions > 1:
            out["sessions"] = sessions
    result = run / "RESULT.md"
    if result.exists():
        m = re.search(r"fidelity estimate:?\**\s*\**([0-9.]+)", result.read_text(encoding="utf-8", errors="replace"), re.I)
        if m:
            out["fidelityEstimate"] = float(m.group(1).rstrip("."))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--slug")
    ap.add_argument("--title")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    run = Path(a.run).resolve()
    src = run / "src"
    factory = src / "createObjectModel.ts"
    if not factory.exists():
        sys.exit(f"publish: {factory} not found (is this a finished img2three run?)")
    # Only finished runs: an interrupted run can still hold a half-built factory.
    state_file = run / ".img2threejs" / "state.json"
    status = json.loads(state_file.read_text(encoding="utf-8")).get("status") if state_file.exists() else None
    if status != "complete" or not (run / "RESULT.md").exists():
        sys.exit(f"publish: {run.name} is not a finished run (state status={status!r}, "
                 f"RESULT.md {'present' if (run / 'RESULT.md').exists() else 'missing'})")
    text = factory.read_text(encoding="utf-8")
    m = re.search(r"export function (create(\w+)Model)\(", text)
    if not m:
        sys.exit("publish: no `export function create<Name>Model(` in createObjectModel.ts")
    export_name, type_name = m.group(1), m.group(2)
    slug = a.slug or slugify(re.sub(r"-\d{8}-\d{6}$", "", run.name))
    title = a.title or re.sub(r"(?<!^)(?=[A-Z])", " ", type_name)
    dest = MODELS / slug
    if dest.exists():
        previous = json.loads((dest / "model.json").read_text(encoding="utf-8")).get("sourceRun") \
            if (dest / "model.json").exists() else None
        same_run = previous is not None and Path(previous).name == run.name
        if same_run and a.force:
            shutil.rmtree(dest)  # re-publishing the same run
        elif same_run:
            sys.exit(f"publish: {slug} already holds {run.name} (use --force to re-publish it)")
        else:
            # a different run of the same subject never overwrites: it gets its own slug
            n = 2
            while (MODELS / f"{slug}-{n}").exists():
                n += 1
            slug = f"{slug}-{n}"
            dest = MODELS / slug
    (dest / "assets").mkdir(parents=True)

    # factory sources: copy every .ts, rewriting relative image URLs to library-root paths
    viewer_dir = run / "viewer"  # URLs in the run were resolved against viewer/index.html
    copied: dict[str, str] = {}
    for ts in sorted(src.glob("*.ts")):
        body = ts.read_text(encoding="utf-8")

        def rewrite(match: re.Match) -> str:
            quote, url = match.group(1), match.group(2)
            file = (viewer_dir / url).resolve()
            if not file.is_file():
                return match.group(0)  # not a loadable relative asset (a label, an absolute path, ...)
            try:
                rel = file.relative_to(run)
            except ValueError:
                return match.group(0)
            target = dest / "assets" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            new_url = f"models/{slug}/assets/{rel.as_posix()}"
            copied[url] = new_url
            return f"{quote}{new_url}{quote}"

        def rewrite_import(match: re.Match) -> str:
            # a module import that reaches outside src/ (e.g. '../evidence/x.json'): copy the file into
            # models/<slug>/data/<path inside the run> and point the import there
            quote, spec = match.group(2), match.group(3)
            file = (src / spec).resolve()
            if not file.is_file():
                return match.group(0)
            try:
                rel = file.relative_to(run)
            except ValueError:
                return match.group(0)
            if rel.parts[0] == "src":
                return match.group(0)  # a sibling module, copied with the other .ts files
            target = dest / "data" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
            return f"{match.group(1)}{quote}./data/{rel.as_posix()}{quote}"

        body = REL_IMPORT.sub(rewrite_import, body)
        (dest / ts.name).write_text(ASSET_URL.sub(rewrite, body), encoding="utf-8", newline="\n")

    for name in ("object-sculpt-spec.json", "RESULT.md"):
        if (run / name).exists():
            shutil.copy2(run / name, dest / ("spec.json" if name.endswith(".json") else name))
    refs = sorted((run / "reference").glob("reference.*"))
    if refs:
        shutil.copy2(refs[0], dest / f"reference{refs[0].suffix}")
    final = run / "renders" / "final"
    # agents name the photo-vs-render sheet comparison.png or comparison-<view>.png; fall back to the render
    previews = sorted(final.glob("comparison*.png")) or sorted(final.glob("reference.png"))
    if previews:
        kind = "comparison" if previews[0].name.startswith("comparison") else "reference"
        shutil.copy2(previews[0], dest / f"preview-{kind}.png")

    meta = {"slug": slug, "title": title, "factory": "createObjectModel.ts", "export": export_name,
            "typeName": type_name, "sourceRun": str(run), "published": date.today().isoformat(),
            "assetUrls": copied, **run_numbers(run)}
    (dest / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_index()
    print(json.dumps({"published": slug, "export": export_name, "assets": len(copied)}, indent=2))
    return 0


def write_index() -> None:
    """models/index.ts: one registry entry per published model (namespaced, so helper names never collide)."""
    entries, imports = [], []
    for meta_file in sorted(MODELS.glob("*/model.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        ns = re.sub(r"\W", "_", meta["slug"])
        imports.append(f"import * as {ns} from './{meta['slug']}/createObjectModel';")
        refs = sorted(meta_file.parent.glob("reference.*"))
        info = {k: meta.get(k) for k in ("fidelityEstimate", "turns", "apiEquivalentCostUsd", "durationMinutes", "published")}
        info["reference"] = f"models/{meta['slug']}/{refs[0].name}" if refs else None
        entries.append(
            f"  {{ slug: {json.dumps(meta['slug'])}, title: {json.dumps(meta['title'])}, "
            f"module: {ns} as unknown as Record<string, any>, create: {ns}.{meta['export']}, "
            f"info: {json.dumps(info)} }},"
        )
    body = "\n".join([
        "// Generated by tools/publish.py -- do not edit by hand.",
        "import type * as THREE from 'three';",
        *imports,
        "",
        "export type LibraryModel = {",
        "  slug: string;",
        "  title: string;",
        "  module: Record<string, any>;",
        "  create: (options?: Record<string, unknown>) => THREE.Group;",
        "  info: {",
        "    fidelityEstimate?: number | null;",
        "    turns?: number | null;",
        "    apiEquivalentCostUsd?: number | null;",
        "    durationMinutes?: number | null;",
        "    published?: string | null;",
        "    reference?: string | null;",
        "  };",
        "};",
        "",
        "export const models: LibraryModel[] = [",
        *entries,
        "];",
        "",
    ])
    (MODELS / "index.ts").write_text(body, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    raise SystemExit(main())
