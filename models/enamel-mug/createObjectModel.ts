import * as THREE from 'three';
import { attachInteraction } from "./mugInteraction";
import { DECAL_TEXTURE_URI } from "./decalTexture";
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export type ProceduralModelOptions = {
  wireframe?: boolean;
  castShadow?: boolean;
  receiveShadow?: boolean;
  textureSize?: number;
  textureAnisotropy?: number;
  qualityPriority?: 'reference-fidelity' | 'balanced';
};

export type ProceduralModelRuntime = {
  nodes: Record<string, THREE.Object3D>;
  meshes: Record<string, THREE.Mesh>;
  sockets: Record<string, THREE.Object3D>;
  colliders: Record<string, unknown>;
  destructionGroups: Record<string, THREE.Object3D[]>;
};

type SculptMaterialSpec = Record<string, any>;

// Quadric-error decimation (Garland & Heckbert 1997), ported from
// forge/stage3_build/decimate.py so it can run on geometry that only exists at runtime.
//
// WHY IT RUNS HERE RATHER THAN OFFLINE: the Python version reads a mesh file, and this
// factory does not produce one -- an implicit surface is polygonised in the browser from
// its descriptor, so there is no mesh to decimate until the page is running. Baking a
// vertex array into this file instead would contradict the whole point of a code-only
// procedural model.
//
// WHY IT RUNS BEFORE BINDING: a quadric collapse merges two vertices into one, and
// skinIndex/skinWeight would have to be interpolated across that merge to stay correct.
// The generated bind pass recomputes weights from `position` afterwards, so decimating
// first means the weights are simply computed on the surviving vertices and no
// interpolation of skinning data ever has to happen.
//
// LIMITATION, stated rather than discovered later: only `position` survives. UVs are
// dropped and normals are recomputed, so this must not be applied to geometry carrying
// authored UVs. Its intended input is polygonizeSdf output, which is position + index
// + computed normals and therefore loses nothing.
//
// Two collapses are refused, exactly as in the Python original, because both produce a
// mesh that passes a triangle count while no longer being a surface:
//   * one that would FLIP a face -- the neighbouring normal reversing means the surface
//     folded through itself, which is the defect self_intersection.py exists to catch;
//   * one on a BOUNDARY edge -- an open border erodes inward and the silhouette visibly
//     retreats, the classic "the LOD is smaller than the original" artefact.
function decimateGeometry(source: THREE.BufferGeometry, targetRatio: number): THREE.BufferGeometry {
  const MIN_TRIANGLES = 4;
  const position = source.getAttribute('position');
  if (!position || !(targetRatio > 0) || targetRatio >= 1) return source;

  const vertices: number[][] = [];
  for (let i = 0; i < position.count; i++) {
    vertices.push([position.getX(i), position.getY(i), position.getZ(i)]);
  }
  const index = source.getIndex();
  const faces: number[][] = [];
  if (index) {
    for (let i = 0; i + 2 < index.count; i += 3) {
      faces.push([index.getX(i), index.getX(i + 1), index.getX(i + 2)]);
    }
  } else {
    for (let i = 0; i + 2 < position.count; i += 3) faces.push([i, i + 1, i + 2]);
  }
  if (!faces.length) return source;

  const originalCount = faces.length;
  const targetCount = Math.max(MIN_TRIANGLES, Math.floor(originalCount * targetRatio));
  if (originalCount <= targetCount) return source;

  const planeOf = (a: number[], b: number[], c: number[]): number[] | null => {
    const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [
      e1[1] * e2[2] - e1[2] * e2[1],
      e1[2] * e2[0] - e1[0] * e2[2],
      e1[0] * e2[1] - e1[1] * e2[0],
    ];
    const len = Math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len <= 1e-12) return null;
    const u = [n[0] / len, n[1] / len, n[2] / len];
    return [u[0], u[1], u[2], -(u[0] * a[0] + u[1] * a[1] + u[2] * a[2])];
  };
  // Upper triangle of the symmetric 4x4, row-major -- same 10-element layout as the Python side.
  const quadricOf = (p: number[]): number[] => {
    const [a, b, c, d] = p;
    return [a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d];
  };
  const addQuadric = (q1: number[], q2: number[]): number[] => q1.map((x, i) => x + q2[i]);
  const quadricError = (q: number[], v: number[]): number => {
    const [x, y, z] = v;
    return (
      q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x +
      q[4] * y * y + 2 * q[5] * y * z + 2 * q[6] * y +
      q[7] * z * z + 2 * q[8] * z + q[9]
    );
  };

  const alive = faces.map(() => true);
  const vertexFaces: Set<number>[] = vertices.map(() => new Set<number>());
  faces.forEach((face, i) => face.forEach((v) => vertexFaces[v].add(i)));

  const quadrics: number[][] = vertices.map(() => new Array(10).fill(0));
  faces.forEach((face, i) => {
    const p = planeOf(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
    if (!p) { alive[i] = false; return; }
    const q = quadricOf(p);
    face.forEach((v) => { quadrics[v] = addQuadric(quadrics[v], q); });
  });

  const edgeFaces = new Map<string, number>();
  const edgeKey = (u: number, v: number) => (u < v ? u + ':' + v : v + ':' + u);
  faces.forEach((face) => {
    const [a, b, c] = face;
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const k = edgeKey(u, v);
      edgeFaces.set(k, (edgeFaces.get(k) ?? 0) + 1);
    }
  });
  const boundary = new Set<string>();
  edgeFaces.forEach((count, k) => { if (count === 1) boundary.add(k); });

  const midpointOf = (u: number, v: number): number[] =>
    [0, 1, 2].map((i) => (vertices[u][i] + vertices[v][i]) / 2);
  // Midpoint rather than the quadric-optimal position: solving the 3x3 needs an inverse that
  // is singular on flat regions, and the midpoint keeps the collapse on the original surface.
  const costOf = (u: number, v: number): number =>
    quadricError(addQuadric(quadrics[u], quadrics[v]), midpointOf(u, v));

  // Binary heap keyed on collapse cost; entries are stale-checked on pop rather than updated
  // in place, the same lazy-deletion approach heapq gets in the Python original.
  const heap: { cost: number; u: number; v: number }[] = [];
  const heapPush = (entry: { cost: number; u: number; v: number }) => {
    heap.push(entry);
    let i = heap.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (heap[parent].cost <= heap[i].cost) break;
      [heap[parent], heap[i]] = [heap[i], heap[parent]];
      i = parent;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = 2 * i + 2;
        let small = i;
        if (l < heap.length && heap[l].cost < heap[small].cost) small = l;
        if (r < heap.length && heap[r].cost < heap[small].cost) small = r;
        if (small === i) break;
        [heap[small], heap[i]] = [heap[i], heap[small]];
        i = small;
      }
    }
    return top;
  };

  edgeFaces.forEach((_count, k) => {
    if (boundary.has(k)) return;
    const [u, v] = k.split(':').map(Number);
    heapPush({ cost: costOf(u, v), u, v });
  });

  let liveFaces = alive.reduce((n, flag) => n + (flag ? 1 : 0), 0);
  while (heap.length && liveFaces > targetCount) {
    const { u, v } = heapPop();
    if (u === v || !vertexFaces[u].size || !vertexFaces[v].size) continue;
    const shared = new Set<number>();
    vertexFaces[u].forEach((i) => { if (vertexFaces[v].has(i)) shared.add(i); });
    let anyAlive = false;
    shared.forEach((i) => { if (alive[i]) anyAlive = true; });
    if (!anyAlive) continue;
    const merged = midpointOf(u, v);

    let flips = false;
    const affected = new Set<number>();
    vertexFaces[u].forEach((i) => { if (!shared.has(i)) affected.add(i); });
    vertexFaces[v].forEach((i) => { if (!shared.has(i)) affected.add(i); });
    for (const i of affected) {
      if (!alive[i] || flips) continue;
      const face = faces[i];
      const before = planeOf(vertices[face[0]], vertices[face[1]], vertices[face[2]]);
      const pts = face.map((w) => (w === u || w === v ? merged : vertices[w]));
      const after = planeOf(pts[0], pts[1], pts[2]);
      if (!before || !after) { flips = true; break; }
      if (before[0] * after[0] + before[1] * after[1] + before[2] * after[2] < 0) { flips = true; break; }
    }
    if (flips) continue;

    vertices[u] = merged;
    quadrics[u] = addQuadric(quadrics[u], quadrics[v]);
    shared.forEach((i) => { if (alive[i]) { alive[i] = false; liveFaces -= 1; } });
    Array.from(vertexFaces[v]).forEach((i) => {
      if (!alive[i]) return;
      faces[i] = faces[i].map((w) => (w === v ? u : w));
      vertexFaces[u].add(i);
    });
    vertexFaces[v] = new Set<number>();
    Array.from(vertexFaces[u]).forEach((i) => {
      if (!alive[i]) return;
      faces[i].forEach((w) => { if (w !== u) heapPush({ cost: costOf(u, w), u, v: w }); });
    });
  }

  const kept = faces.filter((_face, i) => alive[i]);
  const used = Array.from(new Set(kept.flat())).sort((a, b) => a - b);
  const remap = new Map<number, number>();
  used.forEach((old, next) => remap.set(old, next));
  const outPositions: number[] = [];
  used.forEach((old) => outPositions.push(vertices[old][0], vertices[old][1], vertices[old][2]));
  const outIndices: number[] = [];
  kept.forEach((face) => face.forEach((w) => outIndices.push(remap.get(w)!)));

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(outPositions, 3));
  geometry.setIndex(outIndices);
  geometry.computeVertexNormals();
  geometry.userData.decimation = {
    originalTriangleCount: originalCount,
    triangleCount: kept.length,
    targetTriangleCount: targetCount,
    // Reported because stopping short is not a failure but IS something the caller must know:
    // on a mesh whose remaining collapses would all flip a face, the honest outcome is fewer
    // triangles removed, not a folded mesh that hits the number.
    reachedTarget: kept.length <= targetCount,
  };
  return geometry;
}

// THREE.CapsuleGeometry duplicates every UV-seam vertex (measured: 194 boundary
// edges on the default radius/segments below) -- same benign pattern as box/
// cylinder/sphere/torus, all of which weld cleanly to 0 given a CORRECT weld.
// (A naive vertex-only mergeVertices() reports 64 'non-manifold' edges here, but
// that is a counting artifact, not a real defect: it double-counts a handful of
// near-pole triangles that become degenerate once two of their three corners
// coincide -- confirmed by replicating subdivideCatmullClark's own degenerate-
// triangle-aware vertex identity, which finds a perfectly ordinary 2-manifold.)
// A capsule is the primary shape for skinned limbs/torso (PLAN_1.5), and skinning
// weight computation is O(vertices x bones), so fewer, guaranteed-simple vertices
// is worth having regardless -- authored as a deterministic, closed-by-
// construction mesh instead: shared pole vertices, and
// the radial index taken `% radialSegments` so the seam is never a duplicate
// vertex in the first place, rather than something to weld away afterward.
// Adapted from forge/stage5_rig/emit_rig.py's buildWatertightCapsule (verified
// there: 0 boundary edges, 0 non-manifold edges, deterministic across repeated
// runs) -- ported here rather than imported because this factory and the rig
// emitter are separate generated-output surfaces with no shared runtime module;
// see forge/tests/test_primitive_watertightness.py for the measured proof, and
// coordinate with the rig owner before changing either copy independently.
function buildWatertightCapsule(
  radius: number,
  cylLength: number,
  capSegments: number,
  radialSegments: number,
  heightSegments: number,
): THREE.BufferGeometry {
  const positions: number[] = [];
  const indices: number[] = [];
  const uvs: number[] = [];
  const halfCyl = cylLength / 2;
  const totalSpan = 2 * (Math.PI / 2 * radius) + Math.max(0, cylLength);
  const vOf = (fromBottom: number) => (totalSpan > 0 ? fromBottom / totalSpan : 0);

  const bottomPoleIndex = positions.length / 3;
  positions.push(0, -halfCyl - radius, 0);
  uvs.push(0.5, vOf(0));

  const ringStarts: number[] = [];
  const ringV: number[] = [];
  for (let ring = 1; ring <= capSegments; ring += 1) {
    const phi = (Math.PI / 2) * (ring / capSegments);
    const y = -halfCyl - radius * Math.cos(phi);
    const r = radius * Math.sin(phi);
    const start = positions.length / 3;
    ringStarts.push(start);
    ringV.push(vOf(radius * phi));
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const theta = (radial / radialSegments) * Math.PI * 2;
      positions.push(r * Math.cos(theta), y, r * Math.sin(theta));
      uvs.push(radial / radialSegments, vOf(radius * phi));
    }
  }

  const cylinderRingStarts: number[] = [];
  if (cylLength > 0) {
    for (let step = 1; step <= heightSegments; step += 1) {
      const y = -halfCyl + (cylLength * step) / heightSegments;
      const start = positions.length / 3;
      cylinderRingStarts.push(start);
      const v = vOf(radius * (Math.PI / 2) + halfCyl + y);
      for (let radial = 0; radial < radialSegments; radial += 1) {
        const theta = (radial / radialSegments) * Math.PI * 2;
        positions.push(radius * Math.cos(theta), y, radius * Math.sin(theta));
        uvs.push(radial / radialSegments, v);
      }
    }
  }

  const topRingStarts: number[] = [];
  for (let ring = capSegments - 1; ring >= 1; ring -= 1) {
    const phi = (Math.PI / 2) * (ring / capSegments);
    const y = halfCyl + radius * Math.cos(phi);
    const r = radius * Math.sin(phi);
    const start = positions.length / 3;
    topRingStarts.push(start);
    const v = vOf(radius * (Math.PI / 2) + Math.max(0, cylLength) + radius * (Math.PI / 2 - phi));
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const theta = (radial / radialSegments) * Math.PI * 2;
      positions.push(r * Math.cos(theta), y, r * Math.sin(theta));
      uvs.push(radial / radialSegments, v);
    }
  }

  const topPoleIndex = positions.length / 3;
  positions.push(0, halfCyl + radius, 0);
  uvs.push(0.5, vOf(totalSpan));

  const firstBottomRing = ringStarts[0];
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const next = (radial + 1) % radialSegments;
    indices.push(bottomPoleIndex, firstBottomRing + radial, firstBottomRing + next);
  }

  const allRings = [...ringStarts, ...cylinderRingStarts, ...topRingStarts];
  for (let i = 0; i < allRings.length - 1; i += 1) {
    const a = allRings[i];
    const b = allRings[i + 1];
    for (let radial = 0; radial < radialSegments; radial += 1) {
      const next = (radial + 1) % radialSegments;
      indices.push(a + radial, a + next, b + next);
      indices.push(a + radial, b + next, b + radial);
    }
  }

  const lastRing = allRings[allRings.length - 1];
  for (let radial = 0; radial < radialSegments; radial += 1) {
    const next = (radial + 1) % radialSegments;
    indices.push(topPoleIndex, lastRing + next, lastRing + radial);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
}

function buildLatheGeometry(profile: { points: [number, number][]; segments?: number }): THREE.LatheGeometry {
  const points = profile.points.map(([x, y]) => new THREE.Vector2(Math.max(0.0001, x), y));
  return new THREE.LatheGeometry(points, profile.segments ?? 24);
}

function buildTubeGeometry(
  path: { points: [number, number, number][]; radius?: number; radialSegments?: number; closed?: boolean },
): THREE.TubeGeometry {
  const vectors = path.points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
  const curve = new THREE.CatmullRomCurve3(vectors, path.closed ?? false);
  const tubularSegments = Math.max(8, path.points.length * 6);
  return new THREE.TubeGeometry(curve, tubularSegments, path.radius ?? 0.05, path.radialSegments ?? 8, path.closed ?? false);
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function readLayerNumber(value: unknown, keys: string[], fallback: number): number {
  if (typeof value === 'number') return value;
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    for (const key of keys) {
      if (typeof record[key] === 'number') return record[key] as number;
    }
  }
  return fallback;
}

function hexToRgb(hex: string): [number, number, number] {
  const normalized = /^#[0-9a-f]{3}$/i.test(hex)
    ? '#' + hex.slice(1).split('').map((part) => part + part).join('')
    : hex;
  const value = /^#[0-9a-f]{6}$/i.test(normalized) ? Number.parseInt(normalized.slice(1), 16) : 0x8a7a5f;
  return [clampAlbedoChannel((value >> 16) & 255), clampAlbedoChannel((value >> 8) & 255), clampAlbedoChannel(value & 255)];
}

function materialPalette(spec: SculptMaterialSpec): string[] {
  const palette = spec.colorVariation?.palette;
  if (Array.isArray(palette) && palette.length > 0) return palette.filter((value) => typeof value === 'string');
  const secondary = spec.albedo?.secondary;
  const colors = [spec.baseColor ?? spec.color ?? spec.albedo?.dominant, ...(Array.isArray(secondary) ? secondary : [])];
  return colors.filter((value): value is string => typeof value === 'string' && value.startsWith('#'));
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function clampAlbedoChannel(value: number): number {
  return Math.max(30, Math.min(240, Math.round(value)));
}

function clampPbrF0(value: number): number {
  return Math.max(0.02, Math.min(1, value));
}

function clampPbrIor(value: number): number {
  return Math.max(1, Math.min(2.5, value));
}

function clampPbrMetalness(value: number): number {
  return value >= 0.5 ? 1 : 0;
}

function clampedAlbedoColor(spec: SculptMaterialSpec): THREE.Color {
  const source = typeof spec.baseColor === 'string' ? spec.baseColor : '#8A7A5F';
  // setStyle with an explicit SRGBColorSpace, NOT the numeric constructor.
  //
  // `new THREE.Color(r, g, b)` treats its arguments as LINEAR working-space components,
  // while an authored `baseColor` hex is sRGB. Feeding one to the other skipped the
  // transfer function and lifted every dark albedo: #2e2a28, authored as a near-black
  // vinyl, rendered at roughly sRGB 0.46 — a mid grey. The error is largest exactly where
  // it matters most, because the transfer curve is steepest near black.
  return new THREE.Color().setStyle(source, THREE.SRGBColorSpace);
}

function smoothCurve(value: number): number {
  return value * value * (3 - 2 * value);
}

function periodicHash(x: number, y: number, seed: number, periodX: number, periodY: number): number {
  const wrappedX = ((x % periodX) + periodX) % periodX;
  const wrappedY = ((y % periodY) + periodY) % periodY;
  let value = Math.imul(wrappedX + seed * 17, 374761393) ^ Math.imul(wrappedY + seed * 31, 668265263);
  value = Math.imul(value ^ (value >>> 13), 1274126177);
  return ((value ^ (value >>> 16)) >>> 0) / 4294967295;
}

function periodicValueNoise(u: number, v: number, seed: number, periodX: number, periodY: number): number {
  const x = u * periodX;
  const y = v * periodY;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = smoothCurve(x - x0);
  const ty = smoothCurve(y - y0);
  const a = periodicHash(x0, y0, seed, periodX, periodY);
  const b = periodicHash(x0 + 1, y0, seed, periodX, periodY);
  const c = periodicHash(x0, y0 + 1, seed, periodX, periodY);
  const d = periodicHash(x0 + 1, y0 + 1, seed, periodX, periodY);
  return THREE.MathUtils.lerp(THREE.MathUtils.lerp(a, b, tx), THREE.MathUtils.lerp(c, d, tx), ty);
}

type SurfaceBand = {
  frequency: number;
  amplitude: number;
  stretchX: number;
  stretchY: number;
  ridge: boolean;
};

function surfaceBands(spec: SculptMaterialSpec): SurfaceBand[] {
  const source = Array.isArray(spec.surfaceFrequencyBands) ? spec.surfaceFrequencyBands : [];
  const parsed = source.flatMap((item: unknown) => {
    if (!item || typeof item !== 'object') return [];
    const band = item as Record<string, unknown>;
    const frequency = typeof band.frequency === 'number' ? band.frequency : 0;
    const amplitude = typeof band.amplitude === 'number' ? band.amplitude : 0;
    if (frequency <= 0 || amplitude <= 0) return [];
    const stretch = Array.isArray(band.stretch) ? band.stretch : [1, 1];
    const description = `${String(band.pattern ?? '')} ${String(band.role ?? '')}`.toLowerCase();
    return [{
      frequency,
      amplitude,
      stretchX: typeof stretch[0] === 'number' ? Math.max(0.1, stretch[0]) : 1,
      stretchY: typeof stretch[1] === 'number' ? Math.max(0.1, stretch[1]) : 1,
      ridge: /(ridge|groove|grain|fiber|striated|crack)/.test(description),
    }];
  });
  return parsed.length > 0 ? parsed : [
    { frequency: 2, amplitude: 0.42, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 12, amplitude: 0.22, stretchX: 1, stretchY: 1, ridge: false },
    { frequency: 56, amplitude: 0.08, stretchX: 1, stretchY: 1, ridge: false },
  ];
}

function sampleSurface(u: number, v: number, bands: SurfaceBand[], seed: number): number {
  let value = 0;
  let weight = 0;
  for (let index = 0; index < bands.length; index += 1) {
    const band = bands[index];
    const periodX = Math.max(1, Math.round(band.frequency * band.stretchX));
    const periodY = Math.max(1, Math.round(band.frequency * band.stretchY));
    let sample = periodicValueNoise(u, v, seed + index * 1013, periodX, periodY);
    if (band.ridge) sample = 1 - Math.abs(sample * 2 - 1);
    value += sample * band.amplitude;
    weight += band.amplitude;
  }
  return weight > 0 ? clamp01(value / weight) : 0.5;
}

function mixPalette(colors: [number, number, number][], value: number): [number, number, number] {
  if (colors.length === 1) return colors[0];
  const scaled = clamp01(value) * (colors.length - 1);
  const index = Math.min(colors.length - 2, Math.floor(scaled));
  const mix = scaled - index;
  const a = colors[index];
  const b = colors[index + 1];
  return [
    Math.round(THREE.MathUtils.lerp(a[0], b[0], mix)),
    Math.round(THREE.MathUtils.lerp(a[1], b[1], mix)),
    Math.round(THREE.MathUtils.lerp(a[2], b[2], mix)),
  ];
}

type ColorGradientStop = { offset: number; color: string };
type ColorGradientSpec = {
  type: 'linear' | 'radial';
  axis: [number, number];
  stops: ColorGradientStop[];
};

function parseRgba(value: string): [number, number, number] {
  const match = /rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/.exec(value);
  if (!match) return [138, 122, 95];
  return [clampAlbedoChannel(Number(match[1])), clampAlbedoChannel(Number(match[2])), clampAlbedoChannel(Number(match[3]))];
}

// Analytical per-pixel gradient sample. The extraction schema's colorGradient carries
// exact rgba(...) stop colors (see extract_part_color_recipe.py), so this samples the
// same trend directly in JS math rather than round-tripping through a Canvas 2D
// createLinearGradient/createRadialGradient object — same visual result, and it composes
// directly with the existing noise/height-correlated colorVariation blend below.
function sampleColorGradient(gradient: ColorGradientSpec, u: number, v: number): [number, number, number] {
  const stops = gradient.stops.length >= 2 ? gradient.stops : [{ offset: 0, color: 'rgba(138,122,95,1)' }, { offset: 1, color: 'rgba(138,122,95,1)' }];
  let t: number;
  if (gradient.type === 'radial') {
    const [cx, cy] = gradient.axis;
    const dx = u - cx;
    const dy = v - cy;
    const maxRadius = Math.max(0.001, Math.hypot(Math.max(cx, 1 - cx), Math.max(cy, 1 - cy)));
    t = clamp01(Math.hypot(dx, dy) / maxRadius);
  } else {
    const [ax, ay] = gradient.axis;
    const projection = (u - 0.5) * ax + (v - 0.5) * ay;
    const maxProjection = 0.5 * (Math.abs(ax) + Math.abs(ay)) || 0.5;
    t = clamp01(projection / maxProjection + 0.5);
  }
  const scaled = t * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.max(0, Math.floor(scaled)));
  const mix = scaled - index;
  const a = parseRgba(stops[index].color);
  const b = parseRgba(stops[index + 1].color);
  return [
    THREE.MathUtils.lerp(a[0], b[0], mix),
    THREE.MathUtils.lerp(a[1], b[1], mix),
    THREE.MathUtils.lerp(a[2], b[2], mix),
  ];
}

function writePixel(data: Uint8ClampedArray, offset: number, red: number, green: number, blue: number): void {
  data[offset] = Math.max(0, Math.min(255, Math.round(red)));
  data[offset + 1] = Math.max(0, Math.min(255, Math.round(green)));
  data[offset + 2] = Math.max(0, Math.min(255, Math.round(blue)));
  data[offset + 3] = 255;
}

function makeCanvas(size: number): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  return canvas;
}

function createMapTexture(
  canvas: HTMLCanvasElement,
  colorSpace: THREE.ColorSpace,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): THREE.CanvasTexture {
  const texture = new THREE.CanvasTexture(canvas);
  const projection = spec.textureProjection && typeof spec.textureProjection === 'object' ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [2, 2];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === 'number' ? repeat[0] : 2,
    typeof repeat[1] === 'number' ? repeat[1] : 2,
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}

type ProceduralTextureSet = {
  albedo: THREE.Texture;
  roughness: THREE.Texture;
  height: THREE.Texture;
  normal: THREE.Texture;
  ao: THREE.Texture;
  source: 'reference-pixel-extraction' | 'procedural';
};

function referenceMapUrl(spec: SculptMaterialSpec, channel: string): string | null {
  const reference = spec.referencePbr;
  if (!reference || typeof reference !== 'object') return null;
  if (reference.usable === false) return null;
  const confidence = typeof reference.confidence === 'number'
    ? reference.confidence
    : (typeof reference.estimatedFidelity === 'number' ? reference.estimatedFidelity : 0);
  const threshold = typeof reference.targetThreshold === 'number' ? reference.targetThreshold : 0.7;
  if (confidence < threshold) return null;
  const maps = reference.maps;
  if (!maps || typeof maps !== 'object') return null;
  const map = (maps as Record<string, unknown>)[channel];
  if (!map || typeof map !== 'object') return null;
  const record = map as Record<string, unknown>;
  const url = typeof record.url === 'string' && record.url.trim() ? record.url : record.path;
  return typeof url === 'string' && url.trim() ? url : null;
}

function createLoadedMapTexture(
  url: string,
  colorSpace: THREE.ColorSpace,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): THREE.Texture {
  const texture = new THREE.TextureLoader().load(url);
  const projection = spec.textureProjection && typeof spec.textureProjection === 'object' ? spec.textureProjection : {};
  const repeat = Array.isArray(projection.repeat) ? projection.repeat : [1, 1];
  texture.colorSpace = colorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(
    typeof repeat[0] === 'number' ? repeat[0] : 1,
    typeof repeat[1] === 'number' ? repeat[1] : 1,
  );
  texture.anisotropy = Math.max(1, Math.round(options.textureAnisotropy ?? projection.anisotropy ?? 8));
  texture.needsUpdate = true;
  return texture;
}

function makeReferenceTextureSet(spec: SculptMaterialSpec, options: ProceduralModelOptions): ProceduralTextureSet | null {
  const albedo = referenceMapUrl(spec, 'albedo');
  const roughness = referenceMapUrl(spec, 'roughness');
  const height = referenceMapUrl(spec, 'height');
  const normal = referenceMapUrl(spec, 'normal');
  const ao = referenceMapUrl(spec, 'ao');
  if (!albedo || !roughness || !height || !normal || !ao) return null;
  return {
    albedo: createLoadedMapTexture(albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createLoadedMapTexture(roughness, THREE.NoColorSpace, spec, options),
    height: createLoadedMapTexture(height, THREE.NoColorSpace, spec, options),
    normal: createLoadedMapTexture(normal, THREE.NoColorSpace, spec, options),
    ao: createLoadedMapTexture(ao, THREE.NoColorSpace, spec, options),
    source: 'reference-pixel-extraction',
  };
}

function makeProceduralTextureSet(
  id: string,
  spec: SculptMaterialSpec,
  options: ProceduralModelOptions,
): ProceduralTextureSet | null {
  if (typeof document === 'undefined') return null;
  const qualityFirst = (options.qualityPriority ?? 'reference-fidelity') === 'reference-fidelity';
  const requested = options.textureSize ?? spec.textureResolution;
  const requestedSize = typeof requested === 'number' && Number.isFinite(requested)
    ? requested
    : (qualityFirst ? 1024 : 512);
  const size = Math.max(256, Math.min(2048, 2 ** Math.round(Math.log2(requestedSize))));
  const canvases = {
    albedo: makeCanvas(size),
    roughness: makeCanvas(size),
    height: makeCanvas(size),
    normal: makeCanvas(size),
    ao: makeCanvas(size),
  };
  const contexts = {
    albedo: canvases.albedo.getContext('2d'),
    roughness: canvases.roughness.getContext('2d'),
    height: canvases.height.getContext('2d'),
    normal: canvases.normal.getContext('2d'),
    ao: canvases.ao.getContext('2d'),
  };
  if (!contexts.albedo || !contexts.roughness || !contexts.height || !contexts.normal || !contexts.ao) return null;
  const images = {
    albedo: contexts.albedo.createImageData(size, size),
    roughness: contexts.roughness.createImageData(size, size),
    height: contexts.height.createImageData(size, size),
    normal: contexts.normal.createImageData(size, size),
    ao: contexts.ao.createImageData(size, size),
  };
  const seed = hashString(id);
  const bands = surfaceBands(spec);
  const heightField = new Float32Array(size * size);
  const roughnessField = new Float32Array(size * size);
  const palette = materialPalette(spec);
  const fallback = typeof spec.baseColor === 'string' ? spec.baseColor : '#8A7A5F';
  const colors = (palette.length >= 2 ? palette : [fallback, '#6E614B', '#A08F70']).map(hexToRgb);
  const baseRoughness = clamp01(readLayerNumber(spec.roughness, ['base'], 0.76));
  const roughnessVariation = clamp01(readLayerNumber(spec.roughness, ['variation'], 0.18));
  const colorAmplitude = clamp01(readLayerNumber(spec.colorVariation, ['amplitude', 'variation'], 0.18));
  const heightCorrelation = clamp01(readLayerNumber(spec.colorVariation, ['heightCorrelation'], 0.3));
  const colorGradient: ColorGradientSpec | undefined = spec.colorGradient;
  for (let y = 0; y < size; y += 1) {
    const v = y / size;
    for (let x = 0; x < size; x += 1) {
      const u = x / size;
      const index = y * size + x;
      const height = sampleSurface(u, v, bands, seed + 101);
      const roughNoise = sampleSurface(u, v, bands, seed + 7001);
      const colorNoise = sampleSurface(u, v, bands, seed + 15013);
      heightField[index] = height;
      roughnessField[index] = clamp01(baseRoughness + (roughNoise - 0.5) * roughnessVariation * 2);
      let color: [number, number, number];
      if (colorGradient) {
        // Evidence-derived spatial gradient (Plan 1.3 Workstream C) takes priority
        // over the noise-based palette blend below — it is a measured trend, not a guess.
        color = sampleColorGradient(colorGradient, u, v);
      } else {
        const paletteValue = clamp01(
          0.5 + (colorNoise - 0.5) * colorAmplitude * 2 + (height - 0.5) * heightCorrelation
        );
        color = mixPalette(colors, paletteValue);
      }
      writePixel(images.albedo.data, index * 4, color[0], color[1], color[2]);
    }
  }
  const normalStrength = Math.max(0.05, readLayerNumber(spec.normal, ['strength', 'amplitude'], 0.35));
  const aoStrength = clamp01(readLayerNumber(spec.ambientOcclusion, ['cavityStrength', 'strength'], 0.35));
  for (let y = 0; y < size; y += 1) {
    const up = ((y - 1 + size) % size) * size;
    const down = ((y + 1) % size) * size;
    for (let x = 0; x < size; x += 1) {
      const left = (x - 1 + size) % size;
      const right = (x + 1) % size;
      const index = y * size + x;
      const center = heightField[index];
      const dx = (heightField[y * size + right] - heightField[y * size + left]) * normalStrength * 6;
      const dy = (heightField[down + x] - heightField[up + x]) * normalStrength * 6;
      const inverseLength = 1 / Math.sqrt(dx * dx + dy * dy + 1);
      const normalX = -dx * inverseLength;
      const normalY = -dy * inverseLength;
      const normalZ = inverseLength;
      const neighborAverage = (
        heightField[y * size + left] + heightField[y * size + right]
        + heightField[up + x] + heightField[down + x]
      ) * 0.25;
      const cavity = Math.max(0, neighborAverage - center);
      const ao = clamp01(1 - aoStrength * (cavity * 12 + (1 - center) * 0.16));
      const offset = index * 4;
      const heightByte = center * 255;
      const roughnessByte = roughnessField[index] * 255;
      writePixel(images.height.data, offset, heightByte, heightByte, heightByte);
      writePixel(images.roughness.data, offset, roughnessByte, roughnessByte, roughnessByte);
      writePixel(
        images.normal.data, offset,
        (normalX * 0.5 + 0.5) * 255,
        (normalY * 0.5 + 0.5) * 255,
        (normalZ * 0.5 + 0.5) * 255,
      );
      writePixel(images.ao.data, offset, ao * 255, ao * 255, ao * 255);
    }
  }
  contexts.albedo.putImageData(images.albedo, 0, 0);
  contexts.roughness.putImageData(images.roughness, 0, 0);
  contexts.height.putImageData(images.height, 0, 0);
  contexts.normal.putImageData(images.normal, 0, 0);
  contexts.ao.putImageData(images.ao, 0, 0);
  return {
    albedo: createMapTexture(canvases.albedo, THREE.SRGBColorSpace, spec, options),
    roughness: createMapTexture(canvases.roughness, THREE.NoColorSpace, spec, options),
    height: createMapTexture(canvases.height, THREE.NoColorSpace, spec, options),
    normal: createMapTexture(canvases.normal, THREE.NoColorSpace, spec, options),
    ao: createMapTexture(canvases.ao, THREE.NoColorSpace, spec, options),
    source: 'procedural',
  };
}

function createSculptMaterial(id: string, spec: SculptMaterialSpec, options: ProceduralModelOptions, denseComponent = false): THREE.MeshPhysicalMaterial {
  // A material that declares -- with evidence -- that its subject carries no texture
  // detail gets NO texture set. Synthesising one anyway is not a harmless default: the
  // branch below then forces color to white and roughness to 1 and reads both from the
  // generated maps, so the authored albedo and the reference-derived roughness are both
  // discarded, and the model gains mottling the reference does not have. Measured on the
  // tuxedo cat, whose black fur rendered as speckled grey-and-white from a palette that
  // only ever described two flat regions.
  const textureless = (spec.textureless as { declared?: boolean } | undefined)?.declared === true;
  const textures = textureless
    ? null
    : makeReferenceTextureSet(spec, options) ?? makeProceduralTextureSet(id, spec, options);
  const material = new THREE.MeshPhysicalMaterial({
    color: textures ? 0xffffff : clampedAlbedoColor(spec),
    roughness: textures ? 1 : clamp01(readLayerNumber(spec.roughness, ['base'], 0.76)),
    metalness: clampPbrMetalness(readLayerNumber(spec.metalness, ['base'], 0.0)),
    clearcoat: clamp01(readLayerNumber(spec.clearcoat, ['base', 'amount'], 0)),
    clearcoatRoughness: clamp01(readLayerNumber(spec.clearcoatRoughness, ['base'], 0.25)),
    transmission: clamp01(readLayerNumber(spec.transmission, ['base', 'amount'], 0)),
    ior: clampPbrIor(readLayerNumber(spec.ior, ['base', 'value'], 1.5)),
    thickness: Math.max(0, readLayerNumber(spec.thickness, ['base', 'amount'], 0)),
    attenuationDistance: Math.max(0.001, readLayerNumber(spec.attenuationDistance, ['base', 'value'], Infinity)),
    attenuationColor: new THREE.Color(typeof spec.attenuationColor === 'string' ? spec.attenuationColor : '#ffffff'),
    sheen: clamp01(readLayerNumber(spec.sheen, ['base', 'amount'], 0)),
    sheenColor: new THREE.Color(typeof spec.sheenColor === 'string' ? spec.sheenColor : '#ffffff'),
    sheenRoughness: clamp01(readLayerNumber(spec.sheenRoughness, ['base'], 1.0)),
    iridescence: clamp01(readLayerNumber(spec.iridescence, ['base', 'amount'], 0)),
    iridescenceIOR: clampPbrIor(readLayerNumber(spec.iridescenceIOR, ['base', 'value'], 1.3)),
    anisotropy: clamp01(readLayerNumber(spec.anisotropy, ['base', 'amount'], 0)),
    anisotropyRotation: readLayerNumber(spec.anisotropy, ['rotation'], 0),
    specularIntensity: clampPbrF0(readLayerNumber(spec.specularF0 ?? spec.f0 ?? spec.specularIntensity, ['base', 'value'], 1.0)),
    specularColor: new THREE.Color(typeof spec.specularColor === 'string' ? spec.specularColor : '#ffffff'),
    emissive: new THREE.Color(typeof spec.emissive === 'string' ? spec.emissive : '#000000'),
    emissiveIntensity: Math.max(0, readLayerNumber(spec.emissiveIntensity, ['base'], 1.0)),
    opacity: clamp01(readLayerNumber(spec.opacity, ['base'], 1)),
    transparent: readLayerNumber(spec.transmission, ['base', 'amount'], 0) > 0 || readLayerNumber(spec.opacity, ['base'], 1) < 1,
    alphaTest: Math.max(0, readLayerNumber(spec.alpha, ['cutoff', 'alphaTest'], 0)),
    wireframe: options.wireframe ?? false,
    side: spec.doubleSided === true ? THREE.DoubleSide : THREE.FrontSide,
    flatShading: spec.flatShading === true,
  });
  if (textures) {
    material.map = textures.albedo;
    material.roughnessMap = textures.roughness;
    material.normalMap = textures.normal;
    material.normalScale.setScalar(Math.max(0.05, readLayerNumber(spec.normal, ['strength', 'amplitude'], 0.35)));
    material.aoMap = textures.ao;
    material.aoMap.channel = 0;
    material.aoMapIntensity = readLayerNumber(spec.ambientOcclusion, ['cavityStrength', 'strength'], 0.35);
    const denseMesh = denseComponent || spec.denseMesh === true || spec.geometryDensity === 'dense' || spec.topologyClass === 'dense';
    const bumpScale = Math.max(0, readLayerNumber(spec.bump, ['amplitude', 'strength'], 0));
    const effectiveBumpScale = denseMesh ? Math.max(0.05, bumpScale) : bumpScale;
    if (effectiveBumpScale > 0) {
      material.bumpMap = textures.height;
      material.bumpScale = effectiveBumpScale;
    }
    const displacementScale = Math.max(0, readLayerNumber(spec.displacement, ['amplitude', 'strength'], 0));
    const effectiveDisplacementScale = denseMesh ? Math.max(0.005, displacementScale) : displacementScale;
    if (effectiveDisplacementScale > 0) {
      material.displacementMap = textures.height;
      material.displacementScale = effectiveDisplacementScale;
      material.displacementBias = -effectiveDisplacementScale * 0.5;
    }
  }
  material.envMapIntensity = readLayerNumber(spec, ['envMapIntensity'], 0.8);
  material.userData.sculptMaterial = spec;
  material.userData.proceduralMapsIndependent = true;
  material.userData.pbrConstraints = { albedoRange: [30, 240], binaryMetalness: true, f0Range: [0.02, 1], iorRange: [1, 2.5] };
  material.userData.pbrTextureSource = textures?.source ?? 'flat-fallback';
  material.userData.referencePbr = spec.referencePbr ?? null;
  material.userData.referenceMaterialId = spec.referenceMaterialId ?? spec.materialReference?.profileId ?? null;
  material.userData.materialEvidence = spec.materialEvidence ?? null;
  material.userData.validationViews = spec.materialReference?.validationViews ?? [];
  material.needsUpdate = true;
  return material;
}

type AttachmentEndpoint = {
  start: THREE.Vector3;
  midpoint: THREE.Vector3;
  quaternion: THREE.Quaternion;
  length: number;
  baseRadius: number;
  endRadius: number;
};

function readVector3(value: unknown, fallback: [number, number, number]): THREE.Vector3 {
  if (Array.isArray(value) && value.length === 3 && value.every((item) => typeof item === 'number')) {
    return new THREE.Vector3(value[0], value[1], value[2]);
  }
  return new THREE.Vector3(fallback[0], fallback[1], fallback[2]);
}

function readNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function makeAttachmentEndpoint(attachment: unknown): AttachmentEndpoint | null {
  if (!attachment || typeof attachment !== 'object') return null;
  const record = attachment as Record<string, unknown>;
  const start = readVector3(record.localStart, [0, 0, 0]);
  const end = readVector3(record.localEnd, [0, 1, 0]);
  const delta = end.clone().sub(start);
  const length = delta.length();
  if (length <= 0.0001) return null;
  const direction = delta.clone().normalize();
  const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
  const baseRadius = Math.max(0.005, readNumber(record.baseRadius, 0.06));
  const endRadius = Math.max(0.003, readNumber(record.endRadius, baseRadius * 0.55));
  return {
    start,
    midpoint: delta.multiplyScalar(0.5),
    quaternion,
    length,
    baseRadius,
    endRadius,
  };
}

// Generated from ObjectSculptSpec target: Enamel mug
// Sculpt build pass: optimization-pass
// This factory is intentionally pass-gated. Finish browser screenshot review before unlocking deeper passes.

/* patched:tools/patch_factory.py */
// Procedural cavity AO: inner-facing vertices of a vessel get a vertex-colour occlusion factor (lipAo at the top,
// easing to deepAo by rampDepth below it). The material is cloned so other meshes sharing it are unaffected.
function applyCavityOcclusion(mesh: THREE.Mesh, lipAo: number, deepAo: number, rampDepth: number, bandY0 = 0, bandY1 = 0, bandAo = 1): void {
  const g = mesh.geometry; if (!g.attributes.normal) g.computeVertexNormals();
  const p = g.attributes.position, n = g.attributes.normal; const col = new Float32Array(p.count * 3);
  let top = -Infinity; for (let i = 0; i < p.count; i++) top = Math.max(top, p.getY(i));
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i), r = Math.hypot(x, z);
    const radialN = r > 1e-6 ? (n.getX(i) * x + n.getZ(i) * z) / r : 0;
    const inner = radialN < -0.3 || (n.getY(i) > 0.7 && y > 0.004 && y < top - 0.05);
    let ao = 1;
    if (inner) { const t = Math.min(1, Math.max(0, (top - y) / rampDepth)); const s = t * t * (3 - 2 * t); ao = lipAo + (deepAo - lipAo) * s; }
    else if (radialN > 0.3 && bandY1 > bandY0 && y > bandY0 - 0.012) {
      // overhang occlusion on the outer lip flare under the rim bead (ramps in over 0.012 below bandY0)
      const t = Math.min(1, Math.max(0, (y - (bandY0 - 0.012)) / (bandY1 - bandY0 + 0.012))); ao = 1 + (bandAo - 1) * t * t * (3 - 2 * t);
    }
    col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = ao;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  const mat = (mesh.material as THREE.MeshPhysicalMaterial).clone(); mat.vertexColors = true; mesh.material = mat;
}


/* patched:tools/patch_factory.py */
function decalInkMaterial(base: THREE.Material | undefined): THREE.Material {
  const src = base as THREE.MeshPhysicalMaterial | undefined;
  const m = src && (src as any).isMeshStandardMaterial ? src.clone() : new THREE.MeshPhysicalMaterial({ color: 0x141416, roughness: 0.32 });
  const tex = new THREE.TextureLoader().load(DECAL_TEXTURE_URI);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  (m as THREE.MeshPhysicalMaterial).alphaMap = null;
  (m as THREE.MeshPhysicalMaterial).map = tex;  // rgb = ink colour, alpha = projected ink coverage
  (m as THREE.MeshPhysicalMaterial).color.set(0xffffff);
  m.transparent = true;
  m.alphaTest = 0.02;
  m.depthWrite = false;
  m.polygonOffset = true;
  m.polygonOffsetFactor = -2;
  m.polygonOffsetUnits = -2;
  m.userData.referenceMaterialId = "decal-ink";
  return m;
}

export function createEnamelMugModel(options: ProceduralModelOptions = {}): THREE.Group {
  const root = new THREE.Group();
  root.name = "Enamel mug";
  root.userData.reconstructionEvidence = {"itemFamily": null, "subtype": null, "componentAdapter": null, "route": null, "exactnessTier": null, "referenceCamera": {"solved": true, "fovDegrees": 2.707, "distance": 25.0, "target": [0.157, 0.415, 0.0], "orientation": {"yaw": 0.0, "pitch": 1.0, "roll": 0}, "referenceCrop": {"image": "reference/reference.jpg", "box": [643.5, 196, 1147.5, 556], "renderSize": [840, 600]}, "method": "silhouette extents: 0.85 units = 259 px; crop 360 px tall -> 1.1815 units -> dist = 1.1815 / (2 tan 4 deg); pitch from rim/base ellipse semi-minor/semi-major ~0.03; yaw 0 from handle reach 0.417 D (profile). | refit: dist 8.45 stretched height +4% (front base edge 5% closer); moved to dist 25 (telephoto) keeping the 1.1815-unit frame height. | target y 0.43 -> 0.415: base-centre row in the render was 7.6 px below the reference's (518.4 vs 510.8). | material-pass: pitch 2 -> 1 deg (camera height ~0.85 = rim height; reference rim shows no opening, base sagitta still ~2 deg); profile heights re-derived with elevation varying 0 (rim) .. 2 deg (base). | lighting-pass: reverted the pitch-1 refit. Rim top edge drops ~7 px toward the ends (back arc seen from above) and base sagitta ~5.3 px -> camera looks down ~2-2.5 deg at both; near-orthographic dist 60, fov 1.128, pitch 2.3. The dark rim band is bead + shadowed interior (cavity AO bake on the lathe interior). | lighting-pass: tested near-orthographic dist 60 / pitch 2.3 (band 17 px vs ref 10 px) -> reverted to dist 25 / pitch 1.0 (band 245-255 vs ref 245-254). Rim-end frown unmodelled.", "confidence": 0.7}, "approximationNotes": []};
  root.userData.materialPipeline = {};
  root.userData.materialReferenceRegistry = null;

  const materialMap: Record<string, THREE.Material> = {};
  materialMap["white-enamel"] = createSculptMaterial(
    "white-enamel",
    {"id": "white-enamel", "name": "White vitreous enamel", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#E8ECEE", "color": "#E8ECEE", "albedo": {"dominant": "#E8ECEE", "secondary": ["#E2E3E1", "#C8CDD0"], "samplingNotes": "lit wall 218-226 luma (cool white, b>r by 4-8), shaded right limb 130-160; de-lit albedo ~#ECECE8"}, "colorVariation": {"palette": ["#E8ECEE", "#E2E3E1", "#C8CDD0"], "pattern": "uniform (flat fired finish)", "amplitude": 0.0, "heightCorrelation": 0.0}, "roughness": {"base": 0.3, "variation": 0.03, "map": "none (uniform glass coat)", "localResponse": "see localOverrides"}, "metalness": {"base": 0.0, "variation": 0.0}, "clearcoat": 0.5, "clearcoatRoughness": 0.12, "localOverrides": [{"id": "enamel-glass-coat", "kind": "gloss", "region": "whole wall", "roughness": 0.3, "clearcoat": 0.5, "clearcoatRoughness": 0.12, "evidenceRef": "analysis/crop2x.png", "notes": "broad soft highlight, no mirror reflections"}, {"id": "base-edge-chip", "kind": "chip", "region": "base edge azimuth -52..-34 deg", "color": "#262422", "metalness": 0.5, "roughness": 0.55, "realizedBy": "base-chip component", "evidenceRef": "detail-inventory/base.png"}], "qualityTier": "hero", "textureless": {"declared": true, "evidence": ["Zone crops detail-inventory/body-left.png, body-right.png: wall luma varies only with lighting (smooth gradient 226 -> 130 toward the shaded limb); no grain, pores, orange-peel or speckle resolvable at 259 px diameter.", "Fired vitreous enamel is a glass coat: uniform albedo, uniform low roughness."]}, "materialEvidence": {"crop": "material-evidence/white-enamel-crop.png (742,270,782,440)", "pbrExtraction": {"confidence": 0.716, "verdict": "pass", "warnings": "tiny foreground mask (crop is all subject; expected)"}, "analyzeTexture": {"tool": {"finishClass": "brushed-steel", "metalness": 1.0, "roughness": 0.35, "clearcoat": 0, "anisotropy": 1.0, "proceduralTexture": "brushed"}, "override": {"finishClass": "gloss-enamel", "metalness": 0.0, "roughness": 0.3, "clearcoat": 0.5, "clearcoatRoughness": 0.12, "anisotropy": 0, "proceduralTexture": "none"}, "reason": "CLAUDE.md lesson 4 reproduced: white enamel misread as brushed steel. Vitreous enamel is a glass (dielectric) coat: the reference shows no streaked/anisotropic highlight, no metallic tint in reflections, and a diffuse white albedo of 229-236 under soft light. Metal with albedo #E8ECEE would render as chrome."}, "palette": ["#E0E5E7", "#E6EAED", "#E5E9EC", "#E3E7EA"], "source": "material-evidence/material-evidence.json"}, "finishClass": "gloss-enamel"},
    options
  );
  materialMap["black-enamel"] = createSculptMaterial(
    "black-enamel",
    {"id": "black-enamel", "name": "Black enamel (rim bead)", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#323238", "color": "#323238", "albedo": {"dominant": "#323238", "secondary": ["#2E2F35"], "samplingNotes": "rim rows 245-254 luma 39-50, highlight streak to ~60. material-pass: #18181B/r0.25/cc0.6 rendered a broad crown highlight over a L30 body (comparator dE00 22.7, material_feedback proposed the lit crop mean #4B4C54); albedo set to #2A2A2F (lifted-black grading, same as ink) and gloss softened"}, "colorVariation": {"palette": ["#323238", "#2E2F35"], "pattern": "uniform (flat fired finish)", "amplitude": 0.0, "heightCorrelation": 0.0}, "roughness": {"base": 0.35, "variation": 0.03, "map": "none (uniform glass coat)", "localResponse": "see localOverrides"}, "metalness": {"base": 0.0, "variation": 0.0}, "clearcoat": 0.35, "clearcoatRoughness": 0.2, "localOverrides": [{"id": "rim-highlight-streak", "kind": "gloss", "region": "top of bead", "roughness": 0.3, "evidenceRef": "detail-inventory/rim.png", "notes": "thin specular line along the bead crown from the overhead key"}], "qualityTier": "hero", "textureless": {"declared": true, "evidence": ["detail-inventory/rim.png: bead is uniform black with one specular streak; no texture detail."]}, "materialEvidence": {"crop": "material-evidence/black-enamel-crop.png (760,247,940,253)", "pbrExtraction": {"confidence": 0.86, "verdict": "pass"}, "analyzeTexture": {"tool": {"finishClass": "worn-composite", "roughness": 0.9, "clearcoat": 0, "proceduralTexture": "mottle"}, "override": {"finishClass": "gloss-enamel", "roughness": 0.25, "clearcoat": 0.6, "clearcoatRoughness": 0.1, "proceduralTexture": "none"}, "reason": "The bead shows a sharp thin specular streak along its crown (detail-inventory/rim.png); roughness 0.9 cannot produce it. 'Mottle' is the specular streak being read as albedo variation; palette entry #7B8495 is that highlight, not albedo."}, "palette": ["#2E2D31", "#303740", "#2C2D31"], "source": "material-evidence/material-evidence.json"}, "finishClass": "gloss-enamel"},
    options
  );
  materialMap["decal-ink"] = createSculptMaterial(
    "decal-ink",
    {"id": "decal-ink", "name": "Fired-on black print", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#3E3E46", "color": "#3E3E46", "albedo": {"dominant": "#3E3E46", "secondary": [], "samplingNotes": "skull ink core luma 25-40 in the lit region"}, "colorVariation": {"palette": ["#3E3E46"], "pattern": "uniform (flat fired finish)", "amplitude": 0.0, "heightCorrelation": 0.0}, "roughness": {"base": 0.32, "variation": 0.03, "map": "none (uniform glass coat)", "localResponse": "see localOverrides"}, "metalness": {"base": 0.0, "variation": 0.0}, "clearcoat": 0.5, "clearcoatRoughness": 0.12, "localOverrides": [{"id": "skull-print", "kind": "decal", "region": "azimuth -29..31 deg, y 0.15..0.66", "mask": "projection/decal_uv.png (alpha)", "evidenceRef": "detail-inventory/decal.png"}, {"id": "lettering-knockout", "kind": "linework", "region": "inside cranium", "mask": "alpha holes in projection/decal_uv.png", "notes": "lettering is enamel showing through the ink", "evidenceRef": "analysis/crop2x.png"}, {"id": "sunburst-strokes", "kind": "decal", "region": "azimuth -76 deg, y 0.38", "mask": "drawn strokes in projection/decal_uv.png", "evidenceRef": "analysis/leftedge3x.png"}], "qualityTier": "hero", "textureless": {"declared": true, "evidence": ["Ink is a flat fired colour with the same gloss as the enamel (detail-inventory/decal.png); its only structure is the print boundary, carried as an alpha mask on the decal-shell component, not as a PBR texture."]}, "materialEvidence": {"crop": "material-evidence/decal-ink-crop.png (880,392,910,402)", "pbrExtraction": {"confidence": 0.805, "verdict": "pass"}, "analyzeTexture": {"tool": {"finishClass": "worn-composite", "roughness": 0.9, "clearcoat": 0}, "override": {"roughness": 0.32, "clearcoat": 0.5, "clearcoatRoughness": 0.12}, "reason": "Fired-on print sits under/in the same glass coat as the white enamel; the wall highlight passes over the ink without a gloss break in the reference crop."}, "palette": ["#2E2D33", "#302F36", "#26252E"], "source": "material-evidence/material-evidence.json"}, "finishClass": "fired-print"},
    options
  );
  materialMap["exposed-steel"] = createSculptMaterial(
    "exposed-steel",
    {"id": "exposed-steel", "name": "Exposed steel at chip", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#262422", "color": "#262422", "albedo": {"dominant": "#262422", "secondary": ["#3a2f28"], "samplingNotes": "chip luma 35-50"}, "colorVariation": {"palette": ["#262422", "#3a2f28"], "pattern": "uniform (flat fired finish)", "amplitude": 0.0, "heightCorrelation": 0.0}, "roughness": {"base": 0.55, "variation": 0.03, "map": "none (uniform glass coat)", "localResponse": "see localOverrides"}, "metalness": {"base": 0.5, "variation": 0.0}, "clearcoat": 0.0, "clearcoatRoughness": 0.1, "localOverrides": [], "qualityTier": "hero", "textureless": {"declared": true, "evidence": ["Chip is ~20 px; no resolvable surface detail."]}, "materialEvidence": {"crop": "detail-inventory/base.png (chip, too small for extraction)", "pbrExtraction": {"confidence": null, "verdict": "not-run: ~20 px region"}, "analyzeTexture": null, "palette": [], "source": "material-evidence/material-evidence.json"}, "finishClass": "raw-steel"},
    options
  );

  const nodes: Record<string, THREE.Object3D> = { root };
  const meshes: Record<string, THREE.Mesh> = {};
  const sockets: Record<string, THREE.Object3D> = {};
  const colliders: Record<string, unknown> = {};
  const destructionGroups: Record<string, THREE.Object3D[]> = {};

  const endpoint_body_shell_0 = makeAttachmentEndpoint(null);
  const node_body_shell_0 = new THREE.Group();
  node_body_shell_0.name = "Mug body shell (pressed steel, white enamel)__pivot";
  node_body_shell_0.scale.set(1, 1, 1);
  if (endpoint_body_shell_0) {
    node_body_shell_0.position.copy(endpoint_body_shell_0.start);
    node_body_shell_0.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_body_shell_0.position.set(0.0, 0.0, 0.0);
    node_body_shell_0.rotation.set(0.0, 0.0, 0.0);
  }
  node_body_shell_0.userData.sculptComponent = {"id": "body-shell", "name": "Mug body shell (pressed steel, white enamel)", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.85, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Revolved thin-walled vessel: one continuous surface of revolution (floor, radiused base corner, straight wall, flared lip) -> lathe, never a cylinder primitive.", "geometryDescriptor": {"topologyIntent": "thin-walled surface of revolution with inner and outer skin", "edgeTreatment": {"type": "fillet", "bevelRadius": 0.02, "segments": 4}, "deformationStack": [], "uvStrategy": "lathe (u = azimuth, v = profile length)", "normalStrategy": "vertex normals from lathe", "latheProfile": {"points": [[0.0, 0.0], [0.4, 0.0], [0.413, 0.0015], [0.4205, 0.006], [0.424, 0.013], [0.425, 0.022], [0.425, 0.7861], [0.4265, 0.8078], [0.4315, 0.8215], [0.4385, 0.8313], [0.4335, 0.8363], [0.4225, 0.8235], [0.4185, 0.8038], [0.417, 0.7861], [0.417, 0.0405], [0.4164, 0.0348], [0.4148, 0.0293], [0.4122, 0.0242], [0.4086, 0.0197], [0.4042, 0.0159], [0.3992, 0.0131], [0.3937, 0.0113], [0.388, 0.0105], [0.387, 0.0105], [0.0, 0.0105]], "segments": 96}, "cavityOcclusion": {"method": "vertex-colour AO bake on inner-facing lathe vertices (tools/patch_factory.py)", "lipAo": 0.5, "deepAo": 0.22, "rampDepth": 0.22, "overhangBand": {"y0": 0.788, "y1": 0.8121, "ao": 0.5, "reason": "outer lip flare under the bead overhang reads ~0.6 of wall brightness in the reference (view rows 100-108); viewer shadow normalBias 0.015 cannot resolve a 0.04 overhang"}, "reason": "hemisphere/environment fill has no occlusion and lit the cup interior white; in the reference the interior seen through the opening is dark (rim band rows 245-254 solid)"}}, "parent": null, "attachment": null, "dimensions": {"width": 0.85, "height": 0.846, "depth": 0.85, "units": "scene (1 = 10 cm)", "confidence": 0.85}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "base", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "rim-socket", "localPosition": [0, 0.8286, 0], "localRotation": [0, 0, 0], "purpose": "rim bead seat"}, {"id": "handle-upper-socket", "localPosition": [0.391, 0.6367, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle upper weld"}, {"id": "handle-lower-socket", "localPosition": [0.391, 0.0897, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle lower weld"}, {"id": "decal-socket", "localPosition": [0, 0.44, 0], "localRotation": [0, 0, 0], "purpose": "print carrier"}], "collider": {"type": "cylinder", "offset": [0, 0.43, 0], "scale": [0.85, 0.86, 0.85], "isTrigger": false, "notes": "body proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "base-corner-fillet", "kind": "bevel", "description": "radius ~0.022 at the base corner (lathe points 1-5)", "effect": "geometry", "evidenceRef": "detail-inventory/base.png"}, {"id": "interior-cavity", "kind": "hole", "description": "open top, wall 0.008 thick, inner floor at y=0.0105; interior white enamel (inferred, unseen)", "effect": "geometry", "evidenceRef": "analysis/crop2x.png"}, {"id": "lip-flare", "kind": "ridge", "description": "wall flares from r=0.425 at y=0.80 into the rolled bead (shaded band under rim, rows 255-261)", "effect": "geometry", "evidenceRef": "detail-inventory/rim.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["full-object", "analysis/crop2x.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(162, 170, 174, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.85, "evidenceNote": "lit wall (740-780, 300-480) vs shaded right limb (930-965); vitreous enamel = glass coat on steel -> ceramic class"}};
  node_body_shell_0.userData.actionProfile = {"animationRole": "root", "pivot": {"mode": "base", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "rim-socket", "localPosition": [0, 0.8286, 0], "localRotation": [0, 0, 0], "purpose": "rim bead seat"}, {"id": "handle-upper-socket", "localPosition": [0.391, 0.6367, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle upper weld"}, {"id": "handle-lower-socket", "localPosition": [0.391, 0.0897, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle lower weld"}, {"id": "decal-socket", "localPosition": [0, 0.44, 0], "localRotation": [0, 0, 0], "purpose": "print carrier"}], "collider": {"type": "cylinder", "offset": [0, 0.43, 0], "scale": [0.85, 0.86, 0.85], "isTrigger": false, "notes": "body proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}};
  (nodes["root"] ?? root).add(node_body_shell_0);
  nodes["body-shell"] = node_body_shell_0;
  const mesh_body_shell_0Geometry = endpoint_body_shell_0
    ? new THREE.CylinderGeometry(endpoint_body_shell_0.endRadius, endpoint_body_shell_0.baseRadius, endpoint_body_shell_0.length, 16, 6)
    : buildLatheGeometry({"points": [[0.0, 0.0], [0.4, 0.0], [0.413, 0.0015], [0.4205, 0.006], [0.424, 0.013], [0.425, 0.022], [0.425, 0.7861], [0.4265, 0.8078], [0.4315, 0.8215], [0.4385, 0.8313], [0.4335, 0.8363], [0.4225, 0.8235], [0.4185, 0.8038], [0.417, 0.7861], [0.417, 0.0405], [0.4164, 0.0348], [0.4148, 0.0293], [0.4122, 0.0242], [0.4086, 0.0197], [0.4042, 0.0159], [0.3992, 0.0131], [0.3937, 0.0113], [0.388, 0.0105], [0.387, 0.0105], [0.0, 0.0105]], "segments": 96});
  if (!endpoint_body_shell_0) {
    mesh_body_shell_0Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_body_shell_0 = new THREE.Mesh(
    mesh_body_shell_0Geometry,
    materialMap["white-enamel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  applyCavityOcclusion(mesh_body_shell_0, 0.5, 0.22, 0.22, 0.788, 0.8121, 0.5); /* patched:tools/patch_factory.py */
  mesh_body_shell_0.name = "Mug body shell (pressed steel, white enamel)";
  if (endpoint_body_shell_0) {
    mesh_body_shell_0.position.copy(endpoint_body_shell_0.midpoint);
    mesh_body_shell_0.quaternion.copy(endpoint_body_shell_0.quaternion);
  }
  mesh_body_shell_0.castShadow = options.castShadow ?? true;
  mesh_body_shell_0.receiveShadow = options.receiveShadow ?? true;
  mesh_body_shell_0.userData.sculptComponent = {"id": "body-shell", "name": "Mug body shell (pressed steel, white enamel)", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.85, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Revolved thin-walled vessel: one continuous surface of revolution (floor, radiused base corner, straight wall, flared lip) -> lathe, never a cylinder primitive.", "geometryDescriptor": {"topologyIntent": "thin-walled surface of revolution with inner and outer skin", "edgeTreatment": {"type": "fillet", "bevelRadius": 0.02, "segments": 4}, "deformationStack": [], "uvStrategy": "lathe (u = azimuth, v = profile length)", "normalStrategy": "vertex normals from lathe", "latheProfile": {"points": [[0.0, 0.0], [0.4, 0.0], [0.413, 0.0015], [0.4205, 0.006], [0.424, 0.013], [0.425, 0.022], [0.425, 0.7861], [0.4265, 0.8078], [0.4315, 0.8215], [0.4385, 0.8313], [0.4335, 0.8363], [0.4225, 0.8235], [0.4185, 0.8038], [0.417, 0.7861], [0.417, 0.0405], [0.4164, 0.0348], [0.4148, 0.0293], [0.4122, 0.0242], [0.4086, 0.0197], [0.4042, 0.0159], [0.3992, 0.0131], [0.3937, 0.0113], [0.388, 0.0105], [0.387, 0.0105], [0.0, 0.0105]], "segments": 96}, "cavityOcclusion": {"method": "vertex-colour AO bake on inner-facing lathe vertices (tools/patch_factory.py)", "lipAo": 0.5, "deepAo": 0.22, "rampDepth": 0.22, "overhangBand": {"y0": 0.788, "y1": 0.8121, "ao": 0.5, "reason": "outer lip flare under the bead overhang reads ~0.6 of wall brightness in the reference (view rows 100-108); viewer shadow normalBias 0.015 cannot resolve a 0.04 overhang"}, "reason": "hemisphere/environment fill has no occlusion and lit the cup interior white; in the reference the interior seen through the opening is dark (rim band rows 245-254 solid)"}}, "parent": null, "attachment": null, "dimensions": {"width": 0.85, "height": 0.846, "depth": 0.85, "units": "scene (1 = 10 cm)", "confidence": 0.85}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "base", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "rim-socket", "localPosition": [0, 0.8286, 0], "localRotation": [0, 0, 0], "purpose": "rim bead seat"}, {"id": "handle-upper-socket", "localPosition": [0.391, 0.6367, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle upper weld"}, {"id": "handle-lower-socket", "localPosition": [0.391, 0.0897, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle lower weld"}, {"id": "decal-socket", "localPosition": [0, 0.44, 0], "localRotation": [0, 0, 0], "purpose": "print carrier"}], "collider": {"type": "cylinder", "offset": [0, 0.43, 0], "scale": [0.85, 0.86, 0.85], "isTrigger": false, "notes": "body proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "base-corner-fillet", "kind": "bevel", "description": "radius ~0.022 at the base corner (lathe points 1-5)", "effect": "geometry", "evidenceRef": "detail-inventory/base.png"}, {"id": "interior-cavity", "kind": "hole", "description": "open top, wall 0.008 thick, inner floor at y=0.0105; interior white enamel (inferred, unseen)", "effect": "geometry", "evidenceRef": "analysis/crop2x.png"}, {"id": "lip-flare", "kind": "ridge", "description": "wall flares from r=0.425 at y=0.80 into the rolled bead (shaded band under rim, rows 255-261)", "effect": "geometry", "evidenceRef": "detail-inventory/rim.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["full-object", "analysis/crop2x.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(162, 170, 174, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.85, "evidenceNote": "lit wall (740-780, 300-480) vs shaded right limb (930-965); vitreous enamel = glass coat on steel -> ceramic class"}};
  node_body_shell_0.add(mesh_body_shell_0);
  meshes["body-shell"] = mesh_body_shell_0;
  colliders["body-shell"] = {"type": "cylinder", "offset": [0, 0.43, 0], "scale": [0.85, 0.86, 0.85], "isTrigger": false, "notes": "body proxy"};
  destructionGroups["body-shell"] ??= [];
  destructionGroups["body-shell"].push(node_body_shell_0);
  const socket_body_shell_rim_socket_0 = new THREE.Object3D();
  socket_body_shell_rim_socket_0.name = "rim-socket";
  socket_body_shell_rim_socket_0.position.set(0.0, 0.8286, 0.0);
  socket_body_shell_rim_socket_0.rotation.set(0.0, 0.0, 0.0);
  socket_body_shell_rim_socket_0.userData.socket = {"id": "rim-socket", "localPosition": [0, 0.8286, 0], "localRotation": [0, 0, 0], "purpose": "rim bead seat"};
  node_body_shell_0.add(socket_body_shell_rim_socket_0);
  sockets["body-shell:rim-socket"] = socket_body_shell_rim_socket_0;
  const socket_body_shell_handle_upper_socket_1 = new THREE.Object3D();
  socket_body_shell_handle_upper_socket_1.name = "handle-upper-socket";
  socket_body_shell_handle_upper_socket_1.position.set(0.391, 0.6367, 0.1907);
  socket_body_shell_handle_upper_socket_1.rotation.set(0.0, 0.0, 0.0);
  socket_body_shell_handle_upper_socket_1.userData.socket = {"id": "handle-upper-socket", "localPosition": [0.391, 0.6367, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle upper weld"};
  node_body_shell_0.add(socket_body_shell_handle_upper_socket_1);
  sockets["body-shell:handle-upper-socket"] = socket_body_shell_handle_upper_socket_1;
  const socket_body_shell_handle_lower_socket_2 = new THREE.Object3D();
  socket_body_shell_handle_lower_socket_2.name = "handle-lower-socket";
  socket_body_shell_handle_lower_socket_2.position.set(0.391, 0.0897, 0.1907);
  socket_body_shell_handle_lower_socket_2.rotation.set(0.0, 0.0, 0.0);
  socket_body_shell_handle_lower_socket_2.userData.socket = {"id": "handle-lower-socket", "localPosition": [0.391, 0.0897, 0.1907], "localRotation": [0, 0, 0], "purpose": "handle lower weld"};
  node_body_shell_0.add(socket_body_shell_handle_lower_socket_2);
  sockets["body-shell:handle-lower-socket"] = socket_body_shell_handle_lower_socket_2;
  const socket_body_shell_decal_socket_3 = new THREE.Object3D();
  socket_body_shell_decal_socket_3.name = "decal-socket";
  socket_body_shell_decal_socket_3.position.set(0.0, 0.44, 0.0);
  socket_body_shell_decal_socket_3.rotation.set(0.0, 0.0, 0.0);
  socket_body_shell_decal_socket_3.userData.socket = {"id": "decal-socket", "localPosition": [0, 0.44, 0], "localRotation": [0, 0, 0], "purpose": "print carrier"};
  node_body_shell_0.add(socket_body_shell_decal_socket_3);
  sockets["body-shell:decal-socket"] = socket_body_shell_decal_socket_3;

  const endpoint_rim_bead_1 = makeAttachmentEndpoint(null);
  const node_rim_bead_1 = new THREE.Group();
  node_rim_bead_1.name = "Rolled rim bead (black enamel)__pivot";
  node_rim_bead_1.scale.set(1, 1, 1);
  if (endpoint_rim_bead_1) {
    node_rim_bead_1.position.copy(endpoint_rim_bead_1.start);
    node_rim_bead_1.rotation.set(1.5707963267948966, 0.0, 0.0);
  } else {
    node_rim_bead_1.position.set(0.0, 0.8286, 0.0);
    node_rim_bead_1.rotation.set(1.5707963267948966, 0.0, 0.0);
  }
  node_rim_bead_1.userData.sculptComponent = {"id": "rim-bead", "name": "Rolled rim bead (black enamel)", "level": "meso", "role": "trim", "importance": 0.95, "confidence": 0.85, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Rolled lip is a closed round-section ring around the opening -> torus (ring-relative tube ratio).", "geometryDescriptor": {"topologyIntent": "round-section rolled bead", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "torus", "normalStrategy": "smooth", "torusTubeRatio": 0.03684}, "parent": "body-shell", "attachment": null, "dimensions": {"width": 0.9288, "height": 0.033, "depth": 0.9288, "units": "scene (1 = 10 cm)", "confidence": 0.85}, "transform": {"position": [0, 0.8286, 0], "rotation": [1.5707963267948966, 0, 0], "scale": [0.99533, 0.99533, 0.99533]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [0.93, 0.033, 0.93], "isTrigger": false, "notes": "thin ring proxy"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "black-enamel"}}, "material": "black-enamel", "materialLayers": ["black-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "rim-rolled-bead", "kind": "ridge", "description": "bead major radius 0.4479, minor 0.0165; overhangs wall by 0.0394 (4.7% of D per side)", "effect": "geometry", "evidenceRef": "detail-inventory/rim.png"}], "surfaceDetail": {"macroRoughness": 0.35, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none (rolled steel under black enamel)", "displacementPattern": "none", "occlusionPattern": "self-shadow under the bead on the lip flare", "edgeWearPattern": "none visible", "notes": "specular streak along the crown from the key"}, "evidenceRefs": ["detail-inventory/rim.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(24, 24, 27, 1.0)", "secondaryAlbedo": "rgba(46, 47, 53, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.85, "evidenceNote": "rim rows 245-254"}};
  node_rim_bead_1.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [0.93, 0.033, 0.93], "isTrigger": false, "notes": "thin ring proxy"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "black-enamel"}};
  (nodes["body-shell"] ?? root).add(node_rim_bead_1);
  nodes["rim-bead"] = node_rim_bead_1;
  const mesh_rim_bead_1Geometry = endpoint_rim_bead_1
    ? new THREE.CylinderGeometry(endpoint_rim_bead_1.endRadius, endpoint_rim_bead_1.baseRadius, endpoint_rim_bead_1.length, 16, 6)
    : new THREE.TorusGeometry(0.45, 0.0166, 12, 48);
  if (!endpoint_rim_bead_1) {
    mesh_rim_bead_1Geometry.scale(0.99533, 0.99533, 0.99533);
  }
  const mesh_rim_bead_1 = new THREE.Mesh(
    mesh_rim_bead_1Geometry,
    materialMap["black-enamel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_rim_bead_1.name = "Rolled rim bead (black enamel)";
  if (endpoint_rim_bead_1) {
    mesh_rim_bead_1.position.copy(endpoint_rim_bead_1.midpoint);
    mesh_rim_bead_1.quaternion.copy(endpoint_rim_bead_1.quaternion);
  }
  mesh_rim_bead_1.castShadow = options.castShadow ?? true;
  mesh_rim_bead_1.receiveShadow = options.receiveShadow ?? true;
  mesh_rim_bead_1.userData.sculptComponent = {"id": "rim-bead", "name": "Rolled rim bead (black enamel)", "level": "meso", "role": "trim", "importance": 0.95, "confidence": 0.85, "primitive": "torus", "topologyClass": "assembled-solid", "topologyRationale": "Rolled lip is a closed round-section ring around the opening -> torus (ring-relative tube ratio).", "geometryDescriptor": {"topologyIntent": "round-section rolled bead", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "torus", "normalStrategy": "smooth", "torusTubeRatio": 0.03684}, "parent": "body-shell", "attachment": null, "dimensions": {"width": 0.9288, "height": 0.033, "depth": 0.9288, "units": "scene (1 = 10 cm)", "confidence": 0.85}, "transform": {"position": [0, 0.8286, 0], "rotation": [1.5707963267948966, 0, 0], "scale": [0.99533, 0.99533, 0.99533]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0, 0, 0], "scale": [0.93, 0.033, 0.93], "isTrigger": false, "notes": "thin ring proxy"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "black-enamel"}}, "material": "black-enamel", "materialLayers": ["black-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "rim-rolled-bead", "kind": "ridge", "description": "bead major radius 0.4479, minor 0.0165; overhangs wall by 0.0394 (4.7% of D per side)", "effect": "geometry", "evidenceRef": "detail-inventory/rim.png"}], "surfaceDetail": {"macroRoughness": 0.35, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none (rolled steel under black enamel)", "displacementPattern": "none", "occlusionPattern": "self-shadow under the bead on the lip flare", "edgeWearPattern": "none visible", "notes": "specular streak along the crown from the key"}, "evidenceRefs": ["detail-inventory/rim.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(24, 24, 27, 1.0)", "secondaryAlbedo": "rgba(46, 47, 53, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.85, "evidenceNote": "rim rows 245-254"}};
  node_rim_bead_1.add(mesh_rim_bead_1);
  meshes["rim-bead"] = mesh_rim_bead_1;
  colliders["rim-bead"] = {"type": "box", "offset": [0, 0, 0], "scale": [0.93, 0.033, 0.93], "isTrigger": false, "notes": "thin ring proxy"};
  destructionGroups["body-shell"] ??= [];
  destructionGroups["body-shell"].push(node_rim_bead_1);

  const attachment_handle_2 = {"parentSocket": "handle-upper-socket", "childSocket": "handle-root", "localStart": [0.391, 0.6367, 0.1907], "localEnd": [0.391, 0.0897, 0.1907], "contactType": "weld", "baseRadius": 0.0245, "endRadius": 0.0245, "embedDepth": 0.012, "overlap": 0.012, "gapTolerance": 0.002, "notes": "D-ring strap: both ends and the straight side are welded to the wall at r=0.4325 (half the strap embedded). localStart/localEnd are the two weld points; the visible loop follows geometryDescriptor.tubePath (generator DEF-5 replaces it with a straight cylinder -> tools/patch_factory.py restores the authored path)."};
  const endpoint_handle_2 = makeAttachmentEndpoint(attachment_handle_2);
  const node_handle_2 = new THREE.Group();
  node_handle_2.name = "D-ring strap handle (white enamel)__pivot";
  node_handle_2.scale.set(1, 1, 1);
  if (endpoint_handle_2) {
    node_handle_2.position.copy(endpoint_handle_2.start);
    node_handle_2.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_handle_2.position.set(0.0, 0.0, 0.0);
    node_handle_2.rotation.set(0.0, 0.0, 0.0);
  }
  node_handle_2.userData.sculptComponent = {"id": "handle", "name": "D-ring strap handle (white enamel)", "level": "macro", "role": "handle", "importance": 0.95, "confidence": 0.75, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Bent strap following a closed planar D path -> tube along the traced centreline (strap section = tube scaled 1.5x in Z).", "geometryDescriptor": {"topologyIntent": "open swept C-loop strap, both ends sunk 0.004 into the wall; straight side = handle-wall-strip", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "tube", "normalStrategy": "smooth", "tubePath": {"points": [[0.3784, 0.6367, 0.1846], [0.409, 0.6352, 0.1995], [0.4314, 0.6336, 0.2104], [0.4972, 0.6339, 0.2425], [0.5927, 0.6311, 0.2891], [0.6564, 0.6152, 0.3201], [0.6978, 0.583, 0.3403], [0.7232, 0.5379, 0.3527], [0.7391, 0.4733, 0.3605], [0.7423, 0.4087, 0.3621], [0.7328, 0.3504, 0.3574], [0.7137, 0.2984, 0.3481], [0.6882, 0.2561, 0.3357], [0.6245, 0.2035, 0.3046], [0.5449, 0.1602, 0.2658], [0.4972, 0.1401, 0.2425], [0.4526, 0.1199, 0.2208], [0.4176, 0.1041, 0.2037], [0.3784, 0.0862, 0.1846]], "radius": 0.0245, "radialSegments": 16, "closed": false}, "strapSectionAspectZ": 1.0}, "parent": "body-shell", "attachment": {"parentSocket": "handle-upper-socket", "childSocket": "handle-root", "localStart": [0.391, 0.6367, 0.1907], "localEnd": [0.391, 0.0897, 0.1907], "contactType": "weld", "baseRadius": 0.0245, "endRadius": 0.0245, "embedDepth": 0.012, "overlap": 0.012, "gapTolerance": 0.002, "notes": "D-ring strap: both ends and the straight side are welded to the wall at r=0.4325 (half the strap embedded). localStart/localEnd are the two weld points; the visible loop follows geometryDescriptor.tubePath (generator DEF-5 replaces it with a straight cylinder -> tools/patch_factory.py restores the authored path)."}, "dimensions": {"width": 0.35, "height": 0.58, "depth": 0.049, "units": "scene (1 = 10 cm)", "confidence": 0.75}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "grab-target", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.75}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "handle-root", "localPosition": [0, 0, 0], "localRotation": [0, 0, 0], "purpose": "attachment root (= upper weld)"}, {"id": "grip-point", "localPosition": [0.2382, -0.2367, 0.1162], "localRotation": [0, 0, 0], "purpose": "hand IK target on the loop's outer side"}], "collider": {"type": "box", "offset": [0.1842, -0.2567, 0.0899], "scale": [0.38, 0.58, 0.08], "isTrigger": true, "notes": "grab volume in the handle node frame (node at the upper weld), plane at 64 deg"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 60.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-upper-weld", "kind": "seam", "description": "upper end welded just under the rim (y=0.650)", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}, {"id": "handle-lower-weld", "kind": "seam", "description": "lower arm meets the wall at y=0.098 with a sharp corner", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}, {"id": "handle-d-loop", "kind": "contour", "description": "handle plane at azimuth 64 deg (yawed 26 deg toward the reference camera); top arm horizontal, rounded outer corner, outer side rho=0.84 (reach 0.40 = 0.47 D beyond the wall), lower arm angled back down", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none", "displacementPattern": "none", "occlusionPattern": "contact occlusion where the strap meets the wall", "edgeWearPattern": "none visible", "notes": "rolled strap reads as a round section"}, "evidenceRefs": ["detail-inventory/handle.png", "analysis/ref_mask_full.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(209, 217, 220, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.8, "evidenceNote": "handle outer side (1070-1080, 350-400)"}};
  node_handle_2.userData.actionProfile = {"animationRole": "grab-target", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.75}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "handle-root", "localPosition": [0, 0, 0], "localRotation": [0, 0, 0], "purpose": "attachment root (= upper weld)"}, {"id": "grip-point", "localPosition": [0.2382, -0.2367, 0.1162], "localRotation": [0, 0, 0], "purpose": "hand IK target on the loop's outer side"}], "collider": {"type": "box", "offset": [0.1842, -0.2567, 0.0899], "scale": [0.38, 0.58, 0.08], "isTrigger": true, "notes": "grab volume in the handle node frame (node at the upper weld), plane at 64 deg"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 60.0, "debrisMaterial": "white-enamel"}};
  (nodes["body-shell"] ?? root).add(node_handle_2);
  nodes["handle"] = node_handle_2;
  const mesh_handle_2Geometry = /* patched:tools/patch_factory.py */ buildTubeGeometry({"points": [[0.3784, 0.6367, 0.1846], [0.409, 0.6352, 0.1995], [0.4314, 0.6336, 0.2104], [0.4972, 0.6339, 0.2425], [0.5927, 0.6311, 0.2891], [0.6564, 0.6152, 0.3201], [0.6978, 0.583, 0.3403], [0.7232, 0.5379, 0.3527], [0.7391, 0.4733, 0.3605], [0.7423, 0.4087, 0.3621], [0.7328, 0.3504, 0.3574], [0.7137, 0.2984, 0.3481], [0.6882, 0.2561, 0.3357], [0.6245, 0.2035, 0.3046], [0.5449, 0.1602, 0.2658], [0.4972, 0.1401, 0.2425], [0.4526, 0.1199, 0.2208], [0.4176, 0.1041, 0.2037], [0.3784, 0.0862, 0.1846]], "radius": 0.0245, "radialSegments": 16, "closed": false});
  mesh_handle_2Geometry.scale(1, 1, 1.0);
  if (!endpoint_handle_2) {
    mesh_handle_2Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_handle_2 = new THREE.Mesh(
    mesh_handle_2Geometry,
    materialMap["white-enamel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_handle_2.name = "D-ring strap handle (white enamel)";
  if (endpoint_handle_2) {
    mesh_handle_2.position.copy(endpoint_handle_2.start).multiplyScalar(-1); /* patched:tools/patch_factory.py */
  }
  mesh_handle_2.castShadow = options.castShadow ?? true;
  mesh_handle_2.receiveShadow = options.receiveShadow ?? true;
  mesh_handle_2.userData.sculptComponent = {"id": "handle", "name": "D-ring strap handle (white enamel)", "level": "macro", "role": "handle", "importance": 0.95, "confidence": 0.75, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Bent strap following a closed planar D path -> tube along the traced centreline (strap section = tube scaled 1.5x in Z).", "geometryDescriptor": {"topologyIntent": "open swept C-loop strap, both ends sunk 0.004 into the wall; straight side = handle-wall-strip", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "tube", "normalStrategy": "smooth", "tubePath": {"points": [[0.3784, 0.6367, 0.1846], [0.409, 0.6352, 0.1995], [0.4314, 0.6336, 0.2104], [0.4972, 0.6339, 0.2425], [0.5927, 0.6311, 0.2891], [0.6564, 0.6152, 0.3201], [0.6978, 0.583, 0.3403], [0.7232, 0.5379, 0.3527], [0.7391, 0.4733, 0.3605], [0.7423, 0.4087, 0.3621], [0.7328, 0.3504, 0.3574], [0.7137, 0.2984, 0.3481], [0.6882, 0.2561, 0.3357], [0.6245, 0.2035, 0.3046], [0.5449, 0.1602, 0.2658], [0.4972, 0.1401, 0.2425], [0.4526, 0.1199, 0.2208], [0.4176, 0.1041, 0.2037], [0.3784, 0.0862, 0.1846]], "radius": 0.0245, "radialSegments": 16, "closed": false}, "strapSectionAspectZ": 1.0}, "parent": "body-shell", "attachment": {"parentSocket": "handle-upper-socket", "childSocket": "handle-root", "localStart": [0.391, 0.6367, 0.1907], "localEnd": [0.391, 0.0897, 0.1907], "contactType": "weld", "baseRadius": 0.0245, "endRadius": 0.0245, "embedDepth": 0.012, "overlap": 0.012, "gapTolerance": 0.002, "notes": "D-ring strap: both ends and the straight side are welded to the wall at r=0.4325 (half the strap embedded). localStart/localEnd are the two weld points; the visible loop follows geometryDescriptor.tubePath (generator DEF-5 replaces it with a straight cylinder -> tools/patch_factory.py restores the authored path)."}, "dimensions": {"width": 0.35, "height": 0.58, "depth": 0.049, "units": "scene (1 = 10 cm)", "confidence": 0.75}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "grab-target", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.75}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "handle-root", "localPosition": [0, 0, 0], "localRotation": [0, 0, 0], "purpose": "attachment root (= upper weld)"}, {"id": "grip-point", "localPosition": [0.2382, -0.2367, 0.1162], "localRotation": [0, 0, 0], "purpose": "hand IK target on the loop's outer side"}], "collider": {"type": "box", "offset": [0.1842, -0.2567, 0.0899], "scale": [0.38, 0.58, 0.08], "isTrigger": true, "notes": "grab volume in the handle node frame (node at the upper weld), plane at 64 deg"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 60.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-upper-weld", "kind": "seam", "description": "upper end welded just under the rim (y=0.650)", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}, {"id": "handle-lower-weld", "kind": "seam", "description": "lower arm meets the wall at y=0.098 with a sharp corner", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}, {"id": "handle-d-loop", "kind": "contour", "description": "handle plane at azimuth 64 deg (yawed 26 deg toward the reference camera); top arm horizontal, rounded outer corner, outer side rho=0.84 (reach 0.40 = 0.47 D beyond the wall), lower arm angled back down", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none", "displacementPattern": "none", "occlusionPattern": "contact occlusion where the strap meets the wall", "edgeWearPattern": "none visible", "notes": "rolled strap reads as a round section"}, "evidenceRefs": ["detail-inventory/handle.png", "analysis/ref_mask_full.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(209, 217, 220, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.8, "evidenceNote": "handle outer side (1070-1080, 350-400)"}};
  node_handle_2.add(mesh_handle_2);
  meshes["handle"] = mesh_handle_2;
  colliders["handle"] = {"type": "box", "offset": [0.1842, -0.2567, 0.0899], "scale": [0.38, 0.58, 0.08], "isTrigger": true, "notes": "grab volume in the handle node frame (node at the upper weld), plane at 64 deg"};
  destructionGroups["handle"] ??= [];
  destructionGroups["handle"].push(node_handle_2);
  const socket_handle_handle_root_0 = new THREE.Object3D();
  socket_handle_handle_root_0.name = "handle-root";
  socket_handle_handle_root_0.position.set(0.0, 0.0, 0.0);
  socket_handle_handle_root_0.rotation.set(0.0, 0.0, 0.0);
  socket_handle_handle_root_0.userData.socket = {"id": "handle-root", "localPosition": [0, 0, 0], "localRotation": [0, 0, 0], "purpose": "attachment root (= upper weld)"};
  node_handle_2.add(socket_handle_handle_root_0);
  sockets["handle:handle-root"] = socket_handle_handle_root_0;
  const socket_handle_grip_point_1 = new THREE.Object3D();
  socket_handle_grip_point_1.name = "grip-point";
  socket_handle_grip_point_1.position.set(0.2382, -0.2367, 0.1162);
  socket_handle_grip_point_1.rotation.set(0.0, 0.0, 0.0);
  socket_handle_grip_point_1.userData.socket = {"id": "grip-point", "localPosition": [0.2382, -0.2367, 0.1162], "localRotation": [0, 0, 0], "purpose": "hand IK target on the loop's outer side"};
  node_handle_2.add(socket_handle_grip_point_1);
  sockets["handle:grip-point"] = socket_handle_grip_point_1;

  const attachment_handle_wall_strip_3 = {"parentSocket": "handle-lower-socket", "childSocket": "strip-root", "localStart": [0.391, 0.0897, 0.1907], "localEnd": [0.391, 0.6367, 0.1907], "contactType": "weld", "baseRadius": 0.0245, "endRadius": 0.0245, "embedDepth": 0.0145, "overlap": 0.0145, "gapTolerance": 0.002};
  const endpoint_handle_wall_strip_3 = makeAttachmentEndpoint(attachment_handle_wall_strip_3);
  const node_handle_wall_strip_3 = new THREE.Group();
  node_handle_wall_strip_3.name = "Handle straight side welded on the wall__pivot";
  node_handle_wall_strip_3.scale.set(1, 1, 1);
  if (endpoint_handle_wall_strip_3) {
    node_handle_wall_strip_3.position.copy(endpoint_handle_wall_strip_3.start);
    node_handle_wall_strip_3.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_handle_wall_strip_3.position.set(0.0, 0.0, 0.0);
    node_handle_wall_strip_3.rotation.set(0.0, 0.0, 0.0);
  }
  node_handle_wall_strip_3.userData.sculptComponent = {"id": "handle-wall-strip", "name": "Handle straight side welded on the wall", "level": "meso", "role": "joint", "importance": 0.7, "confidence": 0.7, "primitive": "capsule", "topologyClass": "fiber-strand", "topologyRationale": "Straight side of the D-ring: a straight round strap lying on the wall between the welds -> segment between attachment endpoints.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "body-shell", "attachment": {"parentSocket": "handle-lower-socket", "childSocket": "strip-root", "localStart": [0.391, 0.0897, 0.1907], "localEnd": [0.391, 0.6367, 0.1907], "contactType": "weld", "baseRadius": 0.0245, "endRadius": 0.0245, "embedDepth": 0.0145, "overlap": 0.0145, "gapTolerance": 0.002}, "dimensions": {"width": 1, "height": 1, "depth": 1, "units": "scene (1 = 10 cm)", "confidence": 0.7}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-straight-side", "kind": "seam", "description": "strip on the wall at azimuth 64 deg, visible inside the body silhouette at 958-976 px", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["detail-inventory/handle.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(200, 206, 210, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceNote": "strip on the wall (958-976 px)"}};
  node_handle_wall_strip_3.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}};
  (nodes["body-shell"] ?? root).add(node_handle_wall_strip_3);
  nodes["handle-wall-strip"] = node_handle_wall_strip_3;
  const mesh_handle_wall_strip_3Geometry = endpoint_handle_wall_strip_3
    ? new THREE.CylinderGeometry(endpoint_handle_wall_strip_3.endRadius, endpoint_handle_wall_strip_3.baseRadius, endpoint_handle_wall_strip_3.length, 16, 6)
    : buildWatertightCapsule(0.35, 0.7, 8, 16, 1);
  if (!endpoint_handle_wall_strip_3) {
    mesh_handle_wall_strip_3Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_handle_wall_strip_3 = new THREE.Mesh(
    mesh_handle_wall_strip_3Geometry,
    materialMap["white-enamel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_handle_wall_strip_3.name = "Handle straight side welded on the wall";
  if (endpoint_handle_wall_strip_3) {
    mesh_handle_wall_strip_3.position.copy(endpoint_handle_wall_strip_3.midpoint);
    mesh_handle_wall_strip_3.quaternion.copy(endpoint_handle_wall_strip_3.quaternion);
  }
  mesh_handle_wall_strip_3.castShadow = options.castShadow ?? true;
  mesh_handle_wall_strip_3.receiveShadow = options.receiveShadow ?? true;
  mesh_handle_wall_strip_3.userData.sculptComponent = {"id": "handle-wall-strip", "name": "Handle straight side welded on the wall", "level": "meso", "role": "joint", "importance": 0.7, "confidence": 0.7, "primitive": "capsule", "topologyClass": "fiber-strand", "topologyRationale": "Straight side of the D-ring: a straight round strap lying on the wall between the welds -> segment between attachment endpoints.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "body-shell", "attachment": {"parentSocket": "handle-lower-socket", "childSocket": "strip-root", "localStart": [0.391, 0.0897, 0.1907], "localEnd": [0.391, 0.6367, 0.1907], "contactType": "weld", "baseRadius": 0.0245, "endRadius": 0.0245, "embedDepth": 0.0145, "overlap": 0.0145, "gapTolerance": 0.002}, "dimensions": {"width": 1, "height": 1, "depth": 1, "units": "scene (1 = 10 cm)", "confidence": 0.7}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-straight-side", "kind": "seam", "description": "strip on the wall at azimuth 64 deg, visible inside the body silhouette at 958-976 px", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["detail-inventory/handle.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(200, 206, 210, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceNote": "strip on the wall (958-976 px)"}};
  node_handle_wall_strip_3.add(mesh_handle_wall_strip_3);
  meshes["handle-wall-strip"] = mesh_handle_wall_strip_3;
  colliders["handle-wall-strip"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"};
  destructionGroups["handle"] ??= [];
  destructionGroups["handle"].push(node_handle_wall_strip_3);

  const endpoint_handle_upper_weld_pad_4 = makeAttachmentEndpoint(null);
  const node_handle_upper_weld_pad_4 = new THREE.Group();
  node_handle_upper_weld_pad_4.name = "Handle upper weld pad__pivot";
  node_handle_upper_weld_pad_4.scale.set(1, 1, 1);
  if (endpoint_handle_upper_weld_pad_4) {
    node_handle_upper_weld_pad_4.position.copy(endpoint_handle_upper_weld_pad_4.start);
    node_handle_upper_weld_pad_4.rotation.set(0.0, -0.4538, 0.0);
  } else {
    node_handle_upper_weld_pad_4.position.set(0.3874, 0.6367, 0.1889);
    node_handle_upper_weld_pad_4.rotation.set(0.0, -0.4538, 0.0);
  }
  node_handle_upper_weld_pad_4.userData.sculptComponent = {"id": "handle-upper-weld-pad", "name": "Handle upper weld pad", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.6, "primitive": "ellipsoid", "topologyClass": "assembled-solid", "topologyRationale": "Strap end pressed flat against the wall at the weld: a flattened ellipsoid pad straddling the wall surface.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decimate": {"targetRatio": 0.25}, "shadowCaster": false, "optimizationNote": "embedded inside the strap ends; 1216 -> ~300 tris, no shadow pass (optimization-pass)"}, "parent": "body-shell", "attachment": {"parentSocket": "handle-upper-socket", "childSocket": "handle-upper-weld-pad-root", "localStart": [0.3874, 0.6367, 0.1889], "localEnd": [0.3874, 0.6367, 0.1889], "contactType": "weld", "baseRadius": 0.03, "endRadius": 0.03, "embedDepth": 0.008, "overlap": 0.008, "gapTolerance": 0.002}, "dimensions": {"width": 0.026, "height": 0.05, "depth": 0.046, "units": "scene (1 = 10 cm)", "confidence": 0.6}, "transform": {"position": [0.3874, 0.6367, 0.1889], "rotation": [0, -0.4538, 0], "scale": [0.026, 0.05, 0.046]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-upper-weld-pad", "kind": "seam", "description": "flattened strap end on the wall (kept within the 0.0245 strap radius: a 0.03 half-width poked out as a knob in structural run 1)", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["detail-inventory/handle.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(200, 206, 210, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceNote": "same enamel as handle"}};
  node_handle_upper_weld_pad_4.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}};
  (nodes["body-shell"] ?? root).add(node_handle_upper_weld_pad_4);
  nodes["handle-upper-weld-pad"] = node_handle_upper_weld_pad_4;
  const mesh_handle_upper_weld_pad_4GeometryDense = endpoint_handle_upper_weld_pad_4
    ? new THREE.CylinderGeometry(endpoint_handle_upper_weld_pad_4.endRadius, endpoint_handle_upper_weld_pad_4.baseRadius, endpoint_handle_upper_weld_pad_4.length, 16, 6)
    : new THREE.SphereGeometry(0.5, 32, 20);
  const mesh_handle_upper_weld_pad_4Geometry = decimateGeometry(mesh_handle_upper_weld_pad_4GeometryDense, 0.25);
  if (!endpoint_handle_upper_weld_pad_4) {
    mesh_handle_upper_weld_pad_4Geometry.scale(0.026, 0.05, 0.046);
  }
  const mesh_handle_upper_weld_pad_4 = new THREE.Mesh(
    mesh_handle_upper_weld_pad_4Geometry,
    materialMap["white-enamel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_handle_upper_weld_pad_4.name = "Handle upper weld pad";
  if (endpoint_handle_upper_weld_pad_4) {
    mesh_handle_upper_weld_pad_4.position.copy(endpoint_handle_upper_weld_pad_4.midpoint);
    mesh_handle_upper_weld_pad_4.quaternion.copy(endpoint_handle_upper_weld_pad_4.quaternion);
  }
  mesh_handle_upper_weld_pad_4.castShadow = false; /* patched:tools/patch_factory.py */
  mesh_handle_upper_weld_pad_4.receiveShadow = options.receiveShadow ?? true;
  mesh_handle_upper_weld_pad_4.userData.sculptComponent = {"id": "handle-upper-weld-pad", "name": "Handle upper weld pad", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.6, "primitive": "ellipsoid", "topologyClass": "assembled-solid", "topologyRationale": "Strap end pressed flat against the wall at the weld: a flattened ellipsoid pad straddling the wall surface.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decimate": {"targetRatio": 0.25}, "shadowCaster": false, "optimizationNote": "embedded inside the strap ends; 1216 -> ~300 tris, no shadow pass (optimization-pass)"}, "parent": "body-shell", "attachment": {"parentSocket": "handle-upper-socket", "childSocket": "handle-upper-weld-pad-root", "localStart": [0.3874, 0.6367, 0.1889], "localEnd": [0.3874, 0.6367, 0.1889], "contactType": "weld", "baseRadius": 0.03, "endRadius": 0.03, "embedDepth": 0.008, "overlap": 0.008, "gapTolerance": 0.002}, "dimensions": {"width": 0.026, "height": 0.05, "depth": 0.046, "units": "scene (1 = 10 cm)", "confidence": 0.6}, "transform": {"position": [0.3874, 0.6367, 0.1889], "rotation": [0, -0.4538, 0], "scale": [0.026, 0.05, 0.046]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-upper-weld-pad", "kind": "seam", "description": "flattened strap end on the wall (kept within the 0.0245 strap radius: a 0.03 half-width poked out as a knob in structural run 1)", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["detail-inventory/handle.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(200, 206, 210, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceNote": "same enamel as handle"}};
  node_handle_upper_weld_pad_4.add(mesh_handle_upper_weld_pad_4);
  meshes["handle-upper-weld-pad"] = mesh_handle_upper_weld_pad_4;
  colliders["handle-upper-weld-pad"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"};
  destructionGroups["handle"] ??= [];
  destructionGroups["handle"].push(node_handle_upper_weld_pad_4);

  const endpoint_handle_lower_weld_pad_5 = makeAttachmentEndpoint(null);
  const node_handle_lower_weld_pad_5 = new THREE.Group();
  node_handle_lower_weld_pad_5.name = "Handle lower weld pad__pivot";
  node_handle_lower_weld_pad_5.scale.set(1, 1, 1);
  if (endpoint_handle_lower_weld_pad_5) {
    node_handle_lower_weld_pad_5.position.copy(endpoint_handle_lower_weld_pad_5.start);
    node_handle_lower_weld_pad_5.rotation.set(0.0, -0.4538, 0.0);
  } else {
    node_handle_lower_weld_pad_5.position.set(0.3874, 0.0897, 0.1889);
    node_handle_lower_weld_pad_5.rotation.set(0.0, -0.4538, 0.0);
  }
  node_handle_lower_weld_pad_5.userData.sculptComponent = {"id": "handle-lower-weld-pad", "name": "Handle lower weld pad", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.6, "primitive": "ellipsoid", "topologyClass": "assembled-solid", "topologyRationale": "Strap end pressed flat against the wall at the weld: a flattened ellipsoid pad straddling the wall surface.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decimate": {"targetRatio": 0.25}, "shadowCaster": false, "optimizationNote": "embedded inside the strap ends; 1216 -> ~300 tris, no shadow pass (optimization-pass)"}, "parent": "body-shell", "attachment": {"parentSocket": "handle-lower-socket", "childSocket": "handle-lower-weld-pad-root", "localStart": [0.3874, 0.0897, 0.1889], "localEnd": [0.3874, 0.0897, 0.1889], "contactType": "weld", "baseRadius": 0.03, "endRadius": 0.03, "embedDepth": 0.008, "overlap": 0.008, "gapTolerance": 0.002}, "dimensions": {"width": 0.026, "height": 0.05, "depth": 0.046, "units": "scene (1 = 10 cm)", "confidence": 0.6}, "transform": {"position": [0.3874, 0.0897, 0.1889], "rotation": [0, -0.4538, 0], "scale": [0.026, 0.05, 0.046]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-lower-weld-pad", "kind": "seam", "description": "flattened strap end on the wall (kept within the 0.0245 strap radius: a 0.03 half-width poked out as a knob in structural run 1)", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["detail-inventory/handle.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(200, 206, 210, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceNote": "same enamel as handle"}};
  node_handle_lower_weld_pad_5.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}};
  (nodes["body-shell"] ?? root).add(node_handle_lower_weld_pad_5);
  nodes["handle-lower-weld-pad"] = node_handle_lower_weld_pad_5;
  const mesh_handle_lower_weld_pad_5GeometryDense = endpoint_handle_lower_weld_pad_5
    ? new THREE.CylinderGeometry(endpoint_handle_lower_weld_pad_5.endRadius, endpoint_handle_lower_weld_pad_5.baseRadius, endpoint_handle_lower_weld_pad_5.length, 16, 6)
    : new THREE.SphereGeometry(0.5, 32, 20);
  const mesh_handle_lower_weld_pad_5Geometry = decimateGeometry(mesh_handle_lower_weld_pad_5GeometryDense, 0.25);
  if (!endpoint_handle_lower_weld_pad_5) {
    mesh_handle_lower_weld_pad_5Geometry.scale(0.026, 0.05, 0.046);
  }
  const mesh_handle_lower_weld_pad_5 = new THREE.Mesh(
    mesh_handle_lower_weld_pad_5Geometry,
    materialMap["white-enamel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_handle_lower_weld_pad_5.name = "Handle lower weld pad";
  if (endpoint_handle_lower_weld_pad_5) {
    mesh_handle_lower_weld_pad_5.position.copy(endpoint_handle_lower_weld_pad_5.midpoint);
    mesh_handle_lower_weld_pad_5.quaternion.copy(endpoint_handle_lower_weld_pad_5.quaternion);
  }
  mesh_handle_lower_weld_pad_5.castShadow = false; /* patched:tools/patch_factory.py */
  mesh_handle_lower_weld_pad_5.receiveShadow = options.receiveShadow ?? true;
  mesh_handle_lower_weld_pad_5.userData.sculptComponent = {"id": "handle-lower-weld-pad", "name": "Handle lower weld pad", "level": "meso", "role": "joint", "importance": 0.6, "confidence": 0.6, "primitive": "ellipsoid", "topologyClass": "assembled-solid", "topologyRationale": "Strap end pressed flat against the wall at the weld: a flattened ellipsoid pad straddling the wall surface.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decimate": {"targetRatio": 0.25}, "shadowCaster": false, "optimizationNote": "embedded inside the strap ends; 1216 -> ~300 tris, no shadow pass (optimization-pass)"}, "parent": "body-shell", "attachment": {"parentSocket": "handle-lower-socket", "childSocket": "handle-lower-weld-pad-root", "localStart": [0.3874, 0.0897, 0.1889], "localEnd": [0.3874, 0.0897, 0.1889], "contactType": "weld", "baseRadius": 0.03, "endRadius": 0.03, "embedDepth": 0.008, "overlap": 0.008, "gapTolerance": 0.002}, "dimensions": {"width": 0.026, "height": 0.05, "depth": 0.046, "units": "scene (1 = 10 cm)", "confidence": 0.6}, "transform": {"position": [0.3874, 0.0897, 0.1889], "rotation": [0, -0.4538, 0], "scale": [0.026, 0.05, 0.046]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "white-enamel"}}, "material": "white-enamel", "materialLayers": ["white-enamel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-lower-weld-pad", "kind": "seam", "description": "flattened strap end on the wall (kept within the 0.0245 strap radius: a 0.03 half-width poked out as a knob in structural run 1)", "effect": "geometry", "evidenceRef": "detail-inventory/handle.png"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "none: vitreous glass coat, orange-peel not resolvable at 259 px diameter (zone crops)", "displacementPattern": "none", "occlusionPattern": "cavity occlusion inside the cup; contact shadow under the base; shading band under the bead overhang (lighting-driven, not baked)", "edgeWearPattern": "single chip at the base edge (base-chip component); no rim wear visible", "notes": "tactile response is gloss only; relief would contradict the reference"}, "evidenceRefs": ["detail-inventory/handle.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(232, 236, 238, 1.0)", "secondaryAlbedo": "rgba(200, 206, 210, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.7, "evidenceNote": "same enamel as handle"}};
  node_handle_lower_weld_pad_5.add(mesh_handle_lower_weld_pad_5);
  meshes["handle-lower-weld-pad"] = mesh_handle_lower_weld_pad_5;
  colliders["handle-lower-weld-pad"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "part of handle collider"};
  destructionGroups["handle"] ??= [];
  destructionGroups["handle"].push(node_handle_lower_weld_pad_5);

  const attachment_decal_shell_6 = {"parentSocket": "decal-socket", "childSocket": "decal-root", "localStart": [0, 0.08, 0], "localEnd": [0, 0.76, 0], "contactType": "flush", "baseRadius": 0.42619999999999997, "endRadius": 0.42619999999999997, "embedDepth": 0.0, "overlap": 0.0012, "gapTolerance": 0.0015, "notes": "Print shell sits 0.0012 proud of the wall (flush decal); localStart/localEnd span its height on the mug axis. Not a sweep: tools/patch_factory.py builds the open cylinder segment."};
  const endpoint_decal_shell_6 = makeAttachmentEndpoint(attachment_decal_shell_6);
  const node_decal_shell_6 = new THREE.Group();
  node_decal_shell_6.name = "Printed decal carrier (skull, lettering, sunburst)__pivot";
  node_decal_shell_6.scale.set(1, 1, 1);
  if (endpoint_decal_shell_6) {
    node_decal_shell_6.position.copy(endpoint_decal_shell_6.start);
    node_decal_shell_6.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_decal_shell_6.position.set(0.0, 0.42, 0.0);
    node_decal_shell_6.rotation.set(0.0, 0.0, 0.0);
  }
  node_decal_shell_6.userData.sculptComponent = {"id": "decal-shell", "name": "Printed decal carrier (skull, lettering, sunburst)", "level": "micro", "role": "decal", "importance": 0.95, "confidence": 0.85, "primitive": "cylinder", "topologyClass": "material-only", "topologyRationale": "Fired-on print with no relief: a thin open cylinder segment 0.0012 proud of the wall carries the projected ink alpha.", "geometryDescriptor": {"topologyIntent": "open cylinder segment, theta -120..60 deg, y 0.08..0.76", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "cylindrical (u = azimuth, v = height)", "normalStrategy": "smooth", "decalProjection": {"texture": "projection/decal_uv.png", "module": "src/decalTexture.ts", "baker": "tools/bake_decal.py", "mode": "orthographic-front-projection", "thetaStartDeg": -120, "thetaLengthDeg": 180, "y0": 0.08, "y1": 0.76, "radius": 0.42619999999999997}}, "parent": "body-shell", "attachment": {"parentSocket": "decal-socket", "childSocket": "decal-root", "localStart": [0, 0.08, 0], "localEnd": [0, 0.76, 0], "contactType": "flush", "baseRadius": 0.42619999999999997, "endRadius": 0.42619999999999997, "embedDepth": 0.0, "overlap": 0.0012, "gapTolerance": 0.0015, "notes": "Print shell sits 0.0012 proud of the wall (flush decal); localStart/localEnd span its height on the mug axis. Not a sweep: tools/patch_factory.py builds the open cylinder segment."}, "dimensions": {"width": 0.853, "height": 0.68, "depth": 0.853, "units": "scene (1 = 10 cm)", "confidence": 0.85}, "transform": {"position": [0, 0.42, 0], "rotation": [0, 0, 0], "scale": [0.8524, 0.68, 0.8524]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "uses body collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "decal-ink"}}, "material": "decal-ink", "materialLayers": ["decal-ink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "skull-decal-print", "kind": "decal", "description": "skull profile + knocked-out lettering projected from reference pixels, azimuth -29..31 deg", "effect": "material", "evidenceRef": "detail-inventory/decal.png"}, {"id": "sunburst-decal", "kind": "decal", "description": "radial sunburst centred at azimuth -76 deg, y=0.38 (redrawn, low confidence)", "effect": "material", "evidenceRef": "detail-inventory/body-left.png"}], "surfaceDetail": {"macroRoughness": 0.32, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: fired print is flush (grazing look-dev shows no relief, matching the reference)", "displacementPattern": "none", "occlusionPattern": "none", "edgeWearPattern": "none", "notes": "print boundary is an alpha mask, not relief"}, "evidenceRefs": ["detail-inventory/decal.png", "projection/decal_uv.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(20, 20, 22, 1.0)", "secondaryAlbedo": "rgba(232, 236, 238, 0.0)", "materialClass": "ceramic", "materialClassConfidence": 0.8, "evidenceNote": "ink core luma 25-40; knock-out = transparent"}};
  node_decal_shell_6.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "uses body collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "decal-ink"}};
  (nodes["body-shell"] ?? root).add(node_decal_shell_6);
  nodes["decal-shell"] = node_decal_shell_6;
  const mesh_decal_shell_6Geometry = /* patched:tools/patch_factory.py */ new THREE.CylinderGeometry(0.42619999999999997, 0.42619999999999997, 0.6800, 160, 1, true, -2.094395, 3.141593);
  mesh_decal_shell_6Geometry.translate(0, 0.4200, 0);
  if (!endpoint_decal_shell_6) {
    mesh_decal_shell_6Geometry.scale(0.8524, 0.68, 0.8524);
  }
  const mesh_decal_shell_6 = new THREE.Mesh(
    mesh_decal_shell_6Geometry,
    decalInkMaterial(materialMap["decal-ink"]) /* patched:tools/patch_factory.py */
  );
  mesh_decal_shell_6.name = "Printed decal carrier (skull, lettering, sunburst)";
  if (endpoint_decal_shell_6) {
    mesh_decal_shell_6.position.copy(endpoint_decal_shell_6.start).multiplyScalar(-1); /* patched:tools/patch_factory.py */
  }
  mesh_decal_shell_6.castShadow = false; mesh_decal_shell_6.renderOrder = 1; mesh_decal_shell_6.userData.explodeWithParent = true; /* patched:tools/patch_factory.py */
  mesh_decal_shell_6.receiveShadow = options.receiveShadow ?? true;
  mesh_decal_shell_6.userData.sculptComponent = {"id": "decal-shell", "name": "Printed decal carrier (skull, lettering, sunburst)", "level": "micro", "role": "decal", "importance": 0.95, "confidence": 0.85, "primitive": "cylinder", "topologyClass": "material-only", "topologyRationale": "Fired-on print with no relief: a thin open cylinder segment 0.0012 proud of the wall carries the projected ink alpha.", "geometryDescriptor": {"topologyIntent": "open cylinder segment, theta -120..60 deg, y 0.08..0.76", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "cylindrical (u = azimuth, v = height)", "normalStrategy": "smooth", "decalProjection": {"texture": "projection/decal_uv.png", "module": "src/decalTexture.ts", "baker": "tools/bake_decal.py", "mode": "orthographic-front-projection", "thetaStartDeg": -120, "thetaLengthDeg": 180, "y0": 0.08, "y1": 0.76, "radius": 0.42619999999999997}}, "parent": "body-shell", "attachment": {"parentSocket": "decal-socket", "childSocket": "decal-root", "localStart": [0, 0.08, 0], "localEnd": [0, 0.76, 0], "contactType": "flush", "baseRadius": 0.42619999999999997, "endRadius": 0.42619999999999997, "embedDepth": 0.0, "overlap": 0.0012, "gapTolerance": 0.0015, "notes": "Print shell sits 0.0012 proud of the wall (flush decal); localStart/localEnd span its height on the mug axis. Not a sweep: tools/patch_factory.py builds the open cylinder segment."}, "dimensions": {"width": 0.853, "height": 0.68, "depth": 0.853, "units": "scene (1 = 10 cm)", "confidence": 0.85}, "transform": {"position": [0, 0.42, 0], "rotation": [0, 0, 0], "scale": [0.8524, 0.68, 0.8524]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "uses body collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "decal-ink"}}, "material": "decal-ink", "materialLayers": ["decal-ink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "skull-decal-print", "kind": "decal", "description": "skull profile + knocked-out lettering projected from reference pixels, azimuth -29..31 deg", "effect": "material", "evidenceRef": "detail-inventory/decal.png"}, {"id": "sunburst-decal", "kind": "decal", "description": "radial sunburst centred at azimuth -76 deg, y=0.38 (redrawn, low confidence)", "effect": "material", "evidenceRef": "detail-inventory/body-left.png"}], "surfaceDetail": {"macroRoughness": 0.32, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: fired print is flush (grazing look-dev shows no relief, matching the reference)", "displacementPattern": "none", "occlusionPattern": "none", "edgeWearPattern": "none", "notes": "print boundary is an alpha mask, not relief"}, "evidenceRefs": ["detail-inventory/decal.png", "projection/decal_uv.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(20, 20, 22, 1.0)", "secondaryAlbedo": "rgba(232, 236, 238, 0.0)", "materialClass": "ceramic", "materialClassConfidence": 0.8, "evidenceNote": "ink core luma 25-40; knock-out = transparent"}};
  node_decal_shell_6.add(mesh_decal_shell_6);
  meshes["decal-shell"] = mesh_decal_shell_6;
  colliders["decal-shell"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "uses body collider"};
  destructionGroups["body-shell"] ??= [];
  destructionGroups["body-shell"].push(node_decal_shell_6);

  const endpoint_base_chip_7 = makeAttachmentEndpoint(null);
  const node_base_chip_7 = new THREE.Group();
  node_base_chip_7.name = "Enamel chip at base edge (exposed steel)__pivot";
  node_base_chip_7.scale.set(1, 1, 1);
  if (endpoint_base_chip_7) {
    node_base_chip_7.position.copy(endpoint_base_chip_7.start);
    node_base_chip_7.rotation.set(0.0, -0.7505, -0.06);
  } else {
    node_base_chip_7.position.set(-0.2886, 0.0075, 0.3095);
    node_base_chip_7.rotation.set(0.0, -0.7505, -0.06);
  }
  node_base_chip_7.userData.sculptComponent = {"id": "base-chip", "name": "Enamel chip at base edge (exposed steel)", "level": "micro", "role": "wear", "importance": 0.4, "confidence": 0.6, "primitive": "ellipsoid", "topologyClass": "surface-relief", "topologyRationale": "Small missing-enamel patch on the base corner; a thin dark plate conforming to the wall tangent.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decimate": {"targetRatio": 0.25}, "shadowCaster": false, "optimizationNote": "20 px sliver; 1216 -> ~300 tris, no shadow pass (optimization-pass)"}, "parent": "body-shell", "attachment": {"parentSocket": "decal-socket", "childSocket": "chip-root", "localStart": [-0.2886, 0.0075, 0.3095], "localEnd": [-0.2886, 0.0075, 0.3095], "contactType": "flush", "baseRadius": 0.0, "endRadius": 0.0, "embedDepth": 0.0015, "overlap": 0.0015, "gapTolerance": 0.002}, "dimensions": {"width": 0.11, "height": 0.008, "depth": 0.006, "units": "scene (1 = 10 cm)", "confidence": 0.6}, "transform": {"position": [-0.2886, 0.0075, 0.3095], "rotation": [0, -0.7505, -0.06], "scale": [0.11, 0.008, 0.006]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "none"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "exposed-steel"}}, "material": "exposed-steel", "materialLayers": ["exposed-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "base-edge-chip-geo", "kind": "chip", "description": "thin dark sliver ~30 px long lying ON the base edge (y 0.002-0.013), slightly tilted; ellipsoid flattened against the fillet (form run 1: a box 5 px above the edge read as a floating tag)", "effect": "geometry", "evidenceRef": "detail-inventory/base.png"}], "surfaceDetail": {"macroRoughness": 0.55, "microRoughness": 0.2, "bumpAmplitude": 0.0, "normalPattern": "none at 20 px", "displacementPattern": "flattened ellipsoid sliver", "occlusionPattern": "none", "edgeWearPattern": "exposed steel at the chip", "notes": ""}, "evidenceRefs": ["detail-inventory/base.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(38, 36, 34, 1.0)", "secondaryAlbedo": "rgba(58, 47, 40, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.6, "evidenceNote": "chip pixels at base (745-775, 500-506)"}};
  node_base_chip_7.userData.actionProfile = {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "none"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "exposed-steel"}};
  (nodes["body-shell"] ?? root).add(node_base_chip_7);
  nodes["base-chip"] = node_base_chip_7;
  const mesh_base_chip_7GeometryDense = endpoint_base_chip_7
    ? new THREE.CylinderGeometry(endpoint_base_chip_7.endRadius, endpoint_base_chip_7.baseRadius, endpoint_base_chip_7.length, 16, 6)
    : new THREE.SphereGeometry(0.5, 32, 20);
  const mesh_base_chip_7Geometry = decimateGeometry(mesh_base_chip_7GeometryDense, 0.25);
  if (!endpoint_base_chip_7) {
    mesh_base_chip_7Geometry.scale(0.11, 0.008, 0.006);
  }
  const mesh_base_chip_7 = new THREE.Mesh(
    mesh_base_chip_7Geometry,
    materialMap["exposed-steel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_base_chip_7.name = "Enamel chip at base edge (exposed steel)";
  mesh_base_chip_7.userData.explodeWithParent = true; /* patched:tools/patch_factory.py */
  if (endpoint_base_chip_7) {
    mesh_base_chip_7.position.copy(endpoint_base_chip_7.midpoint);
    mesh_base_chip_7.quaternion.copy(endpoint_base_chip_7.quaternion);
  }
  mesh_base_chip_7.castShadow = false; /* patched:tools/patch_factory.py */
  mesh_base_chip_7.receiveShadow = options.receiveShadow ?? true;
  mesh_base_chip_7.userData.sculptComponent = {"id": "base-chip", "name": "Enamel chip at base edge (exposed steel)", "level": "micro", "role": "wear", "importance": 0.4, "confidence": 0.6, "primitive": "ellipsoid", "topologyClass": "surface-relief", "topologyRationale": "Small missing-enamel patch on the base corner; a thin dark plate conforming to the wall tangent.", "geometryDescriptor": {"topologyIntent": "", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decimate": {"targetRatio": 0.25}, "shadowCaster": false, "optimizationNote": "20 px sliver; 1216 -> ~300 tris, no shadow pass (optimization-pass)"}, "parent": "body-shell", "attachment": {"parentSocket": "decal-socket", "childSocket": "chip-root", "localStart": [-0.2886, 0.0075, 0.3095], "localEnd": [-0.2886, 0.0075, 0.3095], "contactType": "flush", "baseRadius": 0.0, "endRadius": 0.0, "embedDepth": 0.0015, "overlap": 0.0015, "gapTolerance": 0.002}, "dimensions": {"width": 0.11, "height": 0.008, "depth": 0.006, "units": "scene (1 = 10 cm)", "confidence": 0.6}, "transform": {"position": [-0.2886, 0.0075, 0.3095], "rotation": [0, -0.7505, -0.06], "scale": [0.11, 0.008, 0.006]}, "actionProfile": {"animationRole": "static", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "none"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "body-shell", "seamRefs": ["handle-upper-weld", "handle-lower-weld"], "detachableFragments": ["handle", "rim-bead"], "breakImpulse": 40.0, "debrisMaterial": "exposed-steel"}}, "material": "exposed-steel", "materialLayers": ["exposed-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "base-edge-chip-geo", "kind": "chip", "description": "thin dark sliver ~30 px long lying ON the base edge (y 0.002-0.013), slightly tilted; ellipsoid flattened against the fillet (form run 1: a box 5 px above the edge read as a floating tag)", "effect": "geometry", "evidenceRef": "detail-inventory/base.png"}], "surfaceDetail": {"macroRoughness": 0.55, "microRoughness": 0.2, "bumpAmplitude": 0.0, "normalPattern": "none at 20 px", "displacementPattern": "flattened ellipsoid sliver", "occlusionPattern": "none", "edgeWearPattern": "exposed steel at the chip", "notes": ""}, "evidenceRefs": ["detail-inventory/base.png"], "fidelityTier": "reference-fidelity", "colorMaterialRecipe": {"dominantAlbedo": "rgba(38, 36, 34, 1.0)", "secondaryAlbedo": "rgba(58, 47, 40, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.6, "evidenceNote": "chip pixels at base (745-775, 500-506)"}};
  node_base_chip_7.add(mesh_base_chip_7);
  meshes["base-chip"] = mesh_base_chip_7;
  colliders["base-chip"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "none"};
  destructionGroups["body-shell"] ??= [];
  destructionGroups["body-shell"].push(node_base_chip_7);

  root.userData.sculptRuntime = { nodes, meshes, sockets, colliders, destructionGroups } satisfies ProceduralModelRuntime;
  root.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  root.userData.actionReadiness = {
    note: 'Use root.userData.sculptRuntime.nodes for transforms, sockets for attachments, colliders for physics proxies, and destructionGroups for breakable sets.',
  };
  attachInteraction(root); /* patched:tools/patch_factory.py */
  return root;
}

export function createEnamelMugLookDevLights(
  mode: 'neutral' | 'grazing' | 'reference' = 'neutral',
): THREE.Group {
  const lights = new THREE.Group();
  lights.name = "Enamel mug look-dev lights";
  const hemi = new THREE.HemisphereLight(
    mode === 'reference' ? 0xfff0d6 : 0xf2f4ff,
    0x363b42,
    mode === 'grazing' ? 0.28 : mode === 'reference' ? 0.72 : 0.85,
  );
  lights.add(hemi);
  const key = new THREE.DirectionalLight(
    mode === 'reference' ? 0xffcf8a : 0xfff4e8,
    mode === 'grazing' ? 4.2 : mode === 'reference' ? 2.6 : 2.15,
  );
  if (mode === 'grazing') key.position.set(7.5, 1.1, 4.0);
  else if (mode === 'reference') key.position.set(-4.5, 7.5, 5.0);
  else key.position.set(-4.0, 6.0, 5.5);
  key.castShadow = true;
  key.shadow.mapSize.set(4096, 4096);
  key.shadow.bias = -0.00025;
  key.shadow.normalBias = 0.018;
  key.shadow.radius = 7;
  key.shadow.blurSamples = 24;
  key.shadow.camera.near = 0.5;
  key.shadow.camera.far = 30;
  key.shadow.camera.left = -2.6;
  key.shadow.camera.right = 2.6;
  key.shadow.camera.top = 2.6;
  key.shadow.camera.bottom = -2.6;
  key.shadow.camera.updateProjectionMatrix();
  lights.add(key);
  const fill = new THREE.DirectionalLight(0xa8c4ff, mode === 'grazing' ? 0.12 : 0.42);
  fill.position.set(4.0, 3.0, 3.5);
  lights.add(fill);
  const rim = new THREE.DirectionalLight(0xfff1c4, mode === 'grazing' ? 0.28 : 0.85);
  rim.position.set(0.5, 4.5, -6.0);
  lights.add(rim);
  lights.userData.reviewMode = mode;
  lights.userData.lightingFromPhoto = [{"id": "key", "role": "key light", "type": "directional", "position": [-5.0, 3.4, 3.2], "intensity": 2.6, "color": "#f7f9fd", "castShadow": true, "evidence": "large soft source front-left/above: left half of the wall brightest (228 luma), right limb shaded (169); lighting-pass iterations l0-l4 in renders/lighting-iter (tone_probe)"}, {"id": "fill", "role": "fill light", "type": "hemisphere", "skyColor": "#cddbea", "groundColor": "#4a3c34", "intensity": 0.5, "evidence": "shaded side never falls below ~130 luma; warm bounce from the wooden table"}, {"id": "rim", "role": "rim light", "type": "directional", "position": [3.5, 2.5, -4.0], "intensity": 0.5, "color": "#e8eef5", "castShadow": false, "evidence": "thin bright edge on the handle outer side and bead top"}, {"id": "environment", "role": "environment light", "environmentIntensity": 0.3, "evidence": "soft room reflections; intensity 0.3 so the right limb falls to ~170 like the reference (0.45 left it at 198)"}, {"id": "tone", "role": "exposure / tone mapping", "exposure": 1.2, "toneMapping": "ACESFilmic"}, {"id": "ground", "role": "background + contact shadow", "tableColor": "#76645e", "tableRoughness": 0.55, "backgroundColor": "#8c8375", "noTable": false, "evidence": "table (148,136,131) front, background wall (145,128,106); soft contact shadow under the mug from the key's shadow map"}];
  lights.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  return lights;
}

// PBR materials (clearcoat/iridescence/transmission/anisotropy) need an environment
// map to visually behave as intended — call this once per renderer and assign the
// result to scene.environment before rendering. No external HDR asset required.
export function createEnamelMugEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  return texture;
}

// Plan 1.3 §3.2 — auto-framing by bounding box. The Divine Eye can only compare a
// render to the reference if the object is FRAMED consistently (an object framed
// differently scores as wrong even when its shape is right). This positions the camera
// deterministically from the object's bounding box so it fills the frame at a stable
// margin, and sets near/far to the object scale. Call after adding the model to the
// scene, and again on resize (after updating camera.aspect).
export function frameEnamelMugCamera(
  camera: THREE.PerspectiveCamera,
  object: THREE.Object3D,
  options: { margin?: number; azimuthDeg?: number; elevationDeg?: number } = {},
): void {
  const box = new THREE.Box3().setFromObject(object);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const margin = options.margin ?? 1.15;
  const maxDim = Math.max(size.x, size.y, size.z) * margin;
  const fov = (camera.fov * Math.PI) / 180;
  // distance so the largest object dimension fits vertically in the frame
  const distance = (maxDim / 2) / Math.tan(fov / 2);
  const az = ((options.azimuthDeg ?? 0) * Math.PI) / 180;
  const el = ((options.elevationDeg ?? 0) * Math.PI) / 180;
  const dir = new THREE.Vector3(
    Math.sin(az) * Math.cos(el),
    Math.sin(el),
    Math.cos(az) * Math.cos(el),
  );
  camera.position.copy(center).addScaledVector(dir, distance);
  camera.near = Math.max(0.01, distance - maxDim);
  camera.far = distance + maxDim * 2;
  camera.lookAt(center);
  camera.updateProjectionMatrix();
}

// Plan 1.3 §3.2c — PRESENTATION composer (DOF + bloom). CRITICAL (R-POSTFX): this is
// for the showcase/hero render ONLY. The Divine Eye's EVALUATION render MUST use a
// plain renderer with NO composer — bloom blows highlights and DOF blurs edges, which
// would corrupt the deterministic IoU/DCD/edge/blowout signals. Enable dof/bloom ONLY
// when the reference photo actually exhibits them (detect_reference_effects.py authorizes).
export function createEnamelMugPresentationComposer(
  renderer: THREE.WebGLRenderer,
  scene: THREE.Scene,
  camera: THREE.Camera,
  options: { dof?: boolean; bloom?: boolean; bloomStrength?: number; dofFocus?: number; dofAperture?: number } = {},
): EffectComposer {
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  if (options.dof) {
    composer.addPass(new BokehPass(scene, camera, {
      focus: options.dofFocus ?? 10.0,
      aperture: options.dofAperture ?? 0.0002,
      maxblur: 0.01,
    }));
  }
  if (options.bloom) {
    const size = new THREE.Vector2();
    renderer.getSize(size);
    composer.addPass(new UnrealBloomPass(size, options.bloomStrength ?? 0.4, 0.4, 0.85));
  }
  return composer;
}

export function configureEnamelMugRenderer(renderer: THREE.WebGLRenderer): void {
  // Load-bearing for view-dependent finishes (anodized / Doppler): without ACES + sRGB
  // the environment reflection reads flat/washed instead of a believable metal response.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
}

export function createEnamelMugInspectControls(
  camera: THREE.Camera,
  domElement: HTMLElement,
): OrbitControls {
  // View-dependent finishes only read correctly once the user orbits — their color
  // comes from the environment reflection, not albedo, so free rotation matters here.
  const controls = new OrbitControls(camera, domElement);
  controls.enableDamping = true;
  controls.minDistance = 1.0;
  controls.maxDistance = 8.0;
  controls.autoRotate = false;
  return controls;
}
