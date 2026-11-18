import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { BokehPass } from 'three/examples/jsm/postprocessing/BokehPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { DecalGeometry } from 'three/examples/jsm/geometries/DecalGeometry.js';

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
  parts?: string[];
  partOf?: (object: THREE.Object3D | null | undefined) => string | null;
  pick?: (raycaster: THREE.Raycaster) => string | null;
  explode?: (amount: number) => Record<string, [number, number, number]>;
  manifest?: () => PartManifest;
};

export type PartManifest = {
  model: string;
  parts: { name: string; label: string; kind: 'part'; module: string; triangles: number }[];
  unnamedMeshes: number;
  integralMeshes: number;
};

// Parts = components whose mesh is not surface detail (`userData.explodeWithParent`). The root
// component stays put in explode. Separation SCALES each part's offset from the model centre
// (plus a base clearance so near-centre parts still open), never a uniform push.
function attachPartRuntime(root: THREE.Object3D, rt: ProceduralModelRuntime): ProceduralModelRuntime {
  const isDetail = (m: THREE.Object3D | undefined) => m?.userData?.explodeWithParent === true;
  const rootId = Object.keys(rt.nodes).find((id) => rt.nodes[id].parent === root) ?? null;
  const parts = Object.keys(rt.meshes).filter((id) => !isDetail(rt.meshes[id]));
  const partOf = (object: THREE.Object3D | null | undefined): string | null => {
    let o: THREE.Object3D | null | undefined = object;
    while (o) {
      const id = o.userData?.sculptComponent?.id as string | undefined;
      if (id && rt.meshes[id] === o && !isDetail(o)) return id;
      if (id && rt.nodes[id] === o && rt.meshes[id] && !isDetail(rt.meshes[id])) return id;
      o = o.parent;
    }
    return null;
  };
  const pick = (raycaster: THREE.Raycaster): string | null => {
    for (const hit of raycaster.intersectObject(root, true)) {
      if ((hit.object as THREE.Mesh).isMesh && hit.object.visible) return partOf(hit.object);
    }
    return null;
  };
  const base = new Map<string, THREE.Vector3>();
  const explode = (amount: number): Record<string, [number, number, number]> => {
    root.updateMatrixWorld(true);
    const toRoot = new THREE.Matrix4().copy(root.matrixWorld).invert();
    const centreOf = (o: THREE.Object3D) =>
      new THREE.Box3().setFromObject(o).getCenter(new THREE.Vector3()).applyMatrix4(toRoot);
    const modelBox = new THREE.Box3().setFromObject(root);
    const modelCentre = modelBox.getCenter(new THREE.Vector3()).applyMatrix4(toRoot);
    const clearance = modelBox.getSize(new THREE.Vector3()).length() * 0.06;
    const moved: Record<string, [number, number, number]> = {};
    for (const id of parts) {
      const node = rt.nodes[id];
      if (!node || id === rootId) continue;
      if (!base.has(id)) base.set(id, node.position.clone());
      node.position.copy(base.get(id)!);
      if (amount <= 0) continue;
      root.updateMatrixWorld(true);
      const away = centreOf(rt.meshes[id]).sub(modelCentre);
      const dir = away.lengthSq() > 1e-12 ? away.clone().normalize() : new THREE.Vector3(0, 1, 0);
      const offset = away.multiplyScalar(amount).add(dir.multiplyScalar(clearance * amount));
      // offset is in root space; convert to the node's parent space before applying
      const parent = node.parent ?? root;
      const parentFromRoot = new THREE.Matrix4().copy(parent.matrixWorld).invert().multiply(root.matrixWorld);
      const local = offset.clone().transformDirection(parentFromRoot).multiplyScalar(offset.length());
      node.position.add(local);
      moved[id] = [local.x, local.y, local.z];
    }
    root.updateMatrixWorld(true);
    return moved;
  };
  const manifest = (): PartManifest => {
    const out: PartManifest = { model: root.name, parts: [], unnamedMeshes: 0, integralMeshes: 0 };
    root.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const id = o.userData?.sculptComponent?.id as string | undefined;
      if (!o.name || !id) { out.unnamedMeshes++; return; }
      const g = mesh.geometry;
      const triangles = Math.round(g.index ? g.index.count / 3 : g.attributes.position.count / 3);
      out.parts.push({ name: id, label: o.name, kind: 'part', module: partOf(o) ?? id, triangles });
    });
    return out;
  };
  return { ...rt, parts, partOf, pick, explode, manifest };
}

type SculptMaterialSpec = Record<string, any>;

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

type TaperedStation = { position: [number, number, number]; rx: number; rz: number; twist?: number };

// Frames come from PARALLEL TRANSPORT, not from a Frenet frame. A Frenet frame is defined by
// the curve's normal, which flips sign wherever the path has an inflection or straightens out,
// and every flip twists the surface 180 degrees within one segment. Carrying the previous frame
// forward and removing only its along-path component keeps the twist continuous. THREE's own
// extrudePath and TubeGeometry do not expose this, which is why this is hand-built.
function buildTaperedSweepGeometry(
  sweep: { stations: TaperedStation[]; radialSegments?: number; capEnds?: boolean },
): THREE.BufferGeometry {
  const stations = sweep.stations;
  if (stations.length < 2) throw new Error('tapered-sweep needs at least two stations');
  const radial = Math.max(3, sweep.radialSegments ?? 10);
  const centres = stations.map((s) => new THREE.Vector3(...s.position));

  const tangents = centres.map((_, i) => {
    const prev = centres[Math.max(0, i - 1)];
    const next = centres[Math.min(centres.length - 1, i + 1)];
    const t = next.clone().sub(prev);
    // Coincident neighbours would normalise to NaN and poison every downstream vertex.
    return t.lengthSq() < 1e-12 ? new THREE.Vector3(0, 1, 0) : t.normalize();
  });

  // Seed a reference axis that is not parallel to the first tangent, or the first cross
  // product is degenerate and the whole sweep collapses to a line.
  let ref = new THREE.Vector3(0, 0, 1);
  if (Math.abs(tangents[0].dot(ref)) > 0.9) ref = new THREE.Vector3(1, 0, 0);

  const normals: THREE.Vector3[] = [];
  const binormals: THREE.Vector3[] = [];
  let carried = ref.clone().sub(tangents[0].clone().multiplyScalar(ref.dot(tangents[0]))).normalize();
  for (let i = 0; i < tangents.length; i += 1) {
    const t = tangents[i];
    // Project the carried frame back onto the plane perpendicular to this tangent.
    const n = carried.clone().sub(t.clone().multiplyScalar(carried.dot(t)));
    if (n.lengthSq() < 1e-12) {
      const fallback = Math.abs(t.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
      n.copy(fallback.sub(t.clone().multiplyScalar(fallback.dot(t))));
    }
    n.normalize();
    normals.push(n);
    binormals.push(new THREE.Vector3().crossVectors(t, n).normalize());
    carried = n;
  }

  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];
  const ringStart: number[] = [];
  const isPoint: boolean[] = [];

  for (let i = 0; i < stations.length; i += 1) {
    const st = stations[i];
    const v = i / (stations.length - 1);
    ringStart.push(positions.length / 3);
    // A station whose section has collapsed emits ONE vertex, not a ring of radius zero.
    // A degenerate ring still carries `radial` coincident vertices and `radial` zero-area
    // triangles, so the lock ends in a blunt cap the width of the floating-point noise
    // rather than at a point -- and a hair lock, a horn or a blade tip has to reach a point.
    if (st.rx <= 1e-6 && st.rz <= 1e-6) {
      isPoint.push(true);
      positions.push(centres[i].x, centres[i].y, centres[i].z);
      uvs.push(0.5, v);
      continue;
    }
    isPoint.push(false);
    const twist = ((st.twist ?? 0) * Math.PI) / 180;
    for (let j = 0; j <= radial; j += 1) {
      const theta = (j / radial) * Math.PI * 2 + twist;
      const offset = normals[i].clone().multiplyScalar(Math.cos(theta) * st.rx)
        .add(binormals[i].clone().multiplyScalar(Math.sin(theta) * st.rz));
      const p = centres[i].clone().add(offset);
      positions.push(p.x, p.y, p.z);
      uvs.push(j / radial, v);
    }
  }

  for (let i = 0; i < stations.length - 1; i += 1) {
    const a0 = ringStart[i];
    const b0 = ringStart[i + 1];
    if (isPoint[i] && isPoint[i + 1]) continue;   // two collapsed stations bound nothing
    for (let j = 0; j < radial; j += 1) {
      // Wound so the face normal points radially OUTWARD.
      //
      // Ring vertices advance from `normal` toward `binormal`, and binormal is
      // tangent x normal, so increasing theta runs counter-clockwise seen from the
      // far end of the segment. Taking the ring-to-ring edge first therefore puts
      // the cross product on the inside. Measured as signed volume on the built
      // mesh: every tapered-sweep came out negative -- a torso at -0.0674 and a
      // tail at -0.0044 against a positive ellipsoid head -- so every sweep this
      // generator has ever emitted rendered its back faces, with normals pointing
      // into the solid and every lighting judgement made on the wrong surface.
      if (isPoint[i]) indices.push(a0, b0 + j + 1, b0 + j);
      else if (isPoint[i + 1]) indices.push(a0 + j, a0 + j + 1, b0);
      else indices.push(a0 + j, a0 + j + 1, b0 + j, a0 + j + 1, b0 + j + 1, b0 + j);
    }
  }

  if (sweep.capEnds ?? true) {
    for (const end of [0, stations.length - 1]) {
      if (isPoint[end]) continue;   // a point end is already closed
      const centreIndex = positions.length / 3;
      positions.push(centres[end].x, centres[end].y, centres[end].z);
      uvs.push(0.5, end === 0 ? 0 : 1);
      const base = ringStart[end];
      for (let j = 0; j < radial; j += 1) {
        if (end === 0) indices.push(centreIndex, base + j + 1, base + j);
        else indices.push(centreIndex, base + j, base + j + 1);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
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

// Generated from ObjectSculptSpec target: Porcelain teapot
// Sculpt build pass: optimization-pass
// This factory is intentionally pass-gated. Finish browser screenshot review before unlocking deeper passes.
export function createPorcelainTeapotModel(options: ProceduralModelOptions = {}): THREE.Group {
  const root = new THREE.Group();
  root.name = "Porcelain teapot";
  root.userData.reconstructionEvidence = {"itemFamily": null, "subtype": null, "componentAdapter": null, "route": null, "exactnessTier": null, "referenceCamera": {"solved": true, "fovDegrees": 15, "distance": 10.3, "target": [-0.114, 0.77, 0], "orientation": {"yaw": 0, "pitch": 15, "roll": 0}, "aspect": 1.4, "referenceFrame": "evidence/ref_840.png (reference.jpg x 43..1237 scaled to 840x600)", "method": "forward silhouette fit (scripts/fit_shape.py), pitch fixed by rim/foot ellipse openness", "silhouetteIoU": 0.964}, "approximationNotes": []};
  root.userData.materialPipeline = {"schemaVersion": 1, "status": "proceed", "registry": "/mnt/c/repos/image2threejs/docs/materials/material-reference.json", "analysisArtifact": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-analysis.json", "targetThreshold": 0.7, "unresolvedNotObservedMaterials": [], "regions": [{"componentId": "body", "regionId": "shoulder-harlequin-band", "specMaterialId": "painted-body", "profileId": "ceramic.glazed", "status": "proceed"}, {"componentId": "spout", "regionId": "white-glaze", "specMaterialId": "porcelain-glaze", "profileId": "ceramic.glazed", "status": "proceed"}, {"componentId": "knob-gold-cap", "regionId": "gold-lustre", "specMaterialId": "gold-lustre", "profileId": "metal.gold", "status": "proceed"}, {"componentId": "lid", "regionId": "lid-chrysanthemum", "specMaterialId": "painted-lid", "profileId": "ceramic.glazed", "status": "proceed"}, {"componentId": "spout-flower-decal", "regionId": "spout-flower", "specMaterialId": "enamel-decal", "profileId": "ceramic.glazed", "status": "proceed"}], "controlledViewsRequired": ["albedo-unlit", "environment-reflection", "grazing", "neutral-studio", "reference-beauty"]};
  root.userData.materialReferenceRegistry = "/mnt/c/repos/image2threejs/docs/materials/material-reference.json";

  const materialMap: Record<string, THREE.Material> = {};
  materialMap["painted-body"] = createSculptMaterial(
    "painted-body",
    {"id": "painted-body", "name": "Painted porcelain body (projected)", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#EEECE6", "color": "#EEECE6", "albedo": {"dominant": "#EEECE6", "secondary": ["#EEECE6", "#E8902A", "#2E9A84", "#C8402A", "#D8A040", "#1E2430"], "samplingNotes": "reference-derived"}, "colorVariation": {"palette": ["#EEECE6", "#E8902A", "#2E9A84", "#C8402A", "#D8A040", "#1E2430"], "pattern": "reference-projected", "amplitude": 0.0, "heightCorrelation": 0.0}, "textureResolution": 2048, "textureProjection": {"mode": "uv", "repeat": [1, 1], "anisotropy": 8, "texelDensityIntent": "lathe UV bake, 1 repeat"}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.42, "role": "broad color and height breakup"}, {"id": "meso", "frequency": 12.0, "amplitude": 0.22, "role": "ridges, pores, grain, dents, or equivalent visible relief"}, {"id": "micro", "frequency": 56.0, "amplitude": 0.08, "role": "highlight breakup visible under grazing light"}], "roughness": {"base": 0.16, "variation": 0.08, "map": "independent: glaze 0.14 / enamel 0.22 derived from the enamel paint mask (saturation), not from colour luminance", "localResponse": "enamel slightly less glossy than bare glaze"}, "metalness": {"base": 0.0, "variation": 0.0}, "normal": {"pattern": "enamel relief from paint mask (scripts/make_surface_maps.py)", "strength": 0.3, "scale": 1.0, "space": "tangent"}, "bump": {"pattern": "enamel relief height (paint mask)", "amplitude": 0.0015, "scale": 1.0}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 1.0, "contactShadowBias": 0.35, "notes": "baked per-row cavity AO (gallery, foot) in the ao map"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "ovr-shoulder-harlequin", "region": "v rows of h 1.0-1.26", "albedo": "projected", "notes": "orange #E8902A / teal #2E9A84 diamonds"}, {"id": "ovr-foot-harlequin", "region": "h 0.05-0.2", "albedo": "projected"}, {"id": "ovr-figural-panel", "region": "front phi -60..60 deg, h 0.2-0.95", "albedo": "projected"}, {"id": "ovr-red-chrysanthemum-ground", "region": "panel flanks", "albedo": "projected", "notes": "red #C8402A ground"}, {"id": "ovr-glaze-specular", "roughness": 0.12, "clearcoat": 0.5, "notes": "glossy glaze over the enamel"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "clearcoat": {"base": 0.2}, "clearcoatRoughness": {"base": 0.06}, "envMapIntensity": 1.0, "doubleSided": false, "referencePbr": {"usable": true, "confidence": 0.8, "targetThreshold": 0.7, "source": "reference-pixel-projection", "method": "scripts/bake_projection.py: fitted-camera projection of reference.jpg into LatheGeometry UV, glaze-white shading normalisation", "confidenceRationale": "camera/profile fitted to IoU 0.964; front hemisphere is direct reference pixels; rear/sides are speculative copies (documented)", "maps": {"albedo": {"path": "models/porcelain-teapot/assets/textures/body_albedo.png"}, "roughness": {"path": "models/porcelain-teapot/assets/textures/body_roughness.png"}, "height": {"path": "models/porcelain-teapot/assets/textures/body_height.png"}, "normal": {"path": "models/porcelain-teapot/assets/textures/body_normal.png"}, "ao": {"path": "models/porcelain-teapot/assets/textures/body_ao.png"}}}, "materialEvidence": {"crop": "material-evidence/crop-band.png", "pbrEvidence": "material-evidence/pbr-band.json", "pbrConfidence": 0.791, "analyzeTexture": "material-evidence/analyze-band.json", "classDecision": "agent reading, see material-evidence/README.md (analyzer class rejected)", "regionAnalysis": {"componentId": "body", "regionId": "shoulder-harlequin-band", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/00-shoulder-harlequin-band.png", "bbox": {"x": 470, "y": 320, "width": 230, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0126}, "observations": ["chromatic base-colour response", "visible meso/micro variation", "strong image-space gradient; verify it is material pattern, not lighting", "single-image PBR inference requires controlled render validation"], "hypothesis": {"componentId": "body", "regionId": "shoulder-harlequin-band", "materialId": null, "family": "ceramic", "subtype": "glazed", "finish": "glossy", "aliases": ["porcelain", "overglaze enamel"], "confidence": 0.791, "source": "vision"}, "alternatives": []}, "priorOverride": "registry roughness prior replaced by reference-measured value (material-evidence/README.md)"}, "referenceMaterialId": "ceramic.glazed", "materialFamily": "ceramic", "materialSubtype": "glazed", "materialFinish": "glossy", "materialReference": {"registry": "/mnt/c/repos/image2threejs/docs/materials/material-reference.json", "profileId": "ceramic.glazed", "method": "family-subtype-finish", "confidence": 0.791, "sourceRefs": ["three.mesh-physical", "adobe.pbr-guide-1", "google.filament-pbr"], "requiredMaps": ["map", "roughnessMap"], "optionalMaps": ["normalMap", "clearcoatMap", "clearcoatRoughnessMap"], "validationViews": ["neutral-studio", "grazing", "environment-reflection", "reference-beauty"]}, "textureAnalysis": {"finishClass": "candy-coat", "recipe": {"metalness": 0.35, "roughness": 0.18, "clearcoat": 0.6, "clearcoatRoughness": 0.15, "transmission": 0.0, "ior": 1.5, "envMapIntensity": 0.7, "anisotropy": 0.0, "procedural": "gradient-smoke"}, "palette": ["#989D84", "#657354", "#475D3C", "#825B20", "#A7611D"], "paletteHueRisk": [], "gradientAxis": "vertical", "stats": {"meanLum": 101.9, "meanSaturation": 0.8, "gradientStrength": 0.282, "mottle": 0.045, "streakRatio": 1.6, "streakEnergy": 0.0457, "hueSpread": 0.67, "specularFraction": 0.008}}},
    options
  );
  materialMap["painted-lid"] = createSculptMaterial(
    "painted-lid",
    {"id": "painted-lid", "name": "Painted porcelain lid (projected)", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#EEECE6", "color": "#EEECE6", "albedo": {"dominant": "#EEECE6", "secondary": ["#EEECE6", "#C8402A", "#2E9A84"], "samplingNotes": "reference-derived"}, "colorVariation": {"palette": ["#EEECE6", "#C8402A", "#2E9A84"], "pattern": "reference-projected", "amplitude": 0.0, "heightCorrelation": 0.0}, "textureResolution": 2048, "textureProjection": {"mode": "uv", "repeat": [1, 1], "anisotropy": 8, "texelDensityIntent": "lathe UV bake, 1 repeat"}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.42, "role": "broad color and height breakup"}, {"id": "meso", "frequency": 12.0, "amplitude": 0.22, "role": "ridges, pores, grain, dents, or equivalent visible relief"}, {"id": "micro", "frequency": 56.0, "amplitude": 0.08, "role": "highlight breakup visible under grazing light"}], "roughness": {"base": 0.16, "variation": 0.08, "map": "independent: glaze 0.14 / enamel 0.22 derived from the enamel paint mask (saturation), not from colour luminance", "localResponse": "enamel slightly less glossy than bare glaze"}, "metalness": {"base": 0.0, "variation": 0.0}, "normal": {"pattern": "enamel relief from paint mask (scripts/make_surface_maps.py)", "strength": 0.3, "scale": 1.0, "space": "tangent"}, "bump": {"pattern": "enamel relief height (paint mask)", "amplitude": 0.0015, "scale": 1.0}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 1.0, "contactShadowBias": 0.35, "notes": "baked per-row cavity AO (gallery, foot) in the ao map"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "ovr-lid-chrysanthemum", "region": "front of dome", "albedo": "projected"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "clearcoat": {"base": 0.2}, "clearcoatRoughness": {"base": 0.06}, "envMapIntensity": 1.0, "doubleSided": false, "referencePbr": {"usable": true, "confidence": 0.8, "targetThreshold": 0.7, "source": "reference-pixel-projection", "method": "scripts/bake_projection.py: fitted-camera projection of reference.jpg into LatheGeometry UV, glaze-white shading normalisation", "confidenceRationale": "camera/profile fitted to IoU 0.964; front hemisphere is direct reference pixels; rear/sides are speculative copies (documented)", "maps": {"albedo": {"path": "models/porcelain-teapot/assets/textures/lid_albedo.png"}, "roughness": {"path": "models/porcelain-teapot/assets/textures/lid_roughness.png"}, "height": {"path": "models/porcelain-teapot/assets/textures/lid_height.png"}, "normal": {"path": "models/porcelain-teapot/assets/textures/lid_normal.png"}, "ao": {"path": "models/porcelain-teapot/assets/textures/lid_ao.png"}}}, "materialEvidence": {"crop": "material-evidence/crop-band.png", "pbrEvidence": "material-evidence/pbr-band.json", "pbrConfidence": 0.791, "analyzeTexture": "material-evidence/analyze-band.json", "classDecision": "agent reading, see material-evidence/README.md (analyzer class rejected)", "regionAnalysis": {"componentId": "lid", "regionId": "lid-chrysanthemum", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/03-lid-chrysanthemum.png", "bbox": {"x": 560, "y": 195, "width": 120, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0066}, "observations": ["chromatic base-colour response", "visible meso/micro variation", "strong image-space gradient; verify it is material pattern, not lighting", "single-image PBR inference requires controlled render validation"], "hypothesis": {"componentId": "lid", "regionId": "lid-chrysanthemum", "materialId": null, "family": "ceramic", "subtype": "glazed", "finish": "glossy", "aliases": ["porcelain"], "confidence": 0.86, "source": "vision"}, "alternatives": []}, "priorOverride": "registry roughness prior replaced by reference-measured value (material-evidence/README.md)"}, "referenceMaterialId": "ceramic.glazed", "materialFamily": "ceramic", "materialSubtype": "glazed", "materialFinish": "glossy", "materialReference": {"registry": "/mnt/c/repos/image2threejs/docs/materials/material-reference.json", "profileId": "ceramic.glazed", "method": "family-subtype-finish", "confidence": 0.86, "sourceRefs": ["three.mesh-physical", "adobe.pbr-guide-1", "google.filament-pbr"], "requiredMaps": ["map", "roughnessMap"], "optionalMaps": ["normalMap", "clearcoatMap", "clearcoatRoughnessMap"], "validationViews": ["neutral-studio", "grazing", "environment-reflection", "reference-beauty"]}, "textureAnalysis": {"finishClass": "candy-coat", "recipe": {"metalness": 0.35, "roughness": 0.18, "clearcoat": 0.6, "clearcoatRoughness": 0.15, "transmission": 0.0, "ior": 1.5, "envMapIntensity": 0.7, "anisotropy": 0.0, "procedural": "gradient-smoke"}, "palette": ["#C4C9C5", "#996A61", "#78392D", "#7A3427", "#6E9994"], "paletteHueRisk": [], "gradientAxis": "vertical", "stats": {"meanLum": 107.7, "meanSaturation": 0.553, "gradientStrength": 0.554, "mottle": 0.062, "streakRatio": 1.16, "streakEnergy": 0.0613, "hueSpread": 0.223, "specularFraction": 0.002}}},
    options
  );
  materialMap["porcelain-glaze"] = createSculptMaterial(
    "porcelain-glaze",
    {"id": "porcelain-glaze", "name": "White porcelain glaze", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#EEECE6", "color": "#EEECE6", "albedo": {"dominant": "#EEECE6", "secondary": ["#EEECE6"], "samplingNotes": "reference-derived"}, "colorVariation": {"palette": ["#EEECE6"], "pattern": "flat", "amplitude": 0.0, "heightCorrelation": 0.0}, "roughness": {"base": 0.1, "variation": 0.02, "map": "scalar", "localResponse": "uniform glaze gloss"}, "metalness": {"base": 0.0, "variation": 0.0}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "baked per-row cavity AO (gallery, foot) in the ao map"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "ovr-handle-stain", "region": "handle lower root", "albedo": "#D9CFC2", "notes": "realised by component handle-stain-decal (reference pixels, alpha from warm deviation)"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "clearcoat": {"base": 0.6}, "clearcoatRoughness": {"base": 0.06}, "envMapIntensity": 1.0, "doubleSided": true, "textureless": {"declared": true, "evidence": ["reference view spout-crop and handle-crop: uniform white glaze; only specular window reflections, no grain/texture detail"]}, "materialEvidence": {"crop": "material-evidence/crop-glaze.png", "pbrEvidence": "material-evidence/pbr-glaze.json", "pbrConfidence": 0.75, "analyzeTexture": "material-evidence/analyze-glaze.json", "classDecision": "agent reading, see material-evidence/README.md (analyzer class rejected)", "regionAnalysis": {"componentId": "spout", "regionId": "white-glaze", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/01-white-glaze.png", "bbox": {"x": 215, "y": 330, "width": 85, "height": 90}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.007}, "observations": ["near-neutral colour response", "strong image-space gradient; verify it is material pattern, not lighting", "single-image PBR inference requires controlled render validation"], "hypothesis": {"componentId": "spout", "regionId": "white-glaze", "materialId": null, "family": "ceramic", "subtype": "glazed", "finish": "glossy", "aliases": ["porcelain"], "confidence": 0.796, "source": "vision"}, "alternatives": []}, "priorOverride": "registry roughness prior replaced by reference-measured value (material-evidence/README.md)"}, "referenceMaterialId": "ceramic.glazed", "materialFamily": "ceramic", "materialSubtype": "glazed", "materialFinish": "glossy", "materialReference": {"registry": "/mnt/c/repos/image2threejs/docs/materials/material-reference.json", "profileId": "ceramic.glazed", "method": "family-subtype-finish", "confidence": 0.796, "sourceRefs": ["three.mesh-physical", "adobe.pbr-guide-1", "google.filament-pbr"], "requiredMaps": ["map", "roughnessMap"], "optionalMaps": ["normalMap", "clearcoatMap", "clearcoatRoughnessMap"], "validationViews": ["neutral-studio", "grazing", "environment-reflection", "reference-beauty"]}, "textureAnalysis": {"finishClass": "brushed-steel", "recipe": {"metalness": 1.0, "roughness": 0.35, "clearcoat": 0.0, "clearcoatRoughness": 0.0, "transmission": 0.0, "ior": 1.5, "envMapIntensity": 1.0, "anisotropy": 1.0, "procedural": "brushed"}, "palette": ["#9EABBF", "#B2BCCA", "#DFE8E4", "#CACFCB", "#A19792"], "paletteHueRisk": [{"stop": "#9EABBF", "hueRisk": "blue-collapse", "suggestedRgb": [191, 171, 158]}], "gradientAxis": "horizontal", "stats": {"meanLum": 198.2, "meanSaturation": 0.076, "gradientStrength": 0.303, "mottle": 0.012, "streakRatio": 1.71, "streakEnergy": 0.0115, "hueSpread": 0.003, "specularFraction": 0.084}}},
    options
  );
  materialMap["gold-lustre"] = createSculptMaterial(
    "gold-lustre",
    {"id": "gold-lustre", "name": "Gold lustre", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#9A7236", "color": "#9A7236", "albedo": {"dominant": "#9A7236", "secondary": ["#C9A15A", "#8A6A30"], "samplingNotes": "reference-derived"}, "colorVariation": {"palette": ["#C9A15A", "#8A6A30"], "pattern": "flat", "amplitude": 0.0, "heightCorrelation": 0.0}, "roughness": {"base": 0.3, "variation": 0.02, "map": "scalar", "localResponse": "uniform glaze gloss"}, "metalness": {"base": 1.0, "variation": 0.0}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "baked per-row cavity AO (gallery, foot) in the ao map"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "ovr-gold-polish", "roughness": 0.22, "notes": "knob cap hotspot"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "clearcoat": {"base": 0.0}, "clearcoatRoughness": {"base": 0.06}, "envMapIntensity": 1.0, "doubleSided": false, "textureless": {"declared": true, "evidence": ["reference view lid-crop and handle-crop: smooth burnished gold, no visible texture"]}, "materialEvidence": {"crop": "material-evidence/crop-gold.png", "pbrEvidence": "material-evidence/pbr-gold.json", "pbrConfidence": 0.766, "analyzeTexture": "material-evidence/analyze-gold.json", "classDecision": "agent reading, see material-evidence/README.md (analyzer class rejected)", "regionAnalysis": {"componentId": "knob-gold-cap", "regionId": "gold-lustre", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/02-gold-lustre.png", "bbox": {"x": 640, "y": 124, "width": 72, "height": 24}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0016}, "observations": ["chromatic base-colour response", "strong image-space gradient; verify it is material pattern, not lighting", "single-image PBR inference requires controlled render validation"], "hypothesis": {"componentId": "knob-gold-cap", "regionId": "gold-lustre", "materialId": null, "family": "metal", "subtype": "gold", "finish": "polished", "aliases": ["gold"], "confidence": 0.766, "source": "vision"}, "alternatives": []}, "priorOverride": "registry roughness prior replaced by reference-measured value (material-evidence/README.md)"}, "referenceMaterialId": "metal.gold", "materialFamily": "metal", "materialSubtype": "gold", "materialFinish": "polished", "materialReference": {"registry": "/mnt/c/repos/image2threejs/docs/materials/material-reference.json", "profileId": "metal.gold", "method": "family-subtype-finish", "confidence": 0.766, "sourceRefs": ["three.mesh-standard", "three.pmrem", "gltf.2", "khronos.gltf-pbr", "adobe.pbr-guide-2", "google.filament-pbr"], "requiredMaps": ["map", "roughnessMap"], "optionalMaps": ["normalMap", "metalnessMap"], "validationViews": ["albedo-unlit", "environment-reflection", "grazing", "reference-beauty"]}, "textureAnalysis": {"finishClass": "painted-metal", "recipe": {"metalness": 0.0, "roughness": 0.5, "clearcoat": 1.0, "clearcoatRoughness": 0.05, "transmission": 0.0, "ior": 1.5, "envMapIntensity": 1.0, "anisotropy": 0.0, "procedural": "flat-clearcoat"}, "palette": ["#533527", "#4D2810", "#2F1206", "#3B220F", "#5E4426"], "paletteHueRisk": [], "gradientAxis": "vertical", "stats": {"meanLum": 45.6, "meanSaturation": 0.744, "gradientStrength": 0.183, "mottle": 0.018, "streakRatio": 1.7, "streakEnergy": 0.0188, "hueSpread": 0.004, "specularFraction": 0.0}}, "needsEnvironment": true},
    options
  );
  materialMap["enamel-decal"] = createSculptMaterial(
    "enamel-decal",
    {"id": "enamel-decal", "name": "Overglaze enamel decal", "type": "physical", "shaderModel": "MeshPhysicalMaterial", "baseColor": "#FFFFFF", "color": "#FFFFFF", "albedo": {"dominant": "#FFFFFF", "secondary": ["#FFFFFF"], "samplingNotes": "map supplied by the decal primitive"}, "colorVariation": {"palette": ["#FFFFFF"], "pattern": "flat", "amplitude": 0.0, "heightCorrelation": 0.0}, "roughness": {"base": 0.14, "variation": 0.02, "map": "scalar", "localResponse": "uniform glaze gloss"}, "metalness": {"base": 0.0, "variation": 0.0}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "baked per-row cavity AO (gallery, foot) in the ao map"}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Replace with image-derived color, roughness, noise, and edge-wear notes.", "clearcoat": {"base": 0.5}, "clearcoatRoughness": {"base": 0.06}, "envMapIntensity": 1.0, "doubleSided": false, "textureless": {"declared": true, "evidence": ["decal carries its RGBA map from reference pixels (spout-crop); material itself has no procedural texture"]}, "materialEvidence": {"crop": "material-evidence/crop-band.png", "pbrEvidence": "material-evidence/pbr-band.json", "pbrConfidence": 0.791, "analyzeTexture": "material-evidence/analyze-band.json", "classDecision": "agent reading, see material-evidence/README.md (analyzer class rejected)", "regionAnalysis": {"componentId": "spout-flower-decal", "regionId": "spout-flower", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/04-spout-flower.png", "bbox": {"x": 258, "y": 475, "width": 62, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0034}, "observations": ["chromatic base-colour response", "strong image-space gradient; verify it is material pattern, not lighting", "single-image PBR inference requires controlled render validation"], "hypothesis": {"componentId": "spout-flower-decal", "regionId": "spout-flower", "materialId": null, "family": "ceramic", "subtype": "glazed", "finish": "glossy", "aliases": ["overglaze enamel"], "confidence": 0.86, "source": "vision"}, "alternatives": []}, "priorOverride": "registry roughness prior replaced by reference-measured value (material-evidence/README.md)"}, "referenceMaterialId": "ceramic.glazed", "materialFamily": "ceramic", "materialSubtype": "glazed", "materialFinish": "glossy", "materialReference": {"registry": "/mnt/c/repos/image2threejs/docs/materials/material-reference.json", "profileId": "ceramic.glazed", "method": "family-subtype-finish", "confidence": 0.86, "sourceRefs": ["three.mesh-physical", "adobe.pbr-guide-1", "google.filament-pbr"], "requiredMaps": ["map", "roughnessMap"], "optionalMaps": ["normalMap", "clearcoatMap", "clearcoatRoughnessMap"], "validationViews": ["neutral-studio", "grazing", "environment-reflection", "reference-beauty"]}, "textureAnalysis": {"finishClass": "painted-metal", "recipe": {"metalness": 0.0, "roughness": 0.5, "clearcoat": 1.0, "clearcoatRoughness": 0.05, "transmission": 0.0, "ior": 1.5, "envMapIntensity": 1.0, "anisotropy": 0.0, "procedural": "flat-clearcoat"}, "palette": ["#DADCDA", "#BC958F", "#A57974", "#9A7274", "#948995"], "paletteHueRisk": [], "gradientAxis": "vertical", "stats": {"meanLum": 151.2, "meanSaturation": 0.309, "gradientStrength": 0.384, "mottle": 0.034, "streakRatio": 0.99, "streakEnergy": 0.0343, "hueSpread": 0.018, "specularFraction": 0.005}}},
    options
  );

  const nodes: Record<string, THREE.Object3D> = { root };
  const meshes: Record<string, THREE.Mesh> = {};
  const sockets: Record<string, THREE.Object3D> = {};
  const colliders: Record<string, unknown> = {};
  const destructionGroups: Record<string, THREE.Object3D[]> = {};

  const endpoint_body_0 = makeAttachmentEndpoint(null);
  const node_body_0 = new THREE.Group();
  node_body_0.name = "Teapot body__pivot";
  node_body_0.scale.set(1, 1, 1);
  if (endpoint_body_0) {
    node_body_0.position.copy(endpoint_body_0.start);
    node_body_0.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_body_0.position.set(0.0, 0.0, 0.0);
    node_body_0.rotation.set(0.0, 0.0, 0.0);
  }
  node_body_0.userData.sculptComponent = {"id": "body", "name": "Teapot body", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.85, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Revolved oblate vessel: foot, lower wall, equator (r=1.0 at h~0.7), shoulder, collar and rolled lip with inner gallery seat; one smooth lathe profile fitted to the reference silhouette (IoU 0.964).", "geometryDescriptor": {"topologyIntent": "Revolved oblate vessel: foot, lower wall, equator (r=1.0 at h~0.7), shoulder, collar and rolled lip with inner gallery seat; one smooth lathe profile fitted to the reference silhouette (IoU 0.964).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "LatheGeometry native UV (u=phi/2pi from +Z, v=profile index); reference photo baked into it", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0, 0.0], [0.02203, -3e-05], [0.04406, -0.0001], [0.06609, -0.00018], [0.08812, -0.00027], [0.11015, -0.00036], [0.13218, -0.00046], [0.15421, -0.00055], [0.17624, -0.00064], [0.19827, -0.00072], [0.22031, -0.0008], [0.24234, -0.00087], [0.26437, -0.00094], [0.2864, -0.001], [0.30843, -0.00104], [0.33046, -0.00108], [0.35249, -0.0011], [0.37452, -0.00111], [0.39655, -0.0011], [0.41858, -0.00106], [0.44061, -0.00099], [0.46264, -0.00088], [0.48467, -0.00071], [0.5067, -0.00046], [0.52873, -3e-05], [0.55075, 0.00073], [0.57254, 0.00312], [0.55225, 0.00994], [0.56242, 0.02135], [0.58231, 0.03082], [0.60211, 0.04048], [0.62153, 0.05087], [0.64079, 0.06157], [0.65991, 0.07251], [0.67856, 0.08423], [0.69641, 0.09714], [0.71333, 0.11125], [0.7296, 0.1261], [0.74527, 0.14158], [0.76038, 0.15761], [0.77498, 0.1741], [0.78914, 0.19098], [0.80297, 0.20814], [0.81641, 0.22559], [0.82941, 0.24337], [0.84192, 0.26151], [0.85388, 0.28001], [0.86523, 0.29889], [0.87598, 0.31811], [0.88626, 0.3376], [0.89615, 0.35729], [0.90573, 0.37712], [0.91507, 0.39708], [0.92422, 0.41712], [0.93322, 0.43722], [0.94217, 0.45736], [0.95096, 0.47756], [0.9595, 0.49787], [0.96764, 0.51834], [0.97523, 0.53902], [0.98208, 0.55996], [0.98799, 0.58118], [0.99308, 0.60261], [0.99731, 0.62423], [1.00061, 0.64601], [1.00301, 0.66791], [1.00455, 0.68988], [1.00528, 0.7119], [1.0051, 0.73393], [1.00414, 0.75594], [1.00252, 0.77791], [1.00034, 0.79983], [0.99764, 0.82169], [0.99445, 0.84349], [0.99049, 0.86516], [0.98571, 0.88667], [0.98024, 0.90801], [0.97425, 0.92921], [0.96794, 0.95032], [0.96155, 0.9714], [0.95511, 0.99247], [0.94835, 1.01344], [0.94094, 1.03418], [0.93251, 1.05453], [0.92273, 1.07427], [0.91186, 1.09342], [0.89999, 1.11198], [0.88702, 1.12979], [0.87287, 1.14666], [0.85746, 1.1624], [0.84054, 1.17649], [0.82254, 1.18919], [0.80397, 1.20104], [0.78515, 1.2125], [0.76642, 1.22409], [0.74768, 1.23566], [0.72857, 1.24662], [0.70942, 1.25752], [0.69086, 1.26937], [0.67396, 1.28346], [0.66395, 1.30292], [0.65917, 1.3244], [0.65645, 1.34626], [0.65428, 1.36819], [0.65335, 1.39019], [0.65262, 1.41221], [0.65017, 1.4341], [0.64323, 1.4548], [0.62355, 1.4613], [0.61191, 1.44432], [0.60889, 1.4225], [0.60488, 1.40087], [0.59213, 1.38394], [0.57064, 1.37943], [0.54868, 1.3777], [0.5267, 1.37622], [0.50475, 1.37442], [0.48281, 1.3724], [0.46088, 1.37028], [0.43895, 1.36813], [0.41703, 1.36598], [0.3951, 1.36384], [0.37317, 1.36173], [0.35124, 1.35966], [0.3293, 1.35763], [0.30736, 1.35565], [0.28541, 1.35373], [0.26346, 1.35184], [0.24151, 1.34998], [0.21956, 1.34813], [0.19761, 1.34629], [0.17565, 1.34447], [0.1537, 1.34265], [0.13174, 1.34084], [0.10978, 1.33903], [0.08783, 1.33722], [0.06587, 1.33542], [0.04391, 1.33362], [0.02196, 1.33182], [0.0, 1.33]], "segments": 128}, "profileSource": "scripts/fit_shape.py coordinate descent vs evidence/mask_840.png, Catmull-Rom resampled to 96 arc-length points"}, "parent": null, "attachment": null, "dimensions": {"width": 2.01, "height": 1.462, "depth": 2.01, "units": "world (authored geometry, transform.scale pins it)", "confidence": 0.85}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85, "worldPosition": [0, 0, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "lid-gallery-seat", "localPosition": [0, 1.375, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "lid flange rests here"}, {"id": "spout-root", "localPosition": [-0.64, 0.5, 0.0], "worldPosition": [-0.64, 0.5, 0.0], "frame": "component node", "notes": "spout weld"}, {"id": "handle-roots", "localPosition": [0.76, 1.13, 0.0], "worldPosition": [0.76, 1.13, 0.0], "frame": "component node", "notes": "handle upper weld"}], "collider": {"type": "cylinder", "offset": [0, 0.73, 0], "scale": [2.0, 1.46, 2.0], "isTrigger": false, "notes": "body proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "vessel", "seamRefs": ["spout-root", "handle-roots"], "detachableFragments": ["lid"], "breakImpulse": 6.0, "debrisMaterial": "painted-body"}}, "material": "painted-body", "materialLayers": ["painted-body", "porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "shoulder-harlequin-band", "type": "decal", "description": "projected orange/teal harlequin band h 1.0-1.26", "mapsTo": "ovr-shoulder-harlequin"}, {"id": "foot-harlequin-band", "type": "decal", "description": "projected lower harlequin band h 0.05-0.2", "mapsTo": "ovr-foot-harlequin"}, {"id": "figural-panel", "type": "decal", "description": "projected figural panel, front (phi 0) and rotated copy at rear", "mapsTo": "ovr-figural-panel"}, {"id": "lid-gallery-seat", "type": "socket", "description": "inner seat at h 1.36 that the lid flange rests on"}], "surfaceDetail": {"macroRoughness": 0.16, "microRoughness": 0.08, "bumpAmplitude": 0.0015, "normalPattern": "enamel relief over glaze (paint mask)", "occlusionPattern": "baked cavity AO: gallery 0.35, foot 0.55", "edgeWearPattern": "none observed", "notes": "glossy glaze; painted enamel slightly proud and less glossy"}, "evidenceRefs": ["full-object", "panel-zone"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(232, 144, 42, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["full-object", "panel-zone"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "painted-body"}, "materialRegions": [{"regionId": "shoulder-harlequin-band", "materialId": "painted-body", "profileId": "ceramic.glazed", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/00-shoulder-harlequin-band.png", "bbox": {"x": 470, "y": 320, "width": 230, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0126}}]};
  node_body_0.userData.actionProfile = {"animationRole": "root", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85, "worldPosition": [0, 0, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "lid-gallery-seat", "localPosition": [0, 1.375, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "lid flange rests here"}, {"id": "spout-root", "localPosition": [-0.64, 0.5, 0.0], "worldPosition": [-0.64, 0.5, 0.0], "frame": "component node", "notes": "spout weld"}, {"id": "handle-roots", "localPosition": [0.76, 1.13, 0.0], "worldPosition": [0.76, 1.13, 0.0], "frame": "component node", "notes": "handle upper weld"}], "collider": {"type": "cylinder", "offset": [0, 0.73, 0], "scale": [2.0, 1.46, 2.0], "isTrigger": false, "notes": "body proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "vessel", "seamRefs": ["spout-root", "handle-roots"], "detachableFragments": ["lid"], "breakImpulse": 6.0, "debrisMaterial": "painted-body"}};
  (nodes["root"] ?? root).add(node_body_0);
  nodes["body"] = node_body_0;
  const mesh_body_0Geometry = endpoint_body_0
    ? new THREE.CylinderGeometry(endpoint_body_0.endRadius, endpoint_body_0.baseRadius, endpoint_body_0.length, 32, 12)
    : buildLatheGeometry({"points": [[0.0, 0.0], [0.02203, -3e-05], [0.04406, -0.0001], [0.06609, -0.00018], [0.08812, -0.00027], [0.11015, -0.00036], [0.13218, -0.00046], [0.15421, -0.00055], [0.17624, -0.00064], [0.19827, -0.00072], [0.22031, -0.0008], [0.24234, -0.00087], [0.26437, -0.00094], [0.2864, -0.001], [0.30843, -0.00104], [0.33046, -0.00108], [0.35249, -0.0011], [0.37452, -0.00111], [0.39655, -0.0011], [0.41858, -0.00106], [0.44061, -0.00099], [0.46264, -0.00088], [0.48467, -0.00071], [0.5067, -0.00046], [0.52873, -3e-05], [0.55075, 0.00073], [0.57254, 0.00312], [0.55225, 0.00994], [0.56242, 0.02135], [0.58231, 0.03082], [0.60211, 0.04048], [0.62153, 0.05087], [0.64079, 0.06157], [0.65991, 0.07251], [0.67856, 0.08423], [0.69641, 0.09714], [0.71333, 0.11125], [0.7296, 0.1261], [0.74527, 0.14158], [0.76038, 0.15761], [0.77498, 0.1741], [0.78914, 0.19098], [0.80297, 0.20814], [0.81641, 0.22559], [0.82941, 0.24337], [0.84192, 0.26151], [0.85388, 0.28001], [0.86523, 0.29889], [0.87598, 0.31811], [0.88626, 0.3376], [0.89615, 0.35729], [0.90573, 0.37712], [0.91507, 0.39708], [0.92422, 0.41712], [0.93322, 0.43722], [0.94217, 0.45736], [0.95096, 0.47756], [0.9595, 0.49787], [0.96764, 0.51834], [0.97523, 0.53902], [0.98208, 0.55996], [0.98799, 0.58118], [0.99308, 0.60261], [0.99731, 0.62423], [1.00061, 0.64601], [1.00301, 0.66791], [1.00455, 0.68988], [1.00528, 0.7119], [1.0051, 0.73393], [1.00414, 0.75594], [1.00252, 0.77791], [1.00034, 0.79983], [0.99764, 0.82169], [0.99445, 0.84349], [0.99049, 0.86516], [0.98571, 0.88667], [0.98024, 0.90801], [0.97425, 0.92921], [0.96794, 0.95032], [0.96155, 0.9714], [0.95511, 0.99247], [0.94835, 1.01344], [0.94094, 1.03418], [0.93251, 1.05453], [0.92273, 1.07427], [0.91186, 1.09342], [0.89999, 1.11198], [0.88702, 1.12979], [0.87287, 1.14666], [0.85746, 1.1624], [0.84054, 1.17649], [0.82254, 1.18919], [0.80397, 1.20104], [0.78515, 1.2125], [0.76642, 1.22409], [0.74768, 1.23566], [0.72857, 1.24662], [0.70942, 1.25752], [0.69086, 1.26937], [0.67396, 1.28346], [0.66395, 1.30292], [0.65917, 1.3244], [0.65645, 1.34626], [0.65428, 1.36819], [0.65335, 1.39019], [0.65262, 1.41221], [0.65017, 1.4341], [0.64323, 1.4548], [0.62355, 1.4613], [0.61191, 1.44432], [0.60889, 1.4225], [0.60488, 1.40087], [0.59213, 1.38394], [0.57064, 1.37943], [0.54868, 1.3777], [0.5267, 1.37622], [0.50475, 1.37442], [0.48281, 1.3724], [0.46088, 1.37028], [0.43895, 1.36813], [0.41703, 1.36598], [0.3951, 1.36384], [0.37317, 1.36173], [0.35124, 1.35966], [0.3293, 1.35763], [0.30736, 1.35565], [0.28541, 1.35373], [0.26346, 1.35184], [0.24151, 1.34998], [0.21956, 1.34813], [0.19761, 1.34629], [0.17565, 1.34447], [0.1537, 1.34265], [0.13174, 1.34084], [0.10978, 1.33903], [0.08783, 1.33722], [0.06587, 1.33542], [0.04391, 1.33362], [0.02196, 1.33182], [0.0, 1.33]], "segments": 128});
  if (!endpoint_body_0) {
    mesh_body_0Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_body_0 = new THREE.Mesh(
    mesh_body_0Geometry,
    materialMap["painted-body"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_body_0.name = "Teapot body";
  if (endpoint_body_0) {
    mesh_body_0.position.copy(endpoint_body_0.midpoint);
    mesh_body_0.quaternion.copy(endpoint_body_0.quaternion);
  }
  mesh_body_0.castShadow = options.castShadow ?? true;
  mesh_body_0.receiveShadow = options.receiveShadow ?? true;
  mesh_body_0.userData.sculptComponent = {"id": "body", "name": "Teapot body", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.85, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Revolved oblate vessel: foot, lower wall, equator (r=1.0 at h~0.7), shoulder, collar and rolled lip with inner gallery seat; one smooth lathe profile fitted to the reference silhouette (IoU 0.964).", "geometryDescriptor": {"topologyIntent": "Revolved oblate vessel: foot, lower wall, equator (r=1.0 at h~0.7), shoulder, collar and rolled lip with inner gallery seat; one smooth lathe profile fitted to the reference silhouette (IoU 0.964).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "LatheGeometry native UV (u=phi/2pi from +Z, v=profile index); reference photo baked into it", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0, 0.0], [0.02203, -3e-05], [0.04406, -0.0001], [0.06609, -0.00018], [0.08812, -0.00027], [0.11015, -0.00036], [0.13218, -0.00046], [0.15421, -0.00055], [0.17624, -0.00064], [0.19827, -0.00072], [0.22031, -0.0008], [0.24234, -0.00087], [0.26437, -0.00094], [0.2864, -0.001], [0.30843, -0.00104], [0.33046, -0.00108], [0.35249, -0.0011], [0.37452, -0.00111], [0.39655, -0.0011], [0.41858, -0.00106], [0.44061, -0.00099], [0.46264, -0.00088], [0.48467, -0.00071], [0.5067, -0.00046], [0.52873, -3e-05], [0.55075, 0.00073], [0.57254, 0.00312], [0.55225, 0.00994], [0.56242, 0.02135], [0.58231, 0.03082], [0.60211, 0.04048], [0.62153, 0.05087], [0.64079, 0.06157], [0.65991, 0.07251], [0.67856, 0.08423], [0.69641, 0.09714], [0.71333, 0.11125], [0.7296, 0.1261], [0.74527, 0.14158], [0.76038, 0.15761], [0.77498, 0.1741], [0.78914, 0.19098], [0.80297, 0.20814], [0.81641, 0.22559], [0.82941, 0.24337], [0.84192, 0.26151], [0.85388, 0.28001], [0.86523, 0.29889], [0.87598, 0.31811], [0.88626, 0.3376], [0.89615, 0.35729], [0.90573, 0.37712], [0.91507, 0.39708], [0.92422, 0.41712], [0.93322, 0.43722], [0.94217, 0.45736], [0.95096, 0.47756], [0.9595, 0.49787], [0.96764, 0.51834], [0.97523, 0.53902], [0.98208, 0.55996], [0.98799, 0.58118], [0.99308, 0.60261], [0.99731, 0.62423], [1.00061, 0.64601], [1.00301, 0.66791], [1.00455, 0.68988], [1.00528, 0.7119], [1.0051, 0.73393], [1.00414, 0.75594], [1.00252, 0.77791], [1.00034, 0.79983], [0.99764, 0.82169], [0.99445, 0.84349], [0.99049, 0.86516], [0.98571, 0.88667], [0.98024, 0.90801], [0.97425, 0.92921], [0.96794, 0.95032], [0.96155, 0.9714], [0.95511, 0.99247], [0.94835, 1.01344], [0.94094, 1.03418], [0.93251, 1.05453], [0.92273, 1.07427], [0.91186, 1.09342], [0.89999, 1.11198], [0.88702, 1.12979], [0.87287, 1.14666], [0.85746, 1.1624], [0.84054, 1.17649], [0.82254, 1.18919], [0.80397, 1.20104], [0.78515, 1.2125], [0.76642, 1.22409], [0.74768, 1.23566], [0.72857, 1.24662], [0.70942, 1.25752], [0.69086, 1.26937], [0.67396, 1.28346], [0.66395, 1.30292], [0.65917, 1.3244], [0.65645, 1.34626], [0.65428, 1.36819], [0.65335, 1.39019], [0.65262, 1.41221], [0.65017, 1.4341], [0.64323, 1.4548], [0.62355, 1.4613], [0.61191, 1.44432], [0.60889, 1.4225], [0.60488, 1.40087], [0.59213, 1.38394], [0.57064, 1.37943], [0.54868, 1.3777], [0.5267, 1.37622], [0.50475, 1.37442], [0.48281, 1.3724], [0.46088, 1.37028], [0.43895, 1.36813], [0.41703, 1.36598], [0.3951, 1.36384], [0.37317, 1.36173], [0.35124, 1.35966], [0.3293, 1.35763], [0.30736, 1.35565], [0.28541, 1.35373], [0.26346, 1.35184], [0.24151, 1.34998], [0.21956, 1.34813], [0.19761, 1.34629], [0.17565, 1.34447], [0.1537, 1.34265], [0.13174, 1.34084], [0.10978, 1.33903], [0.08783, 1.33722], [0.06587, 1.33542], [0.04391, 1.33362], [0.02196, 1.33182], [0.0, 1.33]], "segments": 128}, "profileSource": "scripts/fit_shape.py coordinate descent vs evidence/mask_840.png, Catmull-Rom resampled to 96 arc-length points"}, "parent": null, "attachment": null, "dimensions": {"width": 2.01, "height": 1.462, "depth": 2.01, "units": "world (authored geometry, transform.scale pins it)", "confidence": 0.85}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85, "worldPosition": [0, 0, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "lid-gallery-seat", "localPosition": [0, 1.375, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "lid flange rests here"}, {"id": "spout-root", "localPosition": [-0.64, 0.5, 0.0], "worldPosition": [-0.64, 0.5, 0.0], "frame": "component node", "notes": "spout weld"}, {"id": "handle-roots", "localPosition": [0.76, 1.13, 0.0], "worldPosition": [0.76, 1.13, 0.0], "frame": "component node", "notes": "handle upper weld"}], "collider": {"type": "cylinder", "offset": [0, 0.73, 0], "scale": [2.0, 1.46, 2.0], "isTrigger": false, "notes": "body proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "vessel", "seamRefs": ["spout-root", "handle-roots"], "detachableFragments": ["lid"], "breakImpulse": 6.0, "debrisMaterial": "painted-body"}}, "material": "painted-body", "materialLayers": ["painted-body", "porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "shoulder-harlequin-band", "type": "decal", "description": "projected orange/teal harlequin band h 1.0-1.26", "mapsTo": "ovr-shoulder-harlequin"}, {"id": "foot-harlequin-band", "type": "decal", "description": "projected lower harlequin band h 0.05-0.2", "mapsTo": "ovr-foot-harlequin"}, {"id": "figural-panel", "type": "decal", "description": "projected figural panel, front (phi 0) and rotated copy at rear", "mapsTo": "ovr-figural-panel"}, {"id": "lid-gallery-seat", "type": "socket", "description": "inner seat at h 1.36 that the lid flange rests on"}], "surfaceDetail": {"macroRoughness": 0.16, "microRoughness": 0.08, "bumpAmplitude": 0.0015, "normalPattern": "enamel relief over glaze (paint mask)", "occlusionPattern": "baked cavity AO: gallery 0.35, foot 0.55", "edgeWearPattern": "none observed", "notes": "glossy glaze; painted enamel slightly proud and less glossy"}, "evidenceRefs": ["full-object", "panel-zone"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(232, 144, 42, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["full-object", "panel-zone"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "painted-body"}, "materialRegions": [{"regionId": "shoulder-harlequin-band", "materialId": "painted-body", "profileId": "ceramic.glazed", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/00-shoulder-harlequin-band.png", "bbox": {"x": 470, "y": 320, "width": 230, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0126}}]};
  node_body_0.add(mesh_body_0);
  meshes["body"] = mesh_body_0;
  colliders["body"] = {"type": "cylinder", "offset": [0, 0.73, 0], "scale": [2.0, 1.46, 2.0], "isTrigger": false, "notes": "body proxy"};
  destructionGroups["vessel"] ??= [];
  destructionGroups["vessel"].push(node_body_0);
  const socket_body_lid_gallery_seat_0 = new THREE.Object3D();
  socket_body_lid_gallery_seat_0.name = "lid-gallery-seat";
  socket_body_lid_gallery_seat_0.position.set(0.0, 1.375, 0.0);
  socket_body_lid_gallery_seat_0.rotation.set(0, 0, 0);
  socket_body_lid_gallery_seat_0.userData.socket = {"id": "lid-gallery-seat", "localPosition": [0, 1.375, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "lid flange rests here"};
  node_body_0.add(socket_body_lid_gallery_seat_0);
  sockets["body:lid-gallery-seat"] = socket_body_lid_gallery_seat_0;
  const socket_body_spout_root_1 = new THREE.Object3D();
  socket_body_spout_root_1.name = "spout-root";
  socket_body_spout_root_1.position.set(-0.64, 0.5, 0.0);
  socket_body_spout_root_1.rotation.set(0, 0, 0);
  socket_body_spout_root_1.userData.socket = {"id": "spout-root", "localPosition": [-0.64, 0.5, 0.0], "worldPosition": [-0.64, 0.5, 0.0], "frame": "component node", "notes": "spout weld"};
  node_body_0.add(socket_body_spout_root_1);
  sockets["body:spout-root"] = socket_body_spout_root_1;
  const socket_body_handle_roots_2 = new THREE.Object3D();
  socket_body_handle_roots_2.name = "handle-roots";
  socket_body_handle_roots_2.position.set(0.76, 1.13, 0.0);
  socket_body_handle_roots_2.rotation.set(0, 0, 0);
  socket_body_handle_roots_2.userData.socket = {"id": "handle-roots", "localPosition": [0.76, 1.13, 0.0], "worldPosition": [0.76, 1.13, 0.0], "frame": "component node", "notes": "handle upper weld"};
  node_body_0.add(socket_body_handle_roots_2);
  sockets["body:handle-roots"] = socket_body_handle_roots_2;

  const endpoint_lid_1 = makeAttachmentEndpoint(null);
  const node_lid_1 = new THREE.Group();
  node_lid_1.name = "Lid__pivot";
  node_lid_1.scale.set(1, 1, 1);
  if (endpoint_lid_1) {
    node_lid_1.position.copy(endpoint_lid_1.start);
    node_lid_1.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_lid_1.position.set(0.0, 1.362, 0.0);
    node_lid_1.rotation.set(0.0, 0.0, 0.0);
  }
  node_lid_1.userData.sculptComponent = {"id": "lid", "name": "Lid", "level": "macro", "role": "lid", "importance": 1.0, "confidence": 0.8, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Low revolved dome with a flange that drops into the collar; rests on the gallery seat.", "geometryDescriptor": {"topologyIntent": "Low revolved dome with a flange that drops into the collar; rests on the gallery seat.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "LatheGeometry native UV; reference photo baked (lid chrysanthemum)", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0, 0.018], [0.02812, 0.01795], [0.05625, 0.01787], [0.08437, 0.01777], [0.1125, 0.01766], [0.14062, 0.01755], [0.16875, 0.01744], [0.19687, 0.01734], [0.22499, 0.01724], [0.25312, 0.01715], [0.28124, 0.01707], [0.30937, 0.017], [0.33749, 0.01694], [0.36561, 0.01691], [0.39374, 0.01689], [0.42186, 0.0169], [0.44999, 0.01694], [0.47811, 0.01703], [0.50624, 0.01717], [0.53436, 0.01742], [0.56248, 0.01783], [0.59059, 0.01865], [0.61861, 0.02083], [0.59854, 0.03097], [0.60095, 0.0581], [0.58974, 0.08332], [0.56899, 0.10229], [0.54623, 0.11877], [0.52157, 0.13227], [0.49615, 0.14431], [0.47028, 0.15532], [0.44395, 0.16521], [0.41718, 0.17384], [0.39006, 0.18127], [0.36272, 0.18787], [0.33524, 0.19384], [0.30765, 0.1993], [0.27998, 0.20431], [0.25222, 0.2088], [0.22435, 0.21262], [0.19642, 0.21589], [0.16844, 0.21875], [0.14043, 0.22129], [0.1124, 0.22358], [0.08433, 0.22525], [0.05623, 0.22638], [0.02811, 0.2272], [0.0, 0.228]], "segments": 96}}, "parent": "body", "attachment": {"parentSocket": "lid-gallery-seat", "localStart": [0, 1.362, 0], "localEnd": [0, 1.59, 0], "contactType": "socket", "embedDepth": 0.0, "gapTolerance": 0.02, "notes": "flange rests inside the collar, 0.013 radial clearance to the inner wall", "frame": "parent node frame", "anchorNote": "welded to the vessel (embedding contract, not a worn/held item)"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 1.362, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "detachable-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.362, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "knob-grip", "localPosition": [0, 0.398, 0], "worldPosition": [0, 1.76, 0], "frame": "component node", "notes": "pinch-grip point to lift the lid"}, {"id": "lid-seat", "localPosition": [0, 0.013, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "mates body:lid-gallery-seat"}], "collider": {"type": "cylinder", "offset": [0, 0.108, 0], "scale": [1.18, 0.23, 1.18], "isTrigger": false, "notes": "lid proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "lid", "seamRefs": ["lid-seat"], "detachableFragments": [], "breakImpulse": 4.0, "debrisMaterial": "painted-lid"}}, "material": "painted-lid", "materialLayers": ["painted-lid", "porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "lid-chrysanthemum", "type": "decal", "description": "projected red chrysanthemum + teal leaves", "mapsTo": "ovr-lid-chrysanthemum"}], "surfaceDetail": {"macroRoughness": 0.16, "microRoughness": 0.08, "bumpAmplitude": 0.0015, "normalPattern": "enamel relief (chrysanthemum)", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(200, 64, 42, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "painted-lid"}, "materialRegions": [{"regionId": "lid-chrysanthemum", "materialId": "painted-lid", "profileId": "ceramic.glazed", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/03-lid-chrysanthemum.png", "bbox": {"x": 560, "y": 195, "width": 120, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0066}}]};
  node_lid_1.userData.actionProfile = {"animationRole": "detachable-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.362, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "knob-grip", "localPosition": [0, 0.398, 0], "worldPosition": [0, 1.76, 0], "frame": "component node", "notes": "pinch-grip point to lift the lid"}, {"id": "lid-seat", "localPosition": [0, 0.013, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "mates body:lid-gallery-seat"}], "collider": {"type": "cylinder", "offset": [0, 0.108, 0], "scale": [1.18, 0.23, 1.18], "isTrigger": false, "notes": "lid proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "lid", "seamRefs": ["lid-seat"], "detachableFragments": [], "breakImpulse": 4.0, "debrisMaterial": "painted-lid"}};
  (nodes["body"] ?? root).add(node_lid_1);
  nodes["lid"] = node_lid_1;
  const mesh_lid_1Geometry = endpoint_lid_1
    ? new THREE.CylinderGeometry(endpoint_lid_1.endRadius, endpoint_lid_1.baseRadius, endpoint_lid_1.length, 32, 12)
    : buildLatheGeometry({"points": [[0.0, 0.018], [0.02812, 0.01795], [0.05625, 0.01787], [0.08437, 0.01777], [0.1125, 0.01766], [0.14062, 0.01755], [0.16875, 0.01744], [0.19687, 0.01734], [0.22499, 0.01724], [0.25312, 0.01715], [0.28124, 0.01707], [0.30937, 0.017], [0.33749, 0.01694], [0.36561, 0.01691], [0.39374, 0.01689], [0.42186, 0.0169], [0.44999, 0.01694], [0.47811, 0.01703], [0.50624, 0.01717], [0.53436, 0.01742], [0.56248, 0.01783], [0.59059, 0.01865], [0.61861, 0.02083], [0.59854, 0.03097], [0.60095, 0.0581], [0.58974, 0.08332], [0.56899, 0.10229], [0.54623, 0.11877], [0.52157, 0.13227], [0.49615, 0.14431], [0.47028, 0.15532], [0.44395, 0.16521], [0.41718, 0.17384], [0.39006, 0.18127], [0.36272, 0.18787], [0.33524, 0.19384], [0.30765, 0.1993], [0.27998, 0.20431], [0.25222, 0.2088], [0.22435, 0.21262], [0.19642, 0.21589], [0.16844, 0.21875], [0.14043, 0.22129], [0.1124, 0.22358], [0.08433, 0.22525], [0.05623, 0.22638], [0.02811, 0.2272], [0.0, 0.228]], "segments": 96});
  if (!endpoint_lid_1) {
    mesh_lid_1Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_lid_1 = new THREE.Mesh(
    mesh_lid_1Geometry,
    materialMap["painted-lid"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_lid_1.name = "Lid";
  if (endpoint_lid_1) {
    mesh_lid_1.position.copy(endpoint_lid_1.midpoint);
    mesh_lid_1.quaternion.copy(endpoint_lid_1.quaternion);
  }
  mesh_lid_1.castShadow = options.castShadow ?? true;
  mesh_lid_1.receiveShadow = options.receiveShadow ?? true;
  mesh_lid_1.userData.sculptComponent = {"id": "lid", "name": "Lid", "level": "macro", "role": "lid", "importance": 1.0, "confidence": 0.8, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Low revolved dome with a flange that drops into the collar; rests on the gallery seat.", "geometryDescriptor": {"topologyIntent": "Low revolved dome with a flange that drops into the collar; rests on the gallery seat.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "LatheGeometry native UV; reference photo baked (lid chrysanthemum)", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0, 0.018], [0.02812, 0.01795], [0.05625, 0.01787], [0.08437, 0.01777], [0.1125, 0.01766], [0.14062, 0.01755], [0.16875, 0.01744], [0.19687, 0.01734], [0.22499, 0.01724], [0.25312, 0.01715], [0.28124, 0.01707], [0.30937, 0.017], [0.33749, 0.01694], [0.36561, 0.01691], [0.39374, 0.01689], [0.42186, 0.0169], [0.44999, 0.01694], [0.47811, 0.01703], [0.50624, 0.01717], [0.53436, 0.01742], [0.56248, 0.01783], [0.59059, 0.01865], [0.61861, 0.02083], [0.59854, 0.03097], [0.60095, 0.0581], [0.58974, 0.08332], [0.56899, 0.10229], [0.54623, 0.11877], [0.52157, 0.13227], [0.49615, 0.14431], [0.47028, 0.15532], [0.44395, 0.16521], [0.41718, 0.17384], [0.39006, 0.18127], [0.36272, 0.18787], [0.33524, 0.19384], [0.30765, 0.1993], [0.27998, 0.20431], [0.25222, 0.2088], [0.22435, 0.21262], [0.19642, 0.21589], [0.16844, 0.21875], [0.14043, 0.22129], [0.1124, 0.22358], [0.08433, 0.22525], [0.05623, 0.22638], [0.02811, 0.2272], [0.0, 0.228]], "segments": 96}}, "parent": "body", "attachment": {"parentSocket": "lid-gallery-seat", "localStart": [0, 1.362, 0], "localEnd": [0, 1.59, 0], "contactType": "socket", "embedDepth": 0.0, "gapTolerance": 0.02, "notes": "flange rests inside the collar, 0.013 radial clearance to the inner wall", "frame": "parent node frame", "anchorNote": "welded to the vessel (embedding contract, not a worn/held item)"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 1.362, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "detachable-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.362, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "knob-grip", "localPosition": [0, 0.398, 0], "worldPosition": [0, 1.76, 0], "frame": "component node", "notes": "pinch-grip point to lift the lid"}, {"id": "lid-seat", "localPosition": [0, 0.013, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "mates body:lid-gallery-seat"}], "collider": {"type": "cylinder", "offset": [0, 0.108, 0], "scale": [1.18, 0.23, 1.18], "isTrigger": false, "notes": "lid proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "lid", "seamRefs": ["lid-seat"], "detachableFragments": [], "breakImpulse": 4.0, "debrisMaterial": "painted-lid"}}, "material": "painted-lid", "materialLayers": ["painted-lid", "porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "lid-chrysanthemum", "type": "decal", "description": "projected red chrysanthemum + teal leaves", "mapsTo": "ovr-lid-chrysanthemum"}], "surfaceDetail": {"macroRoughness": 0.16, "microRoughness": 0.08, "bumpAmplitude": 0.0015, "normalPattern": "enamel relief (chrysanthemum)", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(200, 64, 42, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "painted-lid"}, "materialRegions": [{"regionId": "lid-chrysanthemum", "materialId": "painted-lid", "profileId": "ceramic.glazed", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/03-lid-chrysanthemum.png", "bbox": {"x": 560, "y": 195, "width": 120, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0066}}]};
  node_lid_1.add(mesh_lid_1);
  meshes["lid"] = mesh_lid_1;
  colliders["lid"] = {"type": "cylinder", "offset": [0, 0.108, 0], "scale": [1.18, 0.23, 1.18], "isTrigger": false, "notes": "lid proxy"};
  destructionGroups["lid"] ??= [];
  destructionGroups["lid"].push(node_lid_1);
  const socket_lid_knob_grip_0 = new THREE.Object3D();
  socket_lid_knob_grip_0.name = "knob-grip";
  socket_lid_knob_grip_0.position.set(0.0, 0.398, 0.0);
  socket_lid_knob_grip_0.rotation.set(0, 0, 0);
  socket_lid_knob_grip_0.userData.socket = {"id": "knob-grip", "localPosition": [0, 0.398, 0], "worldPosition": [0, 1.76, 0], "frame": "component node", "notes": "pinch-grip point to lift the lid"};
  node_lid_1.add(socket_lid_knob_grip_0);
  sockets["lid:knob-grip"] = socket_lid_knob_grip_0;
  const socket_lid_lid_seat_1 = new THREE.Object3D();
  socket_lid_lid_seat_1.name = "lid-seat";
  socket_lid_lid_seat_1.position.set(0.0, 0.013, 0.0);
  socket_lid_lid_seat_1.rotation.set(0, 0, 0);
  socket_lid_lid_seat_1.userData.socket = {"id": "lid-seat", "localPosition": [0, 0.013, 0], "worldPosition": [0, 1.375, 0], "frame": "component node", "notes": "mates body:lid-gallery-seat"};
  node_lid_1.add(socket_lid_lid_seat_1);
  sockets["lid:lid-seat"] = socket_lid_lid_seat_1;

  const endpoint_knob_2 = makeAttachmentEndpoint(null);
  const node_knob_2 = new THREE.Group();
  node_knob_2.name = "Lid knob__pivot";
  node_knob_2.scale.set(1, 1, 1);
  if (endpoint_knob_2) {
    node_knob_2.position.copy(endpoint_knob_2.start);
    node_knob_2.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_knob_2.position.set(0.0, 0.188, 0.0);
    node_knob_2.rotation.set(0.0, 0.0, 0.0);
  }
  node_knob_2.userData.sculptComponent = {"id": "knob", "name": "Lid knob", "level": "meso", "role": "knob", "importance": 0.7, "confidence": 0.8, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Turned knob neck rising from the lid apex.", "geometryDescriptor": {"topologyIntent": "Turned knob neck rising from the lid apex.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0, 0.0], [0.01247, -0.00011], [0.02494, -0.00029], [0.03741, -0.00048], [0.04988, -0.00067], [0.06236, -0.00084], [0.07483, -0.00097], [0.0873, -0.00107], [0.09977, -0.00111], [0.11224, -0.00106], [0.12471, -0.00085], [0.13717, -0.00028], [0.14467, 0.00392], [0.13315, 0.00869], [0.12229, 0.01462], [0.11679, 0.02577], [0.11417, 0.03793], [0.11391, 0.05039], [0.11503, 0.06281], [0.11796, 0.0749], [0.12528, 0.08485], [0.13532, 0.09224], [0.14614, 0.09841], [0.1571, 0.10434], [0.16848, 0.10931], [0.17452, 0.11493], [0.16213, 0.11622], [0.14966, 0.11658], [0.13719, 0.11674], [0.12472, 0.11681], [0.11225, 0.11681], [0.09978, 0.11678], [0.0873, 0.11671], [0.07483, 0.11663], [0.06236, 0.11652], [0.04989, 0.11641], [0.03742, 0.11629], [0.02494, 0.11617], [0.01247, 0.11606], [0.0, 0.116]], "segments": 48}}, "parent": "lid", "attachment": {"parentSocket": "lid-apex", "localStart": [0, 0.188, 0], "localEnd": [0, 0.304, 0], "contactType": "embed", "embedDepth": 0.04, "gapTolerance": 0.0, "frame": "parent node frame", "anchor": "lid", "maxOffset": 0.3}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 0.188, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.55, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "lid", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}};
  node_knob_2.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.55, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "lid", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}};
  (nodes["lid"] ?? root).add(node_knob_2);
  nodes["knob"] = node_knob_2;
  const mesh_knob_2Geometry = endpoint_knob_2
    ? new THREE.CylinderGeometry(endpoint_knob_2.endRadius, endpoint_knob_2.baseRadius, endpoint_knob_2.length, 32, 12)
    : buildLatheGeometry({"points": [[0.0, 0.0], [0.01247, -0.00011], [0.02494, -0.00029], [0.03741, -0.00048], [0.04988, -0.00067], [0.06236, -0.00084], [0.07483, -0.00097], [0.0873, -0.00107], [0.09977, -0.00111], [0.11224, -0.00106], [0.12471, -0.00085], [0.13717, -0.00028], [0.14467, 0.00392], [0.13315, 0.00869], [0.12229, 0.01462], [0.11679, 0.02577], [0.11417, 0.03793], [0.11391, 0.05039], [0.11503, 0.06281], [0.11796, 0.0749], [0.12528, 0.08485], [0.13532, 0.09224], [0.14614, 0.09841], [0.1571, 0.10434], [0.16848, 0.10931], [0.17452, 0.11493], [0.16213, 0.11622], [0.14966, 0.11658], [0.13719, 0.11674], [0.12472, 0.11681], [0.11225, 0.11681], [0.09978, 0.11678], [0.0873, 0.11671], [0.07483, 0.11663], [0.06236, 0.11652], [0.04989, 0.11641], [0.03742, 0.11629], [0.02494, 0.11617], [0.01247, 0.11606], [0.0, 0.116]], "segments": 48});
  if (!endpoint_knob_2) {
    mesh_knob_2Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_knob_2 = new THREE.Mesh(
    mesh_knob_2Geometry,
    materialMap["porcelain-glaze"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_knob_2.name = "Lid knob";
  if (endpoint_knob_2) {
    mesh_knob_2.position.copy(endpoint_knob_2.midpoint);
    mesh_knob_2.quaternion.copy(endpoint_knob_2.quaternion);
  }
  mesh_knob_2.castShadow = options.castShadow ?? true;
  mesh_knob_2.receiveShadow = options.receiveShadow ?? true;
  mesh_knob_2.userData.sculptComponent = {"id": "knob", "name": "Lid knob", "level": "meso", "role": "knob", "importance": 0.7, "confidence": 0.8, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Turned knob neck rising from the lid apex.", "geometryDescriptor": {"topologyIntent": "Turned knob neck rising from the lid apex.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0, 0.0], [0.01247, -0.00011], [0.02494, -0.00029], [0.03741, -0.00048], [0.04988, -0.00067], [0.06236, -0.00084], [0.07483, -0.00097], [0.0873, -0.00107], [0.09977, -0.00111], [0.11224, -0.00106], [0.12471, -0.00085], [0.13717, -0.00028], [0.14467, 0.00392], [0.13315, 0.00869], [0.12229, 0.01462], [0.11679, 0.02577], [0.11417, 0.03793], [0.11391, 0.05039], [0.11503, 0.06281], [0.11796, 0.0749], [0.12528, 0.08485], [0.13532, 0.09224], [0.14614, 0.09841], [0.1571, 0.10434], [0.16848, 0.10931], [0.17452, 0.11493], [0.16213, 0.11622], [0.14966, 0.11658], [0.13719, 0.11674], [0.12472, 0.11681], [0.11225, 0.11681], [0.09978, 0.11678], [0.0873, 0.11671], [0.07483, 0.11663], [0.06236, 0.11652], [0.04989, 0.11641], [0.03742, 0.11629], [0.02494, 0.11617], [0.01247, 0.11606], [0.0, 0.116]], "segments": 48}}, "parent": "lid", "attachment": {"parentSocket": "lid-apex", "localStart": [0, 0.188, 0], "localEnd": [0, 0.304, 0], "contactType": "embed", "embedDepth": 0.04, "gapTolerance": 0.0, "frame": "parent node frame", "anchor": "lid", "maxOffset": 0.3}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 0.188, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.55, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "lid", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}};
  node_knob_2.add(mesh_knob_2);
  meshes["knob"] = mesh_knob_2;
  colliders["knob"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"};
  destructionGroups["lid"] ??= [];
  destructionGroups["lid"].push(node_knob_2);

  const endpoint_knob_gold_cap_3 = makeAttachmentEndpoint(null);
  const node_knob_gold_cap_3 = new THREE.Group();
  node_knob_gold_cap_3.name = "Knob gold cap__pivot";
  node_knob_gold_cap_3.scale.set(1, 1, 1);
  if (endpoint_knob_gold_cap_3) {
    node_knob_gold_cap_3.position.copy(endpoint_knob_gold_cap_3.start);
    node_knob_gold_cap_3.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_knob_gold_cap_3.position.set(0.0, 0.112, 0.0);
    node_knob_gold_cap_3.rotation.set(0.0, 0.0, 0.0);
  }
  node_knob_gold_cap_3.userData.sculptComponent = {"id": "knob-gold-cap", "name": "Knob gold cap", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.85, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Domed gold lustre cap on the knob.", "geometryDescriptor": {"topologyIntent": "Domed gold lustre cap on the knob.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0, 0.0], [0.166, 0.0], [0.169, 0.028], [0.163, 0.053], [0.148, 0.073], [0.118, 0.088], [0.065, 0.096], [0, 0.098]], "segments": 48}}, "parent": "knob", "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.85}, "transform": {"position": [0, 0.112, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85, "worldPosition": [0, 1.662, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "lid", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "knob-cap-hotspot", "type": "gloss", "description": "metallic hotspot", "mapsTo": "ovr-gold-polish"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "gold-lustre"}, "materialRegions": [{"regionId": "gold-lustre", "materialId": "gold-lustre", "profileId": "metal.gold", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/02-gold-lustre.png", "bbox": {"x": 640, "y": 124, "width": 72, "height": 24}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0016}}]};
  node_knob_gold_cap_3.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85, "worldPosition": [0, 1.662, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "lid", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}};
  (nodes["knob"] ?? root).add(node_knob_gold_cap_3);
  nodes["knob-gold-cap"] = node_knob_gold_cap_3;
  const mesh_knob_gold_cap_3Geometry = endpoint_knob_gold_cap_3
    ? new THREE.CylinderGeometry(endpoint_knob_gold_cap_3.endRadius, endpoint_knob_gold_cap_3.baseRadius, endpoint_knob_gold_cap_3.length, 32, 12)
    : buildLatheGeometry({"points": [[0, 0.0], [0.166, 0.0], [0.169, 0.028], [0.163, 0.053], [0.148, 0.073], [0.118, 0.088], [0.065, 0.096], [0, 0.098]], "segments": 48});
  if (!endpoint_knob_gold_cap_3) {
    mesh_knob_gold_cap_3Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_knob_gold_cap_3 = new THREE.Mesh(
    mesh_knob_gold_cap_3Geometry,
    materialMap["gold-lustre"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_knob_gold_cap_3.name = "Knob gold cap";
  if (endpoint_knob_gold_cap_3) {
    mesh_knob_gold_cap_3.position.copy(endpoint_knob_gold_cap_3.midpoint);
    mesh_knob_gold_cap_3.quaternion.copy(endpoint_knob_gold_cap_3.quaternion);
  }
  mesh_knob_gold_cap_3.castShadow = options.castShadow ?? true;
  mesh_knob_gold_cap_3.receiveShadow = options.receiveShadow ?? true;
  mesh_knob_gold_cap_3.userData.sculptComponent = {"id": "knob-gold-cap", "name": "Knob gold cap", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.85, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Domed gold lustre cap on the knob.", "geometryDescriptor": {"topologyIntent": "Domed gold lustre cap on the knob.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0, 0.0], [0.166, 0.0], [0.169, 0.028], [0.163, 0.053], [0.148, 0.073], [0.118, 0.088], [0.065, 0.096], [0, 0.098]], "segments": 48}}, "parent": "knob", "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.85}, "transform": {"position": [0, 0.112, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.85, "worldPosition": [0, 1.662, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "lid", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "knob-cap-hotspot", "type": "gloss", "description": "metallic hotspot", "mapsTo": "ovr-gold-polish"}], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "gold-lustre"}, "materialRegions": [{"regionId": "gold-lustre", "materialId": "gold-lustre", "profileId": "metal.gold", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/02-gold-lustre.png", "bbox": {"x": 640, "y": 124, "width": 72, "height": 24}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0016}}]};
  node_knob_gold_cap_3.add(mesh_knob_gold_cap_3);
  meshes["knob-gold-cap"] = mesh_knob_gold_cap_3;
  colliders["knob-gold-cap"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"};
  destructionGroups["lid"] ??= [];
  destructionGroups["lid"].push(node_knob_gold_cap_3);

  const endpoint_spout_4 = makeAttachmentEndpoint(null);
  const node_spout_4 = new THREE.Group();
  node_spout_4.name = "Spout__pivot";
  node_spout_4.scale.set(1, 1, 1);
  if (endpoint_spout_4) {
    node_spout_4.position.copy(endpoint_spout_4.start);
    node_spout_4.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_spout_4.position.set(-0.64, 0.5, 0.0);
    node_spout_4.rotation.set(0.0, 0.0, 0.0);
  }
  node_spout_4.userData.sculptComponent = {"id": "spout", "name": "Spout", "level": "macro", "role": "spout", "importance": 1.0, "confidence": 0.8, "primitive": "tapered-sweep", "topologyClass": "fiber-strand", "topologyRationale": "S-curved tapering spout swept along a fitted centreline (r 0.30 at the root to 0.105 near the tip); tip left open (bore).", "geometryDescriptor": {"topologyIntent": "S-curved tapering spout swept along a fitted centreline (r 0.30 at the root to 0.105 near the tip); tip left open (bore).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.0, 0.0], "rx": 0.22, "rz": 0.22}, {"position": [-0.03062, -0.00849, 0.0], "rx": 0.21562, "rz": 0.21562}, {"position": [-0.07382, -0.0217, 0.0], "rx": 0.20934, "rz": 0.20934}, {"position": [-0.12428, -0.03511, 0.0], "rx": 0.20218, "rz": 0.20218}, {"position": [-0.1767, -0.04418, 0.0], "rx": 0.19518, "rz": 0.19518}, {"position": [-0.22588, -0.04443, 0.0], "rx": 0.18936, "rz": 0.18936}, {"position": [-0.27577, -0.036, 0.0], "rx": 0.1846, "rz": 0.1846}, {"position": [-0.32809, -0.02152, 0.0], "rx": 0.18032, "rz": 0.18032}, {"position": [-0.37924, -0.00215, 0.0], "rx": 0.1764, "rz": 0.1764}, {"position": [-0.42558, 0.02094, 0.0], "rx": 0.1727, "rz": 0.1727}, {"position": [-0.46364, 0.04679, 0.0], "rx": 0.1691, "rz": 0.1691}, {"position": [-0.49427, 0.0783, 0.0], "rx": 0.16555, "rz": 0.16555}, {"position": [-0.51929, 0.11449, 0.0], "rx": 0.16217, "rz": 0.16217}, {"position": [-0.53963, 0.15241, 0.0], "rx": 0.15907, "rz": 0.15907}, {"position": [-0.55618, 0.18906, 0.0], "rx": 0.1564, "rz": 0.1564}, {"position": [-0.56937, 0.22164, 0.0], "rx": 0.15433, "rz": 0.15433}, {"position": [-0.5759, 0.25053, 0.0], "rx": 0.15335, "rz": 0.15335}, {"position": [-0.57777, 0.27769, 0.0], "rx": 0.15296, "rz": 0.15296}, {"position": [-0.57848, 0.30496, 0.0], "rx": 0.15246, "rz": 0.15246}, {"position": [-0.58151, 0.33413, 0.0], "rx": 0.15117, "rz": 0.15117}, {"position": [-0.58964, 0.36705, 0.0], "rx": 0.14837, "rz": 0.14837}, {"position": [-0.60053, 0.40407, 0.0], "rx": 0.14396, "rz": 0.14396}, {"position": [-0.61335, 0.44308, 0.0], "rx": 0.13873, "rz": 0.13873}, {"position": [-0.62784, 0.48172, 0.0], "rx": 0.13353, "rz": 0.13353}, {"position": [-0.64373, 0.51768, 0.0], "rx": 0.12921, "rz": 0.12921}, {"position": [-0.66108, 0.5494, 0.0], "rx": 0.12644, "rz": 0.12644}, {"position": [-0.68056, 0.57859, 0.0], "rx": 0.12474, "rz": 0.12474}, {"position": [-0.7015, 0.60568, 0.0], "rx": 0.12372, "rz": 0.12372}, {"position": [-0.72315, 0.63087, 0.0], "rx": 0.12301, "rz": 0.12301}, {"position": [-0.74471, 0.65437, 0.0], "rx": 0.12224, "rz": 0.12224}, {"position": [-0.76658, 0.67603, 0.0], "rx": 0.12111, "rz": 0.12111}, {"position": [-0.79021, 0.69543, 0.0], "rx": 0.11976, "rz": 0.11976}, {"position": [-0.81382, 0.71314, 0.0], "rx": 0.11853, "rz": 0.11853}, {"position": [-0.83545, 0.72978, 0.0], "rx": 0.11779, "rz": 0.11779}, {"position": [-0.85316, 0.74598, 0.0], "rx": 0.11791, "rz": 0.11791}, {"position": [-0.86597, 0.76258, 0.0], "rx": 0.11935, "rz": 0.11935}, {"position": [-0.87515, 0.77942, 0.0], "rx": 0.122, "rz": 0.122}, {"position": [-0.88161, 0.79517, 0.0], "rx": 0.12514, "rz": 0.12514}, {"position": [-0.88626, 0.80848, 0.0], "rx": 0.12805, "rz": 0.12805}, {"position": [-0.89, 0.818, 0.0], "rx": 0.13, "rz": 0.13}], "radialSegments": 28, "capEnds": false}, "pathSource": "scripts/fit_shape.py fitted centreline, Catmull-Rom sampled to 40 stations"}, "parent": "body", "attachment": {"parentSocket": "spout-root", "localStart": [-0.64, 0.5, 0.0], "localEnd": [-1.53, 1.318, 0.0], "contactType": "embed", "embedDepth": 0.12, "gapTolerance": 0.0, "notes": "root station sits inside the body wall (body r ~0.97 at h 0.56)", "frame": "parent node frame", "anchorNote": "welded to the vessel (embedding contract, not a worn/held item)"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [-0.64, 0.5, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [-0.64, 0.5, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "pour-point", "localPosition": [-0.89, 0.818, 0.0], "worldPosition": [-1.53, 1.318, 0.0], "frame": "component node", "notes": "liquid emitter origin; pour direction along the last spine tangent"}], "collider": {"type": "capsule", "offset": [-0.56, 0.35, 0.0], "scale": [0.25, 0.9, 0.25], "isTrigger": false, "notes": "spout proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "spout", "seamRefs": ["spout-root"], "detachableFragments": [], "breakImpulse": 2.5, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "spout-bore", "type": "hole", "description": "open tip (capEnds false), glaze is double-sided so the bore interior renders"}], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["spout-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["spout-crop"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "porcelain-glaze"}, "materialRegions": [{"regionId": "white-glaze", "materialId": "porcelain-glaze", "profileId": "ceramic.glazed", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/01-white-glaze.png", "bbox": {"x": 215, "y": 330, "width": 85, "height": 90}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.007}}]};
  node_spout_4.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [-0.64, 0.5, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "pour-point", "localPosition": [-0.89, 0.818, 0.0], "worldPosition": [-1.53, 1.318, 0.0], "frame": "component node", "notes": "liquid emitter origin; pour direction along the last spine tangent"}], "collider": {"type": "capsule", "offset": [-0.56, 0.35, 0.0], "scale": [0.25, 0.9, 0.25], "isTrigger": false, "notes": "spout proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "spout", "seamRefs": ["spout-root"], "detachableFragments": [], "breakImpulse": 2.5, "debrisMaterial": "porcelain-glaze"}};
  (nodes["body"] ?? root).add(node_spout_4);
  nodes["spout"] = node_spout_4;
  const mesh_spout_4Geometry = endpoint_spout_4
    ? new THREE.CylinderGeometry(endpoint_spout_4.endRadius, endpoint_spout_4.baseRadius, endpoint_spout_4.length, 32, 12)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.0, 0.0], "rx": 0.22, "rz": 0.22}, {"position": [-0.03062, -0.00849, 0.0], "rx": 0.21562, "rz": 0.21562}, {"position": [-0.07382, -0.0217, 0.0], "rx": 0.20934, "rz": 0.20934}, {"position": [-0.12428, -0.03511, 0.0], "rx": 0.20218, "rz": 0.20218}, {"position": [-0.1767, -0.04418, 0.0], "rx": 0.19518, "rz": 0.19518}, {"position": [-0.22588, -0.04443, 0.0], "rx": 0.18936, "rz": 0.18936}, {"position": [-0.27577, -0.036, 0.0], "rx": 0.1846, "rz": 0.1846}, {"position": [-0.32809, -0.02152, 0.0], "rx": 0.18032, "rz": 0.18032}, {"position": [-0.37924, -0.00215, 0.0], "rx": 0.1764, "rz": 0.1764}, {"position": [-0.42558, 0.02094, 0.0], "rx": 0.1727, "rz": 0.1727}, {"position": [-0.46364, 0.04679, 0.0], "rx": 0.1691, "rz": 0.1691}, {"position": [-0.49427, 0.0783, 0.0], "rx": 0.16555, "rz": 0.16555}, {"position": [-0.51929, 0.11449, 0.0], "rx": 0.16217, "rz": 0.16217}, {"position": [-0.53963, 0.15241, 0.0], "rx": 0.15907, "rz": 0.15907}, {"position": [-0.55618, 0.18906, 0.0], "rx": 0.1564, "rz": 0.1564}, {"position": [-0.56937, 0.22164, 0.0], "rx": 0.15433, "rz": 0.15433}, {"position": [-0.5759, 0.25053, 0.0], "rx": 0.15335, "rz": 0.15335}, {"position": [-0.57777, 0.27769, 0.0], "rx": 0.15296, "rz": 0.15296}, {"position": [-0.57848, 0.30496, 0.0], "rx": 0.15246, "rz": 0.15246}, {"position": [-0.58151, 0.33413, 0.0], "rx": 0.15117, "rz": 0.15117}, {"position": [-0.58964, 0.36705, 0.0], "rx": 0.14837, "rz": 0.14837}, {"position": [-0.60053, 0.40407, 0.0], "rx": 0.14396, "rz": 0.14396}, {"position": [-0.61335, 0.44308, 0.0], "rx": 0.13873, "rz": 0.13873}, {"position": [-0.62784, 0.48172, 0.0], "rx": 0.13353, "rz": 0.13353}, {"position": [-0.64373, 0.51768, 0.0], "rx": 0.12921, "rz": 0.12921}, {"position": [-0.66108, 0.5494, 0.0], "rx": 0.12644, "rz": 0.12644}, {"position": [-0.68056, 0.57859, 0.0], "rx": 0.12474, "rz": 0.12474}, {"position": [-0.7015, 0.60568, 0.0], "rx": 0.12372, "rz": 0.12372}, {"position": [-0.72315, 0.63087, 0.0], "rx": 0.12301, "rz": 0.12301}, {"position": [-0.74471, 0.65437, 0.0], "rx": 0.12224, "rz": 0.12224}, {"position": [-0.76658, 0.67603, 0.0], "rx": 0.12111, "rz": 0.12111}, {"position": [-0.79021, 0.69543, 0.0], "rx": 0.11976, "rz": 0.11976}, {"position": [-0.81382, 0.71314, 0.0], "rx": 0.11853, "rz": 0.11853}, {"position": [-0.83545, 0.72978, 0.0], "rx": 0.11779, "rz": 0.11779}, {"position": [-0.85316, 0.74598, 0.0], "rx": 0.11791, "rz": 0.11791}, {"position": [-0.86597, 0.76258, 0.0], "rx": 0.11935, "rz": 0.11935}, {"position": [-0.87515, 0.77942, 0.0], "rx": 0.122, "rz": 0.122}, {"position": [-0.88161, 0.79517, 0.0], "rx": 0.12514, "rz": 0.12514}, {"position": [-0.88626, 0.80848, 0.0], "rx": 0.12805, "rz": 0.12805}, {"position": [-0.89, 0.818, 0.0], "rx": 0.13, "rz": 0.13}], "radialSegments": 28, "capEnds": false});
  if (!endpoint_spout_4) {
    mesh_spout_4Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_spout_4 = new THREE.Mesh(
    mesh_spout_4Geometry,
    materialMap["porcelain-glaze"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_spout_4.name = "Spout";
  if (endpoint_spout_4) {
    mesh_spout_4.position.copy(endpoint_spout_4.midpoint);
    mesh_spout_4.quaternion.copy(endpoint_spout_4.quaternion);
  }
  mesh_spout_4.castShadow = options.castShadow ?? true;
  mesh_spout_4.receiveShadow = options.receiveShadow ?? true;
  mesh_spout_4.userData.sculptComponent = {"id": "spout", "name": "Spout", "level": "macro", "role": "spout", "importance": 1.0, "confidence": 0.8, "primitive": "tapered-sweep", "topologyClass": "fiber-strand", "topologyRationale": "S-curved tapering spout swept along a fitted centreline (r 0.30 at the root to 0.105 near the tip); tip left open (bore).", "geometryDescriptor": {"topologyIntent": "S-curved tapering spout swept along a fitted centreline (r 0.30 at the root to 0.105 near the tip); tip left open (bore).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.0, 0.0], "rx": 0.22, "rz": 0.22}, {"position": [-0.03062, -0.00849, 0.0], "rx": 0.21562, "rz": 0.21562}, {"position": [-0.07382, -0.0217, 0.0], "rx": 0.20934, "rz": 0.20934}, {"position": [-0.12428, -0.03511, 0.0], "rx": 0.20218, "rz": 0.20218}, {"position": [-0.1767, -0.04418, 0.0], "rx": 0.19518, "rz": 0.19518}, {"position": [-0.22588, -0.04443, 0.0], "rx": 0.18936, "rz": 0.18936}, {"position": [-0.27577, -0.036, 0.0], "rx": 0.1846, "rz": 0.1846}, {"position": [-0.32809, -0.02152, 0.0], "rx": 0.18032, "rz": 0.18032}, {"position": [-0.37924, -0.00215, 0.0], "rx": 0.1764, "rz": 0.1764}, {"position": [-0.42558, 0.02094, 0.0], "rx": 0.1727, "rz": 0.1727}, {"position": [-0.46364, 0.04679, 0.0], "rx": 0.1691, "rz": 0.1691}, {"position": [-0.49427, 0.0783, 0.0], "rx": 0.16555, "rz": 0.16555}, {"position": [-0.51929, 0.11449, 0.0], "rx": 0.16217, "rz": 0.16217}, {"position": [-0.53963, 0.15241, 0.0], "rx": 0.15907, "rz": 0.15907}, {"position": [-0.55618, 0.18906, 0.0], "rx": 0.1564, "rz": 0.1564}, {"position": [-0.56937, 0.22164, 0.0], "rx": 0.15433, "rz": 0.15433}, {"position": [-0.5759, 0.25053, 0.0], "rx": 0.15335, "rz": 0.15335}, {"position": [-0.57777, 0.27769, 0.0], "rx": 0.15296, "rz": 0.15296}, {"position": [-0.57848, 0.30496, 0.0], "rx": 0.15246, "rz": 0.15246}, {"position": [-0.58151, 0.33413, 0.0], "rx": 0.15117, "rz": 0.15117}, {"position": [-0.58964, 0.36705, 0.0], "rx": 0.14837, "rz": 0.14837}, {"position": [-0.60053, 0.40407, 0.0], "rx": 0.14396, "rz": 0.14396}, {"position": [-0.61335, 0.44308, 0.0], "rx": 0.13873, "rz": 0.13873}, {"position": [-0.62784, 0.48172, 0.0], "rx": 0.13353, "rz": 0.13353}, {"position": [-0.64373, 0.51768, 0.0], "rx": 0.12921, "rz": 0.12921}, {"position": [-0.66108, 0.5494, 0.0], "rx": 0.12644, "rz": 0.12644}, {"position": [-0.68056, 0.57859, 0.0], "rx": 0.12474, "rz": 0.12474}, {"position": [-0.7015, 0.60568, 0.0], "rx": 0.12372, "rz": 0.12372}, {"position": [-0.72315, 0.63087, 0.0], "rx": 0.12301, "rz": 0.12301}, {"position": [-0.74471, 0.65437, 0.0], "rx": 0.12224, "rz": 0.12224}, {"position": [-0.76658, 0.67603, 0.0], "rx": 0.12111, "rz": 0.12111}, {"position": [-0.79021, 0.69543, 0.0], "rx": 0.11976, "rz": 0.11976}, {"position": [-0.81382, 0.71314, 0.0], "rx": 0.11853, "rz": 0.11853}, {"position": [-0.83545, 0.72978, 0.0], "rx": 0.11779, "rz": 0.11779}, {"position": [-0.85316, 0.74598, 0.0], "rx": 0.11791, "rz": 0.11791}, {"position": [-0.86597, 0.76258, 0.0], "rx": 0.11935, "rz": 0.11935}, {"position": [-0.87515, 0.77942, 0.0], "rx": 0.122, "rz": 0.122}, {"position": [-0.88161, 0.79517, 0.0], "rx": 0.12514, "rz": 0.12514}, {"position": [-0.88626, 0.80848, 0.0], "rx": 0.12805, "rz": 0.12805}, {"position": [-0.89, 0.818, 0.0], "rx": 0.13, "rz": 0.13}], "radialSegments": 28, "capEnds": false}, "pathSource": "scripts/fit_shape.py fitted centreline, Catmull-Rom sampled to 40 stations"}, "parent": "body", "attachment": {"parentSocket": "spout-root", "localStart": [-0.64, 0.5, 0.0], "localEnd": [-1.53, 1.318, 0.0], "contactType": "embed", "embedDepth": 0.12, "gapTolerance": 0.0, "notes": "root station sits inside the body wall (body r ~0.97 at h 0.56)", "frame": "parent node frame", "anchorNote": "welded to the vessel (embedding contract, not a worn/held item)"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [-0.64, 0.5, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [-0.64, 0.5, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "pour-point", "localPosition": [-0.89, 0.818, 0.0], "worldPosition": [-1.53, 1.318, 0.0], "frame": "component node", "notes": "liquid emitter origin; pour direction along the last spine tangent"}], "collider": {"type": "capsule", "offset": [-0.56, 0.35, 0.0], "scale": [0.25, 0.9, 0.25], "isTrigger": false, "notes": "spout proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "spout", "seamRefs": ["spout-root"], "detachableFragments": [], "breakImpulse": 2.5, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "spout-bore", "type": "hole", "description": "open tip (capEnds false), glaze is double-sided so the bore interior renders"}], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["spout-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["spout-crop"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "porcelain-glaze"}, "materialRegions": [{"regionId": "white-glaze", "materialId": "porcelain-glaze", "profileId": "ceramic.glazed", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/01-white-glaze.png", "bbox": {"x": 215, "y": 330, "width": 85, "height": 90}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.007}}]};
  node_spout_4.add(mesh_spout_4);
  meshes["spout"] = mesh_spout_4;
  colliders["spout"] = {"type": "capsule", "offset": [-0.56, 0.35, 0.0], "scale": [0.25, 0.9, 0.25], "isTrigger": false, "notes": "spout proxy"};
  destructionGroups["spout"] ??= [];
  destructionGroups["spout"].push(node_spout_4);
  const socket_spout_pour_point_0 = new THREE.Object3D();
  socket_spout_pour_point_0.name = "pour-point";
  socket_spout_pour_point_0.position.set(-0.89, 0.818, 0.0);
  socket_spout_pour_point_0.rotation.set(0, 0, 0);
  socket_spout_pour_point_0.userData.socket = {"id": "pour-point", "localPosition": [-0.89, 0.818, 0.0], "worldPosition": [-1.53, 1.318, 0.0], "frame": "component node", "notes": "liquid emitter origin; pour direction along the last spine tangent"};
  node_spout_4.add(socket_spout_pour_point_0);
  sockets["spout:pour-point"] = socket_spout_pour_point_0;

  const endpoint_spout_lip_ring_5 = makeAttachmentEndpoint(null);
  const node_spout_lip_ring_5 = new THREE.Group();
  node_spout_lip_ring_5.name = "Spout lip gold edge__pivot";
  node_spout_lip_ring_5.scale.set(1, 1, 1);
  if (endpoint_spout_lip_ring_5) {
    node_spout_lip_ring_5.position.copy(endpoint_spout_lip_ring_5.start);
    node_spout_lip_ring_5.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_spout_lip_ring_5.position.set(-0.89, 0.818, 0.0);
    node_spout_lip_ring_5.rotation.set(0.0, 0.0, 0.0);
  }
  node_spout_lip_ring_5.userData.sculptComponent = {"id": "spout-lip-ring", "name": "Spout lip gold edge", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.7, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Thin gold band at the spout opening.", "geometryDescriptor": {"topologyIntent": "Thin gold band at the spout opening.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.00168, -0.00428, 0.0], [0.0, 0.0, 0.0]], "radius": 0.137, "radialSegments": 28}}, "parent": "spout", "attachment": {"parentSocket": "spout-surface", "localStart": [-0.88832, 0.81372, 0.0], "localEnd": [-0.89, 0.818, 0.0], "contactType": "embed", "embedDepth": 0.012, "gapTolerance": 0.0, "notes": "bead tube shares the parent centreline, radius 0.012-0.016 larger, so it wraps the parent surface", "frame": "parent node frame", "anchor": "spout", "maxOffset": 1.3}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.7}, "transform": {"position": [-0.89, 0.818, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7, "worldPosition": [-1.53, 1.318, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "spout", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["spout-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["spout-crop"]}};
  node_spout_lip_ring_5.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7, "worldPosition": [-1.53, 1.318, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "spout", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}};
  (nodes["spout"] ?? root).add(node_spout_lip_ring_5);
  nodes["spout-lip-ring"] = node_spout_lip_ring_5;
  const mesh_spout_lip_ring_5Geometry = endpoint_spout_lip_ring_5
    ? new THREE.CylinderGeometry(endpoint_spout_lip_ring_5.endRadius, endpoint_spout_lip_ring_5.baseRadius, endpoint_spout_lip_ring_5.length, 32, 12)
    : buildTubeGeometry({"points": [[0.00168, -0.00428, 0.0], [0.0, 0.0, 0.0]], "radius": 0.137, "radialSegments": 28});
  if (!endpoint_spout_lip_ring_5) {
    mesh_spout_lip_ring_5Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_spout_lip_ring_5 = new THREE.Mesh(
    mesh_spout_lip_ring_5Geometry,
    materialMap["gold-lustre"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_spout_lip_ring_5.name = "Spout lip gold edge";
  if (endpoint_spout_lip_ring_5) {
    mesh_spout_lip_ring_5.position.copy(endpoint_spout_lip_ring_5.midpoint);
    mesh_spout_lip_ring_5.quaternion.copy(endpoint_spout_lip_ring_5.quaternion);
  }
  mesh_spout_lip_ring_5.castShadow = options.castShadow ?? true;
  mesh_spout_lip_ring_5.receiveShadow = options.receiveShadow ?? true;
  mesh_spout_lip_ring_5.userData.sculptComponent = {"id": "spout-lip-ring", "name": "Spout lip gold edge", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.7, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Thin gold band at the spout opening.", "geometryDescriptor": {"topologyIntent": "Thin gold band at the spout opening.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.00168, -0.00428, 0.0], [0.0, 0.0, 0.0]], "radius": 0.137, "radialSegments": 28}}, "parent": "spout", "attachment": {"parentSocket": "spout-surface", "localStart": [-0.88832, 0.81372, 0.0], "localEnd": [-0.89, 0.818, 0.0], "contactType": "embed", "embedDepth": 0.012, "gapTolerance": 0.0, "notes": "bead tube shares the parent centreline, radius 0.012-0.016 larger, so it wraps the parent surface", "frame": "parent node frame", "anchor": "spout", "maxOffset": 1.3}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.7}, "transform": {"position": [-0.89, 0.818, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7, "worldPosition": [-1.53, 1.318, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "spout", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["spout-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["spout-crop"]}};
  mesh_spout_lip_ring_5.userData.explodeWithParent = true;
  node_spout_lip_ring_5.add(mesh_spout_lip_ring_5);
  meshes["spout-lip-ring"] = mesh_spout_lip_ring_5;
  colliders["spout-lip-ring"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"};
  destructionGroups["spout"] ??= [];
  destructionGroups["spout"].push(node_spout_lip_ring_5);

  const endpoint_handle_7 = makeAttachmentEndpoint(null);
  const node_handle_7 = new THREE.Group();
  node_handle_7.name = "Bamboo loop handle__pivot";
  node_handle_7.scale.set(1, 1, 1);
  if (endpoint_handle_7) {
    node_handle_7.position.copy(endpoint_handle_7.start);
    node_handle_7.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_handle_7.position.set(0.76, 1.13, 0.0);
    node_handle_7.rotation.set(0.0, 0.0, 0.0);
  }
  node_handle_7.userData.sculptComponent = {"id": "handle", "name": "Bamboo loop handle", "level": "macro", "role": "handle", "importance": 1.0, "confidence": 0.8, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Closed loop handle swept along a fitted centreline in the spout-handle plane (yaw 0: both roots sit on the body silhouette edge, lesson 10).", "geometryDescriptor": {"topologyIntent": "Closed loop handle swept along a fitted centreline in the spout-handle plane (yaw 0: both roots sit on the body silhouette edge, lesson 10).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.0, 0.0], [0.01857, 0.01986, 0.0], [0.04527, 0.04847, 0.0], [0.07484, 0.0796, 0.0], [0.10207, 0.10702, 0.0], [0.12581, 0.13079, 0.0], [0.14884, 0.15336, 0.0], [0.1729, 0.17425, 0.0], [0.19972, 0.19298, 0.0], [0.22969, 0.21061, 0.0], [0.26174, 0.22653, 0.0], [0.29493, 0.23898, 0.0], [0.32843, 0.24626, 0.0], [0.36356, 0.248, 0.0], [0.39952, 0.24544, 0.0], [0.43436, 0.23983, 0.0], [0.46629, 0.23255, 0.0], [0.49602, 0.22534, 0.0], [0.52393, 0.21541, 0.0], [0.54985, 0.19903, 0.0], [0.57409, 0.17271, 0.0], [0.59898, 0.13692, 0.0], [0.62137, 0.09386, 0.0], [0.63703, 0.04552, 0.0], [0.64213, -0.00759, 0.0], [0.63755, -0.07025, 0.0], [0.62589, -0.13625, 0.0], [0.60958, -0.19766, 0.0], [0.5905, -0.24773, 0.0], [0.56741, -0.2889, 0.0], [0.5399, -0.32581, 0.0], [0.50777, -0.36286, 0.0], [0.46973, -0.404, 0.0], [0.42437, -0.44812, 0.0], [0.3761, -0.49094, 0.0], [0.32973, -0.5281, 0.0], [0.28781, -0.55579, 0.0], [0.24744, -0.57553, 0.0], [0.20893, -0.59191, 0.0], [0.17276, -0.6096, 0.0], [0.1386, -0.63187, 0.0], [0.10601, -0.65638, 0.0], [0.07654, -0.67978, 0.0], [0.0518, -0.69873, 0.0], [0.03265, -0.71156, 0.0], [0.0182, -0.72015, 0.0], [0.0076, -0.72585, 0.0], [0.0, -0.73, 0.0]], "radius": 0.092, "radialSegments": 24, "closed": false}, "sectionNote": "near-constant section (fitted radius 0.092 along the grip); roots embed into the body"}, "parent": "body", "attachment": {"parentSocket": "handle-roots", "localStart": [0.76, 1.13, 0.0], "localEnd": [0.76, 0.4, 0.0], "contactType": "embed", "embedDepth": 0.08, "gapTolerance": 0.0, "notes": "both ends embedded in the body wall; authored stations win over the endpoints", "frame": "parent node frame", "anchorNote": "welded to the vessel (embedding contract, not a worn/held item)"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0.76, 1.13, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0.76, 1.13, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "grip", "localPosition": [0.63, -0.13, 0.0], "worldPosition": [1.39, 1.0, 0], "frame": "component node", "notes": "hand grip centre on the outer loop"}], "collider": {"type": "box", "offset": [0.39, -0.23, 0.0], "scale": [0.6, 1.0, 0.2], "isTrigger": false, "notes": "handle grip proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-roots"], "detachableFragments": [], "breakImpulse": 2.5, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-stain", "type": "stain", "description": "faint brownish smudge near lower root", "mapsTo": "ovr-handle-stain"}], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["handle-crop"]}};
  node_handle_7.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0.76, 1.13, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "grip", "localPosition": [0.63, -0.13, 0.0], "worldPosition": [1.39, 1.0, 0], "frame": "component node", "notes": "hand grip centre on the outer loop"}], "collider": {"type": "box", "offset": [0.39, -0.23, 0.0], "scale": [0.6, 1.0, 0.2], "isTrigger": false, "notes": "handle grip proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-roots"], "detachableFragments": [], "breakImpulse": 2.5, "debrisMaterial": "porcelain-glaze"}};
  (nodes["body"] ?? root).add(node_handle_7);
  nodes["handle"] = node_handle_7;
  const mesh_handle_7Geometry = endpoint_handle_7
    ? new THREE.CylinderGeometry(endpoint_handle_7.endRadius, endpoint_handle_7.baseRadius, endpoint_handle_7.length, 32, 12)
    : buildTubeGeometry({"points": [[0.0, 0.0, 0.0], [0.01857, 0.01986, 0.0], [0.04527, 0.04847, 0.0], [0.07484, 0.0796, 0.0], [0.10207, 0.10702, 0.0], [0.12581, 0.13079, 0.0], [0.14884, 0.15336, 0.0], [0.1729, 0.17425, 0.0], [0.19972, 0.19298, 0.0], [0.22969, 0.21061, 0.0], [0.26174, 0.22653, 0.0], [0.29493, 0.23898, 0.0], [0.32843, 0.24626, 0.0], [0.36356, 0.248, 0.0], [0.39952, 0.24544, 0.0], [0.43436, 0.23983, 0.0], [0.46629, 0.23255, 0.0], [0.49602, 0.22534, 0.0], [0.52393, 0.21541, 0.0], [0.54985, 0.19903, 0.0], [0.57409, 0.17271, 0.0], [0.59898, 0.13692, 0.0], [0.62137, 0.09386, 0.0], [0.63703, 0.04552, 0.0], [0.64213, -0.00759, 0.0], [0.63755, -0.07025, 0.0], [0.62589, -0.13625, 0.0], [0.60958, -0.19766, 0.0], [0.5905, -0.24773, 0.0], [0.56741, -0.2889, 0.0], [0.5399, -0.32581, 0.0], [0.50777, -0.36286, 0.0], [0.46973, -0.404, 0.0], [0.42437, -0.44812, 0.0], [0.3761, -0.49094, 0.0], [0.32973, -0.5281, 0.0], [0.28781, -0.55579, 0.0], [0.24744, -0.57553, 0.0], [0.20893, -0.59191, 0.0], [0.17276, -0.6096, 0.0], [0.1386, -0.63187, 0.0], [0.10601, -0.65638, 0.0], [0.07654, -0.67978, 0.0], [0.0518, -0.69873, 0.0], [0.03265, -0.71156, 0.0], [0.0182, -0.72015, 0.0], [0.0076, -0.72585, 0.0], [0.0, -0.73, 0.0]], "radius": 0.092, "radialSegments": 24, "closed": false});
  if (!endpoint_handle_7) {
    mesh_handle_7Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_handle_7 = new THREE.Mesh(
    mesh_handle_7Geometry,
    materialMap["porcelain-glaze"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_handle_7.name = "Bamboo loop handle";
  if (endpoint_handle_7) {
    mesh_handle_7.position.copy(endpoint_handle_7.midpoint);
    mesh_handle_7.quaternion.copy(endpoint_handle_7.quaternion);
  }
  mesh_handle_7.castShadow = options.castShadow ?? true;
  mesh_handle_7.receiveShadow = options.receiveShadow ?? true;
  mesh_handle_7.userData.sculptComponent = {"id": "handle", "name": "Bamboo loop handle", "level": "macro", "role": "handle", "importance": 1.0, "confidence": 0.8, "primitive": "tube", "topologyClass": "fiber-strand", "topologyRationale": "Closed loop handle swept along a fitted centreline in the spout-handle plane (yaw 0: both roots sit on the body silhouette edge, lesson 10).", "geometryDescriptor": {"topologyIntent": "Closed loop handle swept along a fitted centreline in the spout-handle plane (yaw 0: both roots sit on the body silhouette edge, lesson 10).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[0.0, 0.0, 0.0], [0.01857, 0.01986, 0.0], [0.04527, 0.04847, 0.0], [0.07484, 0.0796, 0.0], [0.10207, 0.10702, 0.0], [0.12581, 0.13079, 0.0], [0.14884, 0.15336, 0.0], [0.1729, 0.17425, 0.0], [0.19972, 0.19298, 0.0], [0.22969, 0.21061, 0.0], [0.26174, 0.22653, 0.0], [0.29493, 0.23898, 0.0], [0.32843, 0.24626, 0.0], [0.36356, 0.248, 0.0], [0.39952, 0.24544, 0.0], [0.43436, 0.23983, 0.0], [0.46629, 0.23255, 0.0], [0.49602, 0.22534, 0.0], [0.52393, 0.21541, 0.0], [0.54985, 0.19903, 0.0], [0.57409, 0.17271, 0.0], [0.59898, 0.13692, 0.0], [0.62137, 0.09386, 0.0], [0.63703, 0.04552, 0.0], [0.64213, -0.00759, 0.0], [0.63755, -0.07025, 0.0], [0.62589, -0.13625, 0.0], [0.60958, -0.19766, 0.0], [0.5905, -0.24773, 0.0], [0.56741, -0.2889, 0.0], [0.5399, -0.32581, 0.0], [0.50777, -0.36286, 0.0], [0.46973, -0.404, 0.0], [0.42437, -0.44812, 0.0], [0.3761, -0.49094, 0.0], [0.32973, -0.5281, 0.0], [0.28781, -0.55579, 0.0], [0.24744, -0.57553, 0.0], [0.20893, -0.59191, 0.0], [0.17276, -0.6096, 0.0], [0.1386, -0.63187, 0.0], [0.10601, -0.65638, 0.0], [0.07654, -0.67978, 0.0], [0.0518, -0.69873, 0.0], [0.03265, -0.71156, 0.0], [0.0182, -0.72015, 0.0], [0.0076, -0.72585, 0.0], [0.0, -0.73, 0.0]], "radius": 0.092, "radialSegments": 24, "closed": false}, "sectionNote": "near-constant section (fitted radius 0.092 along the grip); roots embed into the body"}, "parent": "body", "attachment": {"parentSocket": "handle-roots", "localStart": [0.76, 1.13, 0.0], "localEnd": [0.76, 0.4, 0.0], "contactType": "embed", "embedDepth": 0.08, "gapTolerance": 0.0, "notes": "both ends embedded in the body wall; authored stations win over the endpoints", "frame": "parent node frame", "anchorNote": "welded to the vessel (embedding contract, not a worn/held item)"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0.76, 1.13, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0.76, 1.13, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "grip", "localPosition": [0.63, -0.13, 0.0], "worldPosition": [1.39, 1.0, 0], "frame": "component node", "notes": "hand grip centre on the outer loop"}], "collider": {"type": "box", "offset": [0.39, -0.23, 0.0], "scale": [0.6, 1.0, 0.2], "isTrigger": false, "notes": "handle grip proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "handle", "seamRefs": ["handle-roots"], "detachableFragments": [], "breakImpulse": 2.5, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "handle-stain", "type": "stain", "description": "faint brownish smudge near lower root", "mapsTo": "ovr-handle-stain"}], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "form", "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["handle-crop"]}};
  node_handle_7.add(mesh_handle_7);
  meshes["handle"] = mesh_handle_7;
  colliders["handle"] = {"type": "box", "offset": [0.39, -0.23, 0.0], "scale": [0.6, 1.0, 0.2], "isTrigger": false, "notes": "handle grip proxy"};
  destructionGroups["handle"] ??= [];
  destructionGroups["handle"].push(node_handle_7);
  const socket_handle_grip_0 = new THREE.Object3D();
  socket_handle_grip_0.name = "grip";
  socket_handle_grip_0.position.set(0.63, -0.13, 0.0);
  socket_handle_grip_0.rotation.set(0, 0, 0);
  socket_handle_grip_0.userData.socket = {"id": "grip", "localPosition": [0.63, -0.13, 0.0], "worldPosition": [1.39, 1.0, 0], "frame": "component node", "notes": "hand grip centre on the outer loop"};
  node_handle_7.add(socket_handle_grip_0);
  sockets["handle:grip"] = socket_handle_grip_0;

  const endpoint_handle_node_upper_8 = makeAttachmentEndpoint(null);
  const node_handle_node_upper_8 = new THREE.Group();
  node_handle_node_upper_8.name = "Bamboo node (upper)__pivot";
  node_handle_node_upper_8.scale.set(1, 1, 1);
  if (endpoint_handle_node_upper_8) {
    node_handle_node_upper_8.position.copy(endpoint_handle_node_upper_8.start);
    node_handle_node_upper_8.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_handle_node_upper_8.position.set(0.19972, 0.19298, 0.0);
    node_handle_node_upper_8.rotation.set(0.0, 0.0, 0.0);
  }
  node_handle_node_upper_8.userData.sculptComponent = {"id": "handle-node-upper", "name": "Bamboo node (upper)", "level": "meso", "role": "ridge", "importance": 0.7, "confidence": 0.7, "primitive": "tube", "topologyClass": "surface-relief", "topologyRationale": "Raised bamboo-node ring at the upper root bend.", "geometryDescriptor": {"topologyIntent": "Raised bamboo-node ring at the upper root bend.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[-0.02682, -0.01873, 0.0], [0.0, 0.0, 0.0], [0.02997, 0.01763, 0.0]], "radius": 0.1, "radialSegments": 24}}, "parent": "handle", "attachment": {"parentSocket": "handle-surface", "localStart": [0.1729, 0.17425, 0.0], "localEnd": [0.22969, 0.21061, 0.0], "contactType": "embed", "embedDepth": 0.012, "gapTolerance": 0.0, "notes": "bead tube shares the parent centreline, radius 0.012-0.016 larger, so it wraps the parent surface", "frame": "parent node frame", "anchor": "handle", "maxOffset": 0.9}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.7}, "transform": {"position": [0.19972, 0.19298, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7, "worldPosition": [0.95972, 1.32298, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["handle-crop"]}};
  node_handle_node_upper_8.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7, "worldPosition": [0.95972, 1.32298, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}};
  (nodes["handle"] ?? root).add(node_handle_node_upper_8);
  nodes["handle-node-upper"] = node_handle_node_upper_8;
  const mesh_handle_node_upper_8Geometry = endpoint_handle_node_upper_8
    ? new THREE.CylinderGeometry(endpoint_handle_node_upper_8.endRadius, endpoint_handle_node_upper_8.baseRadius, endpoint_handle_node_upper_8.length, 32, 12)
    : buildTubeGeometry({"points": [[-0.02682, -0.01873, 0.0], [0.0, 0.0, 0.0], [0.02997, 0.01763, 0.0]], "radius": 0.1, "radialSegments": 24});
  if (!endpoint_handle_node_upper_8) {
    mesh_handle_node_upper_8Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_handle_node_upper_8 = new THREE.Mesh(
    mesh_handle_node_upper_8Geometry,
    materialMap["porcelain-glaze"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_handle_node_upper_8.name = "Bamboo node (upper)";
  if (endpoint_handle_node_upper_8) {
    mesh_handle_node_upper_8.position.copy(endpoint_handle_node_upper_8.midpoint);
    mesh_handle_node_upper_8.quaternion.copy(endpoint_handle_node_upper_8.quaternion);
  }
  mesh_handle_node_upper_8.castShadow = options.castShadow ?? true;
  mesh_handle_node_upper_8.receiveShadow = options.receiveShadow ?? true;
  mesh_handle_node_upper_8.userData.sculptComponent = {"id": "handle-node-upper", "name": "Bamboo node (upper)", "level": "meso", "role": "ridge", "importance": 0.7, "confidence": 0.7, "primitive": "tube", "topologyClass": "surface-relief", "topologyRationale": "Raised bamboo-node ring at the upper root bend.", "geometryDescriptor": {"topologyIntent": "Raised bamboo-node ring at the upper root bend.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[-0.02682, -0.01873, 0.0], [0.0, 0.0, 0.0], [0.02997, 0.01763, 0.0]], "radius": 0.1, "radialSegments": 24}}, "parent": "handle", "attachment": {"parentSocket": "handle-surface", "localStart": [0.1729, 0.17425, 0.0], "localEnd": [0.22969, 0.21061, 0.0], "contactType": "embed", "embedDepth": 0.012, "gapTolerance": 0.0, "notes": "bead tube shares the parent centreline, radius 0.012-0.016 larger, so it wraps the parent surface", "frame": "parent node frame", "anchor": "handle", "maxOffset": 0.9}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.7}, "transform": {"position": [0.19972, 0.19298, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7, "worldPosition": [0.95972, 1.32298, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["handle-crop"]}};
  mesh_handle_node_upper_8.userData.explodeWithParent = true;
  node_handle_node_upper_8.add(mesh_handle_node_upper_8);
  meshes["handle-node-upper"] = mesh_handle_node_upper_8;
  colliders["handle-node-upper"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"};
  destructionGroups["handle"] ??= [];
  destructionGroups["handle"].push(node_handle_node_upper_8);

  const endpoint_handle_node_lower_9 = makeAttachmentEndpoint(null);
  const node_handle_node_lower_9 = new THREE.Group();
  node_handle_node_lower_9.name = "Bamboo node (outer bend)__pivot";
  node_handle_node_lower_9.scale.set(1, 1, 1);
  if (endpoint_handle_node_lower_9) {
    node_handle_node_lower_9.position.copy(endpoint_handle_node_lower_9.start);
    node_handle_node_lower_9.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_handle_node_lower_9.position.set(0.62137, 0.09386, 0.0);
    node_handle_node_lower_9.rotation.set(0.0, 0.0, 0.0);
  }
  node_handle_node_lower_9.userData.sculptComponent = {"id": "handle-node-lower", "name": "Bamboo node (outer bend)", "level": "meso", "role": "ridge", "importance": 0.7, "confidence": 0.65, "primitive": "tube", "topologyClass": "surface-relief", "topologyRationale": "Raised bamboo-node ring on the outer descent.", "geometryDescriptor": {"topologyIntent": "Raised bamboo-node ring on the outer descent.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[-0.02239, 0.04306, 0.0], [0.0, 0.0, 0.0], [0.01566, -0.04834, 0.0]], "radius": 0.1, "radialSegments": 24}}, "parent": "handle", "attachment": {"parentSocket": "handle-surface", "localStart": [0.59898, 0.13692, 0.0], "localEnd": [0.63703, 0.04552, 0.0], "contactType": "embed", "embedDepth": 0.012, "gapTolerance": 0.0, "notes": "bead tube shares the parent centreline, radius 0.012-0.016 larger, so it wraps the parent surface", "frame": "parent node frame", "anchor": "handle", "maxOffset": 0.9}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.65}, "transform": {"position": [0.62137, 0.09386, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.65, "worldPosition": [1.38137, 1.22386, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["handle-crop"]}};
  node_handle_node_lower_9.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.65, "worldPosition": [1.38137, 1.22386, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}};
  (nodes["handle"] ?? root).add(node_handle_node_lower_9);
  nodes["handle-node-lower"] = node_handle_node_lower_9;
  const mesh_handle_node_lower_9Geometry = endpoint_handle_node_lower_9
    ? new THREE.CylinderGeometry(endpoint_handle_node_lower_9.endRadius, endpoint_handle_node_lower_9.baseRadius, endpoint_handle_node_lower_9.length, 32, 12)
    : buildTubeGeometry({"points": [[-0.02239, 0.04306, 0.0], [0.0, 0.0, 0.0], [0.01566, -0.04834, 0.0]], "radius": 0.1, "radialSegments": 24});
  if (!endpoint_handle_node_lower_9) {
    mesh_handle_node_lower_9Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_handle_node_lower_9 = new THREE.Mesh(
    mesh_handle_node_lower_9Geometry,
    materialMap["porcelain-glaze"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_handle_node_lower_9.name = "Bamboo node (outer bend)";
  if (endpoint_handle_node_lower_9) {
    mesh_handle_node_lower_9.position.copy(endpoint_handle_node_lower_9.midpoint);
    mesh_handle_node_lower_9.quaternion.copy(endpoint_handle_node_lower_9.quaternion);
  }
  mesh_handle_node_lower_9.castShadow = options.castShadow ?? true;
  mesh_handle_node_lower_9.receiveShadow = options.receiveShadow ?? true;
  mesh_handle_node_lower_9.userData.sculptComponent = {"id": "handle-node-lower", "name": "Bamboo node (outer bend)", "level": "meso", "role": "ridge", "importance": 0.7, "confidence": 0.65, "primitive": "tube", "topologyClass": "surface-relief", "topologyRationale": "Raised bamboo-node ring on the outer descent.", "geometryDescriptor": {"topologyIntent": "Raised bamboo-node ring on the outer descent.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "tubePath": {"points": [[-0.02239, 0.04306, 0.0], [0.0, 0.0, 0.0], [0.01566, -0.04834, 0.0]], "radius": 0.1, "radialSegments": 24}}, "parent": "handle", "attachment": {"parentSocket": "handle-surface", "localStart": [0.59898, 0.13692, 0.0], "localEnd": [0.63703, 0.04552, 0.0], "contactType": "embed", "embedDepth": 0.012, "gapTolerance": 0.0, "notes": "bead tube shares the parent centreline, radius 0.012-0.016 larger, so it wraps the parent surface", "frame": "parent node frame", "anchor": "handle", "maxOffset": 0.9}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.65}, "transform": {"position": [0.62137, 0.09386, 0.0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.65, "worldPosition": [1.38137, 1.22386, 0.0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "porcelain-glaze"}}, "material": "porcelain-glaze", "materialLayers": ["porcelain-glaze"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.1, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: mirror-smooth glaze (window reflections in spout-crop)", "occlusionPattern": "contact shadow from the key light", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(238, 236, 230, 1.0)", "secondaryAlbedo": "rgba(214, 216, 214, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.95, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["handle-crop"]}};
  mesh_handle_node_lower_9.userData.explodeWithParent = true;
  node_handle_node_lower_9.add(mesh_handle_node_lower_9);
  meshes["handle-node-lower"] = mesh_handle_node_lower_9;
  colliders["handle-node-lower"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"};
  destructionGroups["handle"] ??= [];
  destructionGroups["handle"].push(node_handle_node_lower_9);

  const endpoint_collar_gold_line_12 = makeAttachmentEndpoint(null);
  const node_collar_gold_line_12 = new THREE.Group();
  node_collar_gold_line_12.name = "Collar gold line__pivot";
  node_collar_gold_line_12.scale.set(1, 1, 1);
  if (endpoint_collar_gold_line_12) {
    node_collar_gold_line_12.position.copy(endpoint_collar_gold_line_12.start);
    node_collar_gold_line_12.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_collar_gold_line_12.position.set(0.0, 1.34, 0.0);
    node_collar_gold_line_12.rotation.set(0.0, 0.0, 0.0);
  }
  node_collar_gold_line_12.userData.sculptComponent = {"id": "collar-gold-line", "name": "Collar gold line", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.8, "primitive": "lathe", "topologyClass": "surface-relief", "topologyRationale": "Thin gold bead around the collar base above the harlequin band.", "geometryDescriptor": {"topologyIntent": "Thin gold bead around the collar base above the harlequin band.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.6572289204025618, -0.007], [0.6652289204025618, -0.0028], [0.6652289204025618, 0.0028], [0.6572289204025618, 0.007]], "segments": 128}}, "parent": "body", "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 1.34, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.34, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "vessel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}};
  node_collar_gold_line_12.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.34, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "vessel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}};
  (nodes["body"] ?? root).add(node_collar_gold_line_12);
  nodes["collar-gold-line"] = node_collar_gold_line_12;
  const mesh_collar_gold_line_12Geometry = endpoint_collar_gold_line_12
    ? new THREE.CylinderGeometry(endpoint_collar_gold_line_12.endRadius, endpoint_collar_gold_line_12.baseRadius, endpoint_collar_gold_line_12.length, 32, 12)
    : buildLatheGeometry({"points": [[0.6572289204025618, -0.007], [0.6652289204025618, -0.0028], [0.6652289204025618, 0.0028], [0.6572289204025618, 0.007]], "segments": 128});
  if (!endpoint_collar_gold_line_12) {
    mesh_collar_gold_line_12Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_collar_gold_line_12 = new THREE.Mesh(
    mesh_collar_gold_line_12Geometry,
    materialMap["gold-lustre"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_collar_gold_line_12.name = "Collar gold line";
  if (endpoint_collar_gold_line_12) {
    mesh_collar_gold_line_12.position.copy(endpoint_collar_gold_line_12.midpoint);
    mesh_collar_gold_line_12.quaternion.copy(endpoint_collar_gold_line_12.quaternion);
  }
  mesh_collar_gold_line_12.castShadow = options.castShadow ?? true;
  mesh_collar_gold_line_12.receiveShadow = options.receiveShadow ?? true;
  mesh_collar_gold_line_12.userData.sculptComponent = {"id": "collar-gold-line", "name": "Collar gold line", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.8, "primitive": "lathe", "topologyClass": "surface-relief", "topologyRationale": "Thin gold bead around the collar base above the harlequin band.", "geometryDescriptor": {"topologyIntent": "Thin gold bead around the collar base above the harlequin band.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.6572289204025618, -0.007], [0.6652289204025618, -0.0028], [0.6652289204025618, 0.0028], [0.6572289204025618, 0.007]], "segments": 128}}, "parent": "body", "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 1.34, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8, "worldPosition": [0, 1.34, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "vessel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}};
  mesh_collar_gold_line_12.userData.explodeWithParent = true;
  node_collar_gold_line_12.add(mesh_collar_gold_line_12);
  meshes["collar-gold-line"] = mesh_collar_gold_line_12;
  colliders["collar-gold-line"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"};
  destructionGroups["vessel"] ??= [];
  destructionGroups["vessel"].push(node_collar_gold_line_12);

  const endpoint_rim_gold_ring_13 = makeAttachmentEndpoint(null);
  const node_rim_gold_ring_13 = new THREE.Group();
  node_rim_gold_ring_13.name = "Gallery gold line__pivot";
  node_rim_gold_ring_13.scale.set(1, 1, 1);
  if (endpoint_rim_gold_ring_13) {
    node_rim_gold_ring_13.position.copy(endpoint_rim_gold_ring_13.start);
    node_rim_gold_ring_13.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_rim_gold_ring_13.position.set(0.0, 1.425, 0.0);
    node_rim_gold_ring_13.rotation.set(0.0, 0.0, 0.0);
  }
  node_rim_gold_ring_13.userData.sculptComponent = {"id": "rim-gold-ring", "name": "Gallery gold line", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.6, "primitive": "lathe", "topologyClass": "surface-relief", "topologyRationale": "Gold line on the inner collar wall (visible on the far side above the lid).", "geometryDescriptor": {"topologyIntent": "Gold line on the inner collar wall (visible on the far side above the lid).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.609236012832264, 0.008], [0.605236012832264, 0.0032], [0.605236012832264, -0.0032], [0.609236012832264, -0.008]], "segments": 128}}, "parent": "body", "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.6}, "transform": {"position": [0, 1.425, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6, "worldPosition": [0, 1.425, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "vessel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}};
  node_rim_gold_ring_13.userData.actionProfile = {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6, "worldPosition": [0, 1.425, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "vessel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}};
  (nodes["body"] ?? root).add(node_rim_gold_ring_13);
  nodes["rim-gold-ring"] = node_rim_gold_ring_13;
  const mesh_rim_gold_ring_13Geometry = endpoint_rim_gold_ring_13
    ? new THREE.CylinderGeometry(endpoint_rim_gold_ring_13.endRadius, endpoint_rim_gold_ring_13.baseRadius, endpoint_rim_gold_ring_13.length, 32, 12)
    : buildLatheGeometry({"points": [[0.609236012832264, 0.008], [0.605236012832264, 0.0032], [0.605236012832264, -0.0032], [0.609236012832264, -0.008]], "segments": 128});
  if (!endpoint_rim_gold_ring_13) {
    mesh_rim_gold_ring_13Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_rim_gold_ring_13 = new THREE.Mesh(
    mesh_rim_gold_ring_13Geometry,
    materialMap["gold-lustre"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_rim_gold_ring_13.name = "Gallery gold line";
  if (endpoint_rim_gold_ring_13) {
    mesh_rim_gold_ring_13.position.copy(endpoint_rim_gold_ring_13.midpoint);
    mesh_rim_gold_ring_13.quaternion.copy(endpoint_rim_gold_ring_13.quaternion);
  }
  mesh_rim_gold_ring_13.castShadow = options.castShadow ?? true;
  mesh_rim_gold_ring_13.receiveShadow = options.receiveShadow ?? true;
  mesh_rim_gold_ring_13.userData.sculptComponent = {"id": "rim-gold-ring", "name": "Gallery gold line", "level": "meso", "role": "trim", "importance": 0.7, "confidence": 0.6, "primitive": "lathe", "topologyClass": "surface-relief", "topologyRationale": "Gold line on the inner collar wall (visible on the far side above the lid).", "geometryDescriptor": {"topologyIntent": "Gold line on the inner collar wall (visible on the far side above the lid).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.609236012832264, 0.008], [0.605236012832264, 0.0032], [0.605236012832264, -0.0032], [0.609236012832264, -0.008]], "segments": 128}}, "parent": "body", "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.6}, "transform": {"position": [0, 1.425, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "semantic-socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.6, "worldPosition": [0, 1.425, 0]}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "vessel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["lid-crop"], "details": [], "fidelityTier": "form", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["lid-crop"]}};
  mesh_rim_gold_ring_13.userData.explodeWithParent = true;
  node_rim_gold_ring_13.add(mesh_rim_gold_ring_13);
  meshes["rim-gold-ring"] = mesh_rim_gold_ring_13;
  colliders["rim-gold-ring"] = {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"};
  destructionGroups["vessel"] ??= [];
  destructionGroups["vessel"].push(node_rim_gold_ring_13);
  // repetition system "handle-bamboo-nodes" describes 2 parts that are already built individually; not instanced.
  // repetition system "harlequin-diamonds" describes 1 parts that are already built individually; not instanced.

  // decal "spout-flower-decal": reference pixels projected onto "spout"
  root.updateMatrixWorld(true);
  const decal_spout_flower_decal_0Target = meshes["spout"];
  if (decal_spout_flower_decal_0Target) {
    const decal_spout_flower_decal_0Geometry = new DecalGeometry(
      decal_spout_flower_decal_0Target,
      new THREE.Vector3(-1.1268, 0.5298, 0.18),
      new THREE.Euler(-0.2618, 0.0, 0.0),
      new THREE.Vector3(0.4427, 0.3352, 0.6),
    );
    decal_spout_flower_decal_0Geometry.applyMatrix4(new THREE.Matrix4().copy(decal_spout_flower_decal_0Target.matrixWorld).invert());
    const decal_spout_flower_decal_0Material = ((materialMap["enamel-decal"] as THREE.MeshStandardMaterial | undefined) ?? new THREE.MeshStandardMaterial()).clone();
    const decal_spout_flower_decal_0Texture = new THREE.TextureLoader().load("models/porcelain-teapot/assets/textures/spout_flower_decal.png");
    decal_spout_flower_decal_0Texture.colorSpace = THREE.SRGBColorSpace;
    decal_spout_flower_decal_0Material.map = decal_spout_flower_decal_0Texture;
    decal_spout_flower_decal_0Material.color.set(0xffffff);
    decal_spout_flower_decal_0Material.transparent = true;
    decal_spout_flower_decal_0Material.alphaTest = 0.4;
    decal_spout_flower_decal_0Material.depthWrite = false;
    decal_spout_flower_decal_0Material.polygonOffset = true;
    decal_spout_flower_decal_0Material.polygonOffsetFactor = -4;
    const decal_spout_flower_decal_0 = new THREE.Mesh(decal_spout_flower_decal_0Geometry, decal_spout_flower_decal_0Material);
    decal_spout_flower_decal_0.name = "Spout flower spray";
    decal_spout_flower_decal_0.renderOrder = 1;
    decal_spout_flower_decal_0.castShadow = false;
    decal_spout_flower_decal_0.receiveShadow = options.receiveShadow ?? true;
    decal_spout_flower_decal_0.userData.sculptComponent = {"id": "spout-flower-decal", "name": "Spout flower spray", "level": "micro", "role": "decal", "importance": 0.7, "confidence": 0.8, "primitive": "decal", "topologyClass": "material-only", "topologyRationale": "Reference pixels (red flower + teal leaves) projected onto the spout with DecalGeometry.", "geometryDescriptor": {"topologyIntent": "Reference pixels (red flower + teal leaves) projected onto the spout with DecalGeometry.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decal": {"target": "spout", "position": [-1.1268, 0.5298, 0.18], "orientation": [-0.2618, 0, 0], "size": [0.4427, 0.3352, 0.6], "texture": "models/porcelain-teapot/assets/textures/spout_flower_decal.png", "alphaTest": 0.4}}, "parent": "spout", "attachment": {"parentSocket": "spout-surface", "localStart": [-1.1268, 0.5298, 0.18], "localEnd": [-1.1268, 0.5298, -0.12], "contactType": "overlap", "embedDepth": 0.001, "overlap": 0.001, "gapTolerance": 0.002, "notes": "DecalGeometry conforms to the target surface; localEnd is the far face of the projector box", "frame": "root space (decal projector)", "anchorNote": "decal geometry is clipped to the target surface; cannot drift"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "authored", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "spout", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "enamel-decal"}}, "material": "enamel-decal", "materialLayers": ["enamel-decal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.14, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none", "occlusionPattern": "none", "edgeWearPattern": "none", "notes": "decal carrier"}, "evidenceRefs": ["spout-crop"], "details": [], "fidelityTier": "material", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(156, 94, 86, 1.0)", "secondaryAlbedo": "rgba(159, 61, 46, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["spout-crop"]}, "uvContract": {"status": "unwrapped", "strategy": "generated procedural coordinates", "materialId": "enamel-decal"}, "materialRegions": [{"regionId": "spout-flower", "materialId": "enamel-decal", "profileId": "ceramic.glazed", "crop": {"path": "/mnt/c/repos/img2three-runs/porcelain-teapot-20261117-162313/material-evidence/regions/04-spout-flower.png", "bbox": {"x": 258, "y": 475, "width": 62, "height": 60}, "sourceWidth": 1280, "sourceHeight": 853, "loaderWarnings": [], "coverage": 0.0034}}]};
    decal_spout_flower_decal_0.userData.explodeWithParent = true;
    decal_spout_flower_decal_0Target.add(decal_spout_flower_decal_0);
    meshes["spout-flower-decal"] = decal_spout_flower_decal_0;
    nodes["spout-flower-decal"] = decal_spout_flower_decal_0;
  }

  // decal "handle-gold-band": reference pixels projected onto "handle"
  root.updateMatrixWorld(true);
  const decal_handle_gold_band_1Target = meshes["handle"];
  if (decal_handle_gold_band_1Target) {
    const decal_handle_gold_band_1Geometry = new DecalGeometry(
      decal_handle_gold_band_1Target,
      new THREE.Vector3(1.175, 1.39, 0.0),
      new THREE.Euler(-1.5708, 0.0, 0.0),
      new THREE.Vector3(0.4, 0.32, 0.18),
    );
    decal_handle_gold_band_1Geometry.applyMatrix4(new THREE.Matrix4().copy(decal_handle_gold_band_1Target.matrixWorld).invert());
    const decal_handle_gold_band_1Material = ((materialMap["gold-lustre"] as THREE.MeshStandardMaterial | undefined) ?? new THREE.MeshStandardMaterial()).clone();
    const decal_handle_gold_band_1Texture = new THREE.TextureLoader().load("models/porcelain-teapot/assets/textures/gold_solid.png");
    decal_handle_gold_band_1Texture.colorSpace = THREE.SRGBColorSpace;
    decal_handle_gold_band_1Material.map = decal_handle_gold_band_1Texture;
    decal_handle_gold_band_1Material.color.set(0xffffff);
    decal_handle_gold_band_1Material.transparent = true;
    decal_handle_gold_band_1Material.alphaTest = 0.5;
    decal_handle_gold_band_1Material.depthWrite = false;
    decal_handle_gold_band_1Material.polygonOffset = true;
    decal_handle_gold_band_1Material.polygonOffsetFactor = -4;
    const decal_handle_gold_band_1 = new THREE.Mesh(decal_handle_gold_band_1Geometry, decal_handle_gold_band_1Material);
    decal_handle_gold_band_1.name = "Handle gold band";
    decal_handle_gold_band_1.renderOrder = 1;
    decal_handle_gold_band_1.castShadow = false;
    decal_handle_gold_band_1.receiveShadow = options.receiveShadow ?? true;
    decal_handle_gold_band_1.userData.sculptComponent = {"id": "handle-gold-band", "name": "Handle gold band", "level": "meso", "role": "decal", "importance": 0.7, "confidence": 0.8, "primitive": "decal", "topologyClass": "material-only", "topologyRationale": "Gold lustre painted on the upper half of the handle top (x 0.95-1.36), projected from above so the underside stays white.", "geometryDescriptor": {"topologyIntent": "Gold lustre painted on the upper half of the handle top (x 0.95-1.36), projected from above so the underside stays white.", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decal": {"target": "handle", "position": [1.175, 1.39, 0], "orientation": [-1.5708, 0, 0], "size": [0.4, 0.32, 0.18], "texture": "models/porcelain-teapot/assets/textures/gold_solid.png", "alphaTest": 0.5}}, "parent": "handle", "attachment": {"parentSocket": "handle-surface", "localStart": [1.175, 1.39, 0], "localEnd": [1.175, 1.2999999999999998, 0], "contactType": "overlap", "embedDepth": 0.001, "overlap": 0.001, "gapTolerance": 0.002, "notes": "DecalGeometry conforms to the target surface; localEnd is the far face of the projector box", "frame": "root space (decal projector)", "anchorNote": "decal geometry is clipped to the target surface; cannot drift"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.8}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "authored", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.8}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "gold-lustre"}}, "material": "gold-lustre", "materialLayers": ["gold-lustre"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.3, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none: burnished lustre", "occlusionPattern": "none", "edgeWearPattern": "none observed", "notes": "textureless by evidence"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "material", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(78, 41, 18, 1.0)", "secondaryAlbedo": "rgba(41, 25, 18, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.85, "source": "agent reading of reference crops (CLAUDE.md lesson 4: analyze_texture metal/dielectric class not trusted)", "evidenceRefs": ["handle-crop"]}};
    decal_handle_gold_band_1.userData.explodeWithParent = true;
    decal_handle_gold_band_1Target.add(decal_handle_gold_band_1);
    meshes["handle-gold-band"] = decal_handle_gold_band_1;
    nodes["handle-gold-band"] = decal_handle_gold_band_1;
  }

  // decal "handle-stain-decal": reference pixels projected onto "handle"
  root.updateMatrixWorld(true);
  const decal_handle_stain_decal_2Target = meshes["handle"];
  if (decal_handle_stain_decal_2Target) {
    const decal_handle_stain_decal_2Geometry = new DecalGeometry(
      decal_handle_stain_decal_2Target,
      new THREE.Vector3(1.0748, 0.706, 0.05),
      new THREE.Euler(-0.2618, 0.0, 0.0),
      new THREE.Vector3(0.1915, 0.1915, 0.25),
    );
    decal_handle_stain_decal_2Geometry.applyMatrix4(new THREE.Matrix4().copy(decal_handle_stain_decal_2Target.matrixWorld).invert());
    const decal_handle_stain_decal_2Material = ((materialMap["enamel-decal"] as THREE.MeshStandardMaterial | undefined) ?? new THREE.MeshStandardMaterial()).clone();
    const decal_handle_stain_decal_2Texture = new THREE.TextureLoader().load("models/porcelain-teapot/assets/textures/handle_stain_decal.png");
    decal_handle_stain_decal_2Texture.colorSpace = THREE.SRGBColorSpace;
    decal_handle_stain_decal_2Material.map = decal_handle_stain_decal_2Texture;
    decal_handle_stain_decal_2Material.color.set(0xffffff);
    decal_handle_stain_decal_2Material.transparent = true;
    decal_handle_stain_decal_2Material.alphaTest = 0.05;
    decal_handle_stain_decal_2Material.depthWrite = false;
    decal_handle_stain_decal_2Material.polygonOffset = true;
    decal_handle_stain_decal_2Material.polygonOffsetFactor = -4;
    const decal_handle_stain_decal_2 = new THREE.Mesh(decal_handle_stain_decal_2Geometry, decal_handle_stain_decal_2Material);
    decal_handle_stain_decal_2.name = "Handle glaze stain";
    decal_handle_stain_decal_2.renderOrder = 1;
    decal_handle_stain_decal_2.castShadow = false;
    decal_handle_stain_decal_2.receiveShadow = options.receiveShadow ?? true;
    decal_handle_stain_decal_2.userData.sculptComponent = {"id": "handle-stain-decal", "name": "Handle glaze stain", "level": "micro", "role": "decal", "importance": 0.7, "confidence": 0.4, "primitive": "decal", "topologyClass": "material-only", "topologyRationale": "Faint warm-brown discolouration at the lower handle root, reference pixels projected (surface pass).", "geometryDescriptor": {"topologyIntent": "Faint warm-brown discolouration at the lower handle root, reference pixels projected (surface pass).", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "decal": {"target": "handle", "position": [1.0748, 0.706, 0.05], "orientation": [-0.2618, 0, 0], "size": [0.1915, 0.1915, 0.25], "texture": "models/porcelain-teapot/assets/textures/handle_stain_decal.png", "alphaTest": 0.05}}, "parent": "handle", "attachment": {"parentSocket": "handle-surface", "localStart": [1.0748, 0.706, 0.05], "localEnd": [1.0748, 0.706, -0.075], "contactType": "overlap", "embedDepth": 0.001, "overlap": 0.001, "gapTolerance": 0.002, "notes": "DecalGeometry conforms to the target surface; localEnd is the far face of the projector box", "frame": "root space (decal projector)", "anchorNote": "decal geometry is clipped to the target surface; cannot drift"}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "world (authored geometry, not scaled)", "confidence": 0.4}, "transform": {"position": [0, 0, 0], "rotation": [0, 0, 0], "scale": [1, 1, 1]}, "actionProfile": {"animationRole": "static-part", "pivot": {"mode": "authored", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.4}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [1, 1, 1], "isTrigger": false, "notes": "surface detail; picks resolve to the parent part"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "handle", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "enamel-decal"}}, "material": "enamel-decal", "materialLayers": ["enamel-decal"], "deformations": [], "joints": [], "seams": [], "localFeatures": [], "surfaceDetail": {"macroRoughness": 0.14, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "none", "occlusionPattern": "none", "edgeWearPattern": "none", "notes": "decal carrier"}, "evidenceRefs": ["handle-crop"], "details": [], "fidelityTier": "surface", "explodeWithParent": true, "colorMaterialRecipe": {"dominantAlbedo": "rgba(196, 170, 150, 1.0)", "secondaryAlbedo": "rgba(214, 200, 190, 1.0)", "materialClass": "ceramic", "materialClassConfidence": 0.85, "source": "reference handle-crop pixels (warm deviation from glaze)", "evidenceRefs": ["handle-crop"]}};
    decal_handle_stain_decal_2.userData.explodeWithParent = true;
    decal_handle_stain_decal_2Target.add(decal_handle_stain_decal_2);
    meshes["handle-stain-decal"] = decal_handle_stain_decal_2;
    nodes["handle-stain-decal"] = decal_handle_stain_decal_2;
  }

  root.userData.sculptRuntime = attachPartRuntime(root, { nodes, meshes, sockets, colliders, destructionGroups });
  root.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  root.userData.actionReadiness = {
    note: 'Use root.userData.sculptRuntime.nodes for transforms, sockets for attachments, colliders for physics proxies, and destructionGroups for breakable sets.',
  };
  return root;
}

export function createPorcelainTeapotLookDevLights(
  mode: 'neutral' | 'grazing' | 'reference' = 'neutral',
): THREE.Group {
  const lights = new THREE.Group();
  lights.name = "Porcelain teapot look-dev lights";
  const photoRig: Array<Record<string, any>> = [{"type": "directional", "position": [-3.0, 8.0, 7.0], "color": "#fff8f0", "intensity": 1.7, "castShadow": true}, {"type": "directional", "position": [5.0, 3.0, 4.0], "color": "#e8eeff", "intensity": 1.3, "castShadow": false}, {"type": "directional", "position": [0.0, -3.0, 4.0], "color": "#c8d0e8", "intensity": 0.6, "castShadow": false}, {"type": "directional", "position": [4.0, 3.0, -5.0], "color": "#f0f2ff", "intensity": 0.5, "castShadow": false}, {"type": "hemisphere", "skyColor": "#eef2ff", "groundColor": "#a8b4cc", "intensity": 1.1}];
  if (mode === 'reference' && photoRig.length > 0) {
    for (const l of photoRig) {
      if (l.type === 'hemisphere') {
        lights.add(new THREE.HemisphereLight(l.skyColor, l.groundColor, l.intensity));
        continue;
      }
      const d = new THREE.DirectionalLight(l.color, l.intensity);
      d.position.set(l.position[0], l.position[1], l.position[2]);
      if (l.castShadow) {
        d.castShadow = true;
        d.shadow.mapSize.set(4096, 4096);
        d.shadow.bias = -0.00025;
        d.shadow.normalBias = 0.018;
        d.shadow.radius = 7;
        d.shadow.camera.near = 0.5;
        d.shadow.camera.far = 30;
        d.shadow.camera.left = -2.6;
        d.shadow.camera.right = 2.6;
        d.shadow.camera.top = 2.6;
        d.shadow.camera.bottom = -2.6;
        d.shadow.camera.updateProjectionMatrix();
      }
      lights.add(d);
    }
    lights.userData.reviewMode = mode;
    lights.userData.rigSource = 'lightingFromPhoto';
    lights.userData.lightingFromPhoto = [{"id": "key", "type": "directional", "position": [-3.0, 8.0, 7.0], "intensity": 1.7, "color": "#fff8f0", "castShadow": true, "evidence": "window highlights on the left/front of spout and handle; soft cast shadow right-back of the body", "shadow": {"softness": "PCF soft, radius 6, 2048 map (viewer)", "contactShadow": "PCF shadow on the table plane gives the contact/ground shadow under the foot"}, "notes": "PCF soft shadow map gives the ground shadow / contact shadow on the table, as in the photo (shadow right-back of the body)"}, {"id": "fill", "type": "directional", "position": [5.0, 3.0, 4.0], "intensity": 1.3, "color": "#e8eeff", "evidence": "right flank in the photo is dark but not black (region mean 60/55/48); fill swept 0.6/1.0/1.3 -> 49/53/57"}, {"id": "bounce", "type": "directional", "position": [0.0, -3.0, 4.0], "intensity": 0.6, "color": "#c8d0e8", "evidence": "lower band lit by bounce from the lavender cloth (photo 106/95/87 vs 81/72/67 without it); no shadow"}, {"id": "rim", "type": "directional", "position": [4.0, 3.0, -5.0], "intensity": 0.5, "color": "#f0f2ff", "evidence": "faint bright edge on the handle and right shoulder silhouette"}, {"id": "sky", "type": "hemisphere", "skyColor": "#eef2ff", "groundColor": "#a8b4cc", "intensity": 1.1, "evidence": "ambient: cool lavender studio sweep; ground colour = cloth"}, {"id": "environment", "environmentIntensity": 0.45, "notes": "RoomEnvironment PMREM for glaze/gold reflections"}, {"id": "tone", "exposure": 0.6, "toneMapping": "ACESFilmic", "calibration": "material-pass sweep (exposure 1.0/0.7/0.55): glaze white 236/219/205 vs reference 203-213; ACES compressed whites while mid-tones were 1.7x too bright", "notes": "ACES filmic tone mapping (viewer), exposure 1.0"}, {"id": "ground", "tableColor": "#a3b2cc", "tableRoughness": 0.95, "backgroundColor": "#a9b7cf", "notes": "lavender cloth sweep; background close to the table colour so the horizon does not read as an edge"}];
    return lights;
  }
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
  lights.userData.lightingFromPhoto = [{"id": "key", "type": "directional", "position": [-3.0, 8.0, 7.0], "intensity": 1.7, "color": "#fff8f0", "castShadow": true, "evidence": "window highlights on the left/front of spout and handle; soft cast shadow right-back of the body", "shadow": {"softness": "PCF soft, radius 6, 2048 map (viewer)", "contactShadow": "PCF shadow on the table plane gives the contact/ground shadow under the foot"}, "notes": "PCF soft shadow map gives the ground shadow / contact shadow on the table, as in the photo (shadow right-back of the body)"}, {"id": "fill", "type": "directional", "position": [5.0, 3.0, 4.0], "intensity": 1.3, "color": "#e8eeff", "evidence": "right flank in the photo is dark but not black (region mean 60/55/48); fill swept 0.6/1.0/1.3 -> 49/53/57"}, {"id": "bounce", "type": "directional", "position": [0.0, -3.0, 4.0], "intensity": 0.6, "color": "#c8d0e8", "evidence": "lower band lit by bounce from the lavender cloth (photo 106/95/87 vs 81/72/67 without it); no shadow"}, {"id": "rim", "type": "directional", "position": [4.0, 3.0, -5.0], "intensity": 0.5, "color": "#f0f2ff", "evidence": "faint bright edge on the handle and right shoulder silhouette"}, {"id": "sky", "type": "hemisphere", "skyColor": "#eef2ff", "groundColor": "#a8b4cc", "intensity": 1.1, "evidence": "ambient: cool lavender studio sweep; ground colour = cloth"}, {"id": "environment", "environmentIntensity": 0.45, "notes": "RoomEnvironment PMREM for glaze/gold reflections"}, {"id": "tone", "exposure": 0.6, "toneMapping": "ACESFilmic", "calibration": "material-pass sweep (exposure 1.0/0.7/0.55): glaze white 236/219/205 vs reference 203-213; ACES compressed whites while mid-tones were 1.7x too bright", "notes": "ACES filmic tone mapping (viewer), exposure 1.0"}, {"id": "ground", "tableColor": "#a3b2cc", "tableRoughness": 0.95, "backgroundColor": "#a9b7cf", "notes": "lavender cloth sweep; background close to the table colour so the horizon does not read as an edge"}];
  lights.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  return lights;
}

// PBR materials (clearcoat/iridescence/transmission/anisotropy) need an environment
// map to visually behave as intended — call this once per renderer and assign the
// result to scene.environment before rendering. No external HDR asset required.
export function createPorcelainTeapotEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture {
  const pmrem = new THREE.PMREMGenerator(renderer);
  const texture = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  pmrem.dispose();
  return texture;
}

// three r163+ ignores material.envMapIntensity when the environment comes only from
// scene.environment (a measured orange grip rendered #ff8954 instead of #e85f29). Call this
// after assigning scene.environment: materials that set their own envMapIntensity get the
// environment bound per material, with the scene's environmentIntensity folded in (a bound
// envMap no longer receives it). Materials at the default 1 keep using scene.environment.
export function bindPorcelainTeapotEnvironment(
  root: THREE.Object3D,
  environment: THREE.Texture,
  environmentIntensity = 1,
): number {
  let bound = 0;
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    for (const m of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
      const pbr = m as THREE.MeshStandardMaterial;
      if (typeof pbr.envMapIntensity !== 'number' || pbr.envMapIntensity === 1 || pbr.envMap) continue;
      pbr.envMap = environment;
      pbr.envMapIntensity *= environmentIntensity;
      pbr.needsUpdate = true;
      bound++;
    }
  });
  return bound;
}

// Plan 1.3 §3.2 — auto-framing by bounding box. The Divine Eye can only compare a
// render to the reference if the object is FRAMED consistently (an object framed
// differently scores as wrong even when its shape is right). This positions the camera
// deterministically from the object's bounding box so it fills the frame at a stable
// margin, and sets near/far to the object scale. Call after adding the model to the
// scene, and again on resize (after updating camera.aspect).
export function framePorcelainTeapotCamera(
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
export function createPorcelainTeapotPresentationComposer(
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

export function configurePorcelainTeapotRenderer(renderer: THREE.WebGLRenderer): void {
  // Load-bearing for view-dependent finishes (anodized / Doppler): without ACES + sRGB
  // the environment reflection reads flat/washed instead of a believable metal response.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMappingExposure = 0.6;  // lightingFromPhoto tone.exposure
}

export function createPorcelainTeapotInspectControls(
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
