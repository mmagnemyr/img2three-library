import * as THREE from 'three';
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

// bevelEnabled defaults to true on THREE.ExtrudeGeometry and rounds every
// corner — sharp/pointed profiles (blades, fork tines, spikes) need
// bevelEnabled: false plus lineTo()-only path segments near the tip, since a
// curve command cannot produce a true converging point.
function buildExtrudeShape(points: [number, number][], holes?: [number, number][][]): THREE.Shape {
  const shape = new THREE.Shape();
  if (points.length > 0) {
    shape.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i += 1) {
      shape.lineTo(points[i][0], points[i][1]);
    }
  }
  // Cutouts (e.g. an oval wire-cutter hole) as THREE.Path added to shape.holes —
  // dep-free boolean subtraction via the tessellator, no CSG library needed.
  for (const loop of holes ?? []) {
    if (loop.length < 3) continue;
    const path = new THREE.Path();
    path.moveTo(loop[0][0], loop[0][1]);
    for (let i = 1; i < loop.length; i += 1) path.lineTo(loop[i][0], loop[i][1]);
    path.closePath();
    shape.holes.push(path);
  }
  return shape;
}

// Build an N-gon oval loop (for hole authoring from a compact {cx,cy,rx,ry} descriptor).
function ovalLoop(cx: number, cy: number, rx: number, ry: number, seg = 24): [number, number][] {
  const loop: [number, number][] = [];
  for (let i = 0; i < seg; i += 1) {
    const a = (i / seg) * Math.PI * 2;
    loop.push([cx + Math.cos(a) * rx, cy + Math.sin(a) * ry]);
  }
  return loop;
}

function buildExtrudeGeometry(profile: { points: [number, number][]; depth: number; holes?: [number, number][][]; ovalHoles?: { cx: number; cy: number; rx: number; ry: number }[] }): THREE.ExtrudeGeometry {
  const holes = [...(profile.holes ?? []), ...((profile.ovalHoles ?? []).map((o) => ovalLoop(o.cx, o.cy, o.rx, o.ry)))];
  const shape = buildExtrudeShape(profile.points, holes);
  return new THREE.ExtrudeGeometry(shape, {
    depth: profile.depth,
    bevelEnabled: false,
    steps: 1,
  });
}

function buildLatheGeometry(profile: { points: [number, number][]; segments?: number }): THREE.LatheGeometry {
  const points = profile.points.map(([x, y]) => new THREE.Vector2(Math.max(0.0001, x), y));
  return new THREE.LatheGeometry(points, profile.segments ?? 24);
}

type TaperedStation = { position: [number, number, number]; rx: number; rz: number; twist?: number; squareness?: number };  // HAMMER-PATCH P1

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
      // HAMMER-PATCH P1: superellipse section (squareness n; 2 = ellipse)
      const sqE = 2 / Math.max(2, st.squareness ?? 2);
      const cT = Math.cos(theta), sT = Math.sin(theta);
      const cS = Math.sign(cT) * Math.pow(Math.abs(cT), sqE), sS = Math.sign(sT) * Math.pow(Math.abs(sT), sqE);
      const offset = normals[i].clone().multiplyScalar(cS * st.rx)
        .add(binormals[i].clone().multiplyScalar(sS * st.rz));
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

// Generated from ObjectSculptSpec target: Claw hammer
// Sculpt build pass: optimization-pass
// This factory is intentionally pass-gated. Finish browser screenshot review before unlocking deeper passes.

// HAMMER-PATCH P2/P3 -- generated by tools/patch_factory.py from object-sculpt-spec.json; do not hand-edit.
const HAMMER_PATCH_PARAMS = {"carved": {"count": 24, "xStart": 0.15, "xEnd": 0.2846, "depth": 0.0006, "halfWidthFrac": 0.145, "chevronBow": 0.028, "bowRange": [0.08, 1.0], "bowCurve": "ease-out", "edgeFade": [0.08, 0.2], "step": 0.00045, "radialSegments": 72, "panel": {"x0": 0.205, "x1": 0.286, "u0": 0.47, "u1": 0.74, "inset": 0.0005}, "cavityDarken": 0.55, "partingLine": {"x0": 0.112, "x1": 0.296, "depth": 0.00035, "halfWidth": 0.00045}}, "lanyard": {"x": 0.2863, "z": -0.0011, "radius": 0.002, "depth": 0.004, "cavityBoost": 1.75}, "patch": {"script": "tools/patch_factory.py (re-run after every generate_threejs_factory.py)", "why": "generator sweeps are elliptical with linear station lists (kinks), extrude outlines are polylines, and the repetition emitter is radial-only", "sweeps": {"head": {"step": 0.0015, "radialSegments": 48}, "shank": {"step": 0.002, "radialSegments": 40}, "grip-collar": {"step": 0.001, "radialSegments": 48}, "grip-butt": {"step": 0.0008, "radialSegments": 72}}, "claw": {"outer": [0, 7], "inner": [8, 15], "bevel": 0.0004, "curveSegments": 10}, "decals": [{"component": "cheek-stamp", "target": "head", "texture": "models/claw-hammer/assets/evidence/stamp/stamp_ink.png", "box": [112, 118, 222, 205], "region": {"yMin": 0.006, "xMin": -0.02, "xMax": 0.014, "zMin": -0.014, "zMax": 0.04}, "lift": 0.00012, "paddedTop": 167, "paddedSize": [1280, 914], "renderSize": [840, 600], "metalness": 0.3, "roughness": 0.7, "source": "authoring/make_stamp_ink.py (reference pixels, local-steel normalised)"}, {"component": "shank-scuff", "target": "shank", "texture": "models/claw-hammer/assets/evidence/stamp/scuff.png", "box": [225, 150, 560, 320], "region": {"yMin": 0.0, "xMin": -0.01, "xMax": 0.12, "zMin": -0.04, "zMax": 0.04}, "lift": 0.00012, "paddedTop": 167, "paddedSize": [1280, 914], "renderSize": [840, 600], "metalness": 0.0, "roughness": 0.85, "tint": 0.82, "source": "authoring/make_scuff.py (reference pixels: bright low-saturation mill-scale inside the shank band)"}, {"component": "grip-print-marks", "target": "grip", "texture": "models/claw-hammer/assets/evidence/stamp/grip_marks.png", "box": [960, 340, 1220, 455], "regions": [{"yMin": 0.004, "xMin": 0.279, "xMax": 0.294, "zMin": -0.009, "zMax": 0.007}, {"yMin": 0.004, "xMin": 0.222, "xMax": 0.254, "zMin": -0.012, "zMax": 0.012}], "lift": 8e-05, "paddedTop": 167, "paddedSize": [1280, 914], "renderSize": [840, 600], "metalness": 0.0, "roughness": 0.6, "source": "authoring/make_grip_marks.py (reference pixels: lanyard-hole interior + panel logo)"}], "explode": {"scale": 2.0, "clearance": 0.012}, "decalCamera": "spec.referenceCamera (fitted); paddedTop/paddedSize/renderSize below map viewer pixels to reference pixels", "cavityAO": {"claw-prong-upper": {"why": "(extrude geometry frame: lateral normals are +-Z, mapped to hammer +-Y by the node) blackened forging scale on both lateral faces (photo #211e1f on the camera-facing face), darkest on the slot-facing face; edges/top arc keep the steel", "terms": [{"normalY": [0.4, 0.9], "normalSign": -1, "shade": 0.24, "normalAxis": "z"}, {"normalY": [0.4, 0.9], "normalSign": 1, "shade": 0.3, "normalAxis": "z"}]}, "claw-prong-lower": {"why": "same, mirrored", "terms": [{"normalY": [0.4, 0.9], "normalSign": 1, "shade": 0.24, "normalAxis": "z"}, {"normalY": [0.4, 0.9], "normalSign": -1, "shade": 0.3, "normalAxis": "z"}]}, "head": {"why": "under-cheek throat where shank and claw meet the eye block", "terms": [{"x": [0.006, 0.016], "zMax": [0.012, 0.022], "shade": 0.62}, {"x": [0.0, 0.01], "zMax": [-0.016, -0.006], "shade": 0.7}]}, "shank": {"why": "throat fillet next to the eye and claw underside", "terms": [{"xMax": [0.012, 0.03], "zMax": [-0.008, 0.0], "shade": 0.6}, {"xMax": [0.006, 0.016], "shade": 0.8}]}, "striking-face": {"why": "neck waist concavity (lathe-local y = profile height)", "terms": [{"y": [0.0, 0.004], "yMax": [0.01, 0.016], "shade": 0.72}]}}, "steelFinish": {"size": 512, "why": "reference crops (40x32 px) tile as a weave; procedural longitudinal brushing, palette/roughness from the same crops", "materials": {"forged-steel": {"seed": 11, "albedoLow": 0.82, "albedoHigh": 1.0, "roughLow": 0.26, "roughHigh": 0.46, "bumpScale": 0.12, "repeat": [1, 1]}, "face-steel": {"seed": 23, "albedoLow": 0.9, "albedoHigh": 1.0, "roughLow": 0.45, "roughHigh": 0.6, "bumpScale": 0.25, "repeat": [3, 1]}, "claw-steel": {"seed": 31, "albedoLow": 0.88, "albedoHigh": 1.0, "roughLow": 0.4, "roughHigh": 0.52, "bumpScale": 0.08, "repeat": [40, 40]}}}}, "camera": {"solved": true, "fovDegrees": 19.08, "distance": 0.7163, "target": [0.1307, 0.0221, 0.0463], "orientation": {"yaw": 3.571, "pitch": 40.265, "roll": 0}, "aspect": 1.4, "note": "fitted by authoring/fit_camera.py, silhouette IoU 0.924 at 280x200 (evidence/camera-fit.json)"}};

function hammerDensifyStations(stations: TaperedStation[], step: number): TaperedStation[] {
  // Uniform Catmull-Rom through the authored stations (position, rx, rz, squareness): removes the
  // tangent kinks a linear station list leaves on a sweep, without moving any authored station.
  const n = stations.length;
  const val = (i: number) => {
    const s = stations[Math.min(n - 1, Math.max(0, i))];
    return [s.position[0], s.position[1], s.position[2], s.rx, s.rz, s.squareness ?? 2];
  };
  const cr = (p0: number, p1: number, p2: number, p3: number, t: number) =>
    0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t);
  const out: TaperedStation[] = [];
  for (let i = 0; i < n - 1; i += 1) {
    const a = stations[i], b = stations[i + 1];
    const len = Math.hypot(b.position[0] - a.position[0], b.position[1] - a.position[1], b.position[2] - a.position[2]);
    const m = Math.max(1, Math.ceil(len / step));
    const P0 = val(i - 1), P1 = val(i), P2 = val(i + 1), P3 = val(i + 2);
    for (let k = 0; k < m; k += 1) {
      const t = k / m; const v = P0.map((_, j) => cr(P0[j], P1[j], P2[j], P3[j], t));
      out.push({ position: [v[0], v[1], v[2]], rx: Math.max(1e-5, v[3]), rz: Math.max(1e-5, v[4]), squareness: Math.max(2, v[5]) });
    }
  }
  out.push({ ...stations[n - 1] });
  return out;
}

function hammerSmoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

function hammerCarveGrip(mesh: THREE.Mesh): void {
  const C = HAMMER_PATCH_PARAMS.carved as any;
  const L = HAMMER_PATCH_PARAMS.lanyard as any;
  if (!C) return;
  const comp = mesh.userData.sculptComponent as any;
  const sweep = comp.geometryDescriptor.taperedSweep;
  const dense = hammerDensifyStations(sweep.stations as TaperedStation[], C.step ?? 0.0005);
  const geo = buildTaperedSweepGeometry({ stations: dense, radialSegments: C.radialSegments ?? 72, capEnds: true });
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const pitch = (C.xEnd - C.xStart) / Math.max(1, C.count - 1);
  const w = C.halfWidthFrac * pitch;
  for (let i = 0; i < pos.count; i += 1) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const r = Math.hypot(y, z);
    let d = 0;
    let cavity = 0;
    if (r > 1e-6) {
      const a = Math.atan2(y, -z);          // 0 at the -Z edge, +-PI at the +Z edge
      const u = Math.abs(a) / Math.PI;       // 0..1 across a lateral face
      const ub = Math.min(1, Math.max(0, (u - (C.bowRange?.[0] ?? 0.08)) / ((C.bowRange?.[1] ?? 0.92) - (C.bowRange?.[0] ?? 0.08))));
      const bowF = C.bowCurve === 'ease-out' ? 1 - (1 - ub) * (1 - ub) : hammerSmoothstep(0, 1, ub);
      const shift = -C.chevronBow * bowF;
      const edgeFade = C.edgeFade ? hammerSmoothstep(C.edgeFade[0], C.edgeFade[1], u) : 1;
      const t = (x - C.xStart - shift) / pitch;
      const k = Math.round(t);
      const inPanel = y > 0 && C.panel && x > C.panel.x0 && x < C.panel.x1 && u > C.panel.u0 && u < C.panel.u1;
      if (k >= 0 && k < C.count && !inPanel) {
        const delta = (t - k) * pitch;
        const g = Math.exp(-Math.pow(delta / w, 2));
        // pillow: rib crests rounded, groove floors darker
        d += C.depth * g * edgeFade;
        cavity = Math.max(cavity, g * edgeFade);
      }
      if (C.panel && y > 0) {
        const px = hammerSmoothstep(C.panel.x0 - 0.0008, C.panel.x0 + 0.0004, x) * (1 - hammerSmoothstep(C.panel.x1 - 0.0004, C.panel.x1 + 0.0008, x));
        const pu = hammerSmoothstep(C.panel.u0 - 0.03, C.panel.u0 + 0.015, u) * (1 - hammerSmoothstep(C.panel.u1 - 0.015, C.panel.u1 + 0.03, u));
        d += (C.panel.inset ?? 0.0004) * px * pu;
      }
      if (L) {
        const dist = Math.hypot(x - L.x, z - L.z);
        if (dist < L.radius * 1.35 && Math.abs(y) > r * 0.5) {
          const f = 1 - hammerSmoothstep(L.radius * 0.85, L.radius * 1.15, dist);
          d += L.depth * f;
          cavity = Math.max(cavity, f * (L.cavityBoost ?? 1.6));
        }
      }
      if (C.partingLine && x > C.partingLine.x0 && x < C.partingLine.x1) {
        const pl = Math.exp(-Math.pow(y / C.partingLine.halfWidth, 2)) * hammerSmoothstep(0.5, 0.8, Math.abs(z) / r);
        d += C.partingLine.depth * pl; cavity = Math.max(cavity, pl * 0.8);
      }
      const nY = y / r, nZ = z / r;
      pos.setXYZ(i, x, y - nY * d, z - nZ * d);
    }
    const shade = 1 - Math.min(0.9, C.cavityDarken * cavity);
    colors[i * 3] = shade; colors[i * 3 + 1] = shade; colors[i * 3 + 2] = shade;
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  mesh.geometry.dispose();
  mesh.geometry = geo;
  const mat = (mesh.material as THREE.MeshPhysicalMaterial).clone();
  mat.vertexColors = true;
  mesh.material = mat;
}

// HAMMER-PATCH P4: rebuild every other sweep from Catmull-Rom densified stations.
function hammerRebuildSweep(mesh: THREE.Mesh, step: number, radial: number): void {
  const sweep = (mesh.userData.sculptComponent as any).geometryDescriptor.taperedSweep;
  const geo = buildTaperedSweepGeometry({ stations: hammerDensifyStations(sweep.stations as TaperedStation[], step), radialSegments: radial, capEnds: sweep.capEnds ?? true });
  mesh.geometry.dispose(); mesh.geometry = geo;
}

// HAMMER-PATCH P5: claw prong outline as splines (outer arc, straight tip cut, inner arc) + small bevel.
function hammerRebuildClaw(mesh: THREE.Mesh, P: any): void {
  const prof = (mesh.userData.sculptComponent as any).geometryDescriptor.profile2D;
  const pts: [number, number][] = prof.points;
  const V = (i: number) => new THREE.Vector2(pts[i][0], pts[i][1]);
  const shape = new THREE.Shape();
  const o = P.outer as [number, number], inn = P.inner as [number, number];
  shape.moveTo(pts[0][0], pts[0][1]);
  shape.splineThru(Array.from({ length: o[1] - o[0] }, (_, k) => V(o[0] + k + 1)));
  shape.lineTo(pts[inn[0]][0], pts[inn[0]][1]);
  shape.splineThru(Array.from({ length: inn[1] - inn[0] }, (_, k) => V(inn[0] + k + 1)));
  for (let i = inn[1] + 1; i < pts.length; i += 1) shape.lineTo(pts[i][0], pts[i][1]);
  shape.closePath();
  const b = P.bevel ?? 0.0004;
  const geo = new THREE.ExtrudeGeometry(shape, { depth: prof.depth - 2 * b, bevelEnabled: b > 0, bevelThickness: b, bevelSize: b, bevelSegments: 2, curveSegments: P.curveSegments ?? 10, steps: 1 });
  geo.translate(0, 0, b);
  mesh.geometry.dispose(); mesh.geometry = geo;
}

// HAMMER-PATCH P6: cheek stamp = the reference's own ink pixels, projected through the fitted reference camera
// onto a conforming shell lifted off the head's top (+Y) cheek face (CLAUDE.md lesson 6).
function hammerProjectDecal(stampMesh: THREE.Mesh, head: THREE.Mesh, P: any, cam: any): void {
  let top: THREE.Object3D = head; while (top.parent) top = top.parent;
  top.updateMatrixWorld(true);
  const D = Math.PI / 180;
  const camera = new THREE.PerspectiveCamera(cam.fovDegrees, P.renderSize[0] / P.renderSize[1], 0.01, 50);
  const [tx, ty, tz] = cam.target; const az = cam.orientation.yaw * D, el = cam.orientation.pitch * D;
  camera.position.set(tx + cam.distance * Math.sin(az) * Math.cos(el), ty + cam.distance * Math.sin(el), tz + cam.distance * Math.cos(az) * Math.cos(el));
  camera.lookAt(tx, ty, tz); camera.updateMatrixWorld(true); camera.updateProjectionMatrix();
  const g = head.geometry.index ? head.geometry.toNonIndexed() : head.geometry.clone();
  g.computeVertexNormals();
  const pos = g.attributes.position, nrm = g.attributes.normal;
  const outP: number[] = [], outN: number[] = [], outUV: number[] = [];
  const toStamp = new THREE.Matrix4().copy(stampMesh.matrixWorld).invert();
  const nm = new THREE.Matrix3().getNormalMatrix(head.matrixWorld);
  const v = new THREE.Vector3(), n = new THREE.Vector3(), w = new THREE.Vector3();
  const RS = P.regions ?? [P.region];
  for (let t = 0; t < pos.count; t += 3) {
    let ok = true;
    for (let k = 0; k < 3; k += 1) { const i = t + k; const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
      if (!RS.some((R: any) => y > R.yMin && x > R.xMin && x < R.xMax && z > R.zMin && z < R.zMax)) ok = false; }
    if (!ok) continue;
    for (let k = 0; k < 3; k += 1) {
      const i = t + k;
      n.fromBufferAttribute(nrm, i);
      v.fromBufferAttribute(pos, i).addScaledVector(n, P.lift).applyMatrix4(head.matrixWorld);
      w.copy(v).project(camera);
      const px = (w.x + 1) / 2 * P.renderSize[0], py = (1 - w.y) / 2 * P.renderSize[1];
      const ox = px * P.paddedSize[0] / P.renderSize[0], oy = py * P.paddedSize[1] / P.renderSize[1] - P.paddedTop;
      outUV.push((ox - P.box[0]) / (P.box[2] - P.box[0]), 1 - (oy - P.box[1]) / (P.box[3] - P.box[1]));
      const ls = v.clone().applyMatrix4(toStamp); outP.push(ls.x, ls.y, ls.z);
      const wn = n.clone().applyMatrix3(nm).normalize().transformDirection(toStamp); outN.push(wn.x, wn.y, wn.z);
    }
  }
  const shell = new THREE.BufferGeometry();
  shell.setAttribute('position', new THREE.Float32BufferAttribute(outP, 3));
  shell.setAttribute('normal', new THREE.Float32BufferAttribute(outN, 3));
  shell.setAttribute('uv', new THREE.Float32BufferAttribute(outUV, 2));
  const tex = new THREE.TextureLoader().load(P.texture);
  tex.colorSpace = THREE.SRGBColorSpace; tex.wrapS = THREE.ClampToEdgeWrapping; tex.wrapT = THREE.ClampToEdgeWrapping; tex.anisotropy = 8;
  // open decal shell, not a closed surface: tools/export_meshes.py skips it for self_intersection.py by this flag
  (shell as any).parameters = { openEnded: true, decalShell: true };
  stampMesh.geometry.dispose(); stampMesh.geometry = shell;
  const prevMat = stampMesh.material as THREE.Material;
  stampMesh.material = new THREE.MeshStandardMaterial({ map: tex, transparent: true, depthWrite: false, metalness: P.metalness ?? 0.3, roughness: P.roughness ?? 0.7,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
  if (P.tint) (stampMesh.material as THREE.MeshStandardMaterial).color.setScalar(P.tint);
  stampMesh.material.userData = { ...prevMat.userData, projectedDecal: P.texture };
  stampMesh.castShadow = false;
  stampMesh.userData.explodeWithParent = true;
}

// HAMMER-PATCH P7: three r170 applies material.envMapIntensity only when material.envMap is set explicitly;
// with scene.environment alone the spec's per-material envMapIntensity is silently ignored. Bind the scene
// environment on first render for every material whose spec declares envMapIntensity.
function hammerBindEnvIntensity(mesh: THREE.Mesh): void {
  const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
  for (const m of mats as THREE.MeshStandardMaterial[]) {
    const sm = (m.userData as any)?.sculptMaterial;
    if (!sm || typeof sm.envMapIntensity !== 'number') continue;
    m.envMapIntensity = sm.envMapIntensity;
    mesh.onBeforeRender = (_r, scene) => {
      if (!m.envMap && (scene as THREE.Scene).environment) { m.envMap = (scene as THREE.Scene).environment; m.needsUpdate = true; }
    };
  }
}

// HAMMER-PATCH P8: brushed-steel finish. The 40x32 px reference crops tile as a visible weave on the sweep
// UVs; replace their maps with a seeded procedural longitudinal-brushing set (albedo modulation, roughness,
// bump) whose base colour/roughness come from the same crop (spec material values).
function hammerBrushedTextures(seed: number, size: number, P: any): { albedo: THREE.CanvasTexture; rough: THREE.CanvasTexture } {
  const mk = () => { const c = document.createElement('canvas'); c.width = size; c.height = size; return c; };
  const ca = mk(), cr = mk(); const xa = ca.getContext('2d')!, xr = cr.getContext('2d')!;
  const ia = xa.createImageData(size, size), ir = xr.createImageData(size, size);
  let s = seed >>> 0; const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const col = new Float32Array(size); // per-column streak value (brushing runs along v)
  for (let f = 0, amp = 1; f < 4; f += 1, amp *= 0.55) {
    const period = Math.max(2, Math.round(size / (6 * Math.pow(3, f)))); let a = rnd(), b = rnd();
    for (let x = 0; x < size; x += 1) { if (x % period === 0) { a = b; b = rnd(); } const t = (x % period) / period; col[x] += amp * (a + (b - a) * t * t * (3 - 2 * t)); }
  }
  let mn = Infinity, mx = -Infinity; for (const v of col) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
  for (let y = 0; y < size; y += 1) {
    const rowJ = 0.04 * Math.sin(y * 0.05 + seed) ;
    for (let x = 0; x < size; x += 1) {
      const n = (col[x] - mn) / (mx - mn + 1e-6) + rowJ + (rnd() - 0.5) * 0.08;
      const alb = Math.max(0, Math.min(255, 255 * (P.albedoLow + (P.albedoHigh - P.albedoLow) * n)));
      const rg = Math.max(0, Math.min(255, 255 * (P.roughLow + (P.roughHigh - P.roughLow) * (1 - n))));
      const i = (y * size + x) * 4;
      ia.data[i] = ia.data[i + 1] = ia.data[i + 2] = alb; ia.data[i + 3] = 255;
      ir.data[i] = ir.data[i + 1] = ir.data[i + 2] = rg; ir.data[i + 3] = 255;
    }
  }
  xa.putImageData(ia, 0, 0); xr.putImageData(ir, 0, 0);
  const albedo = new THREE.CanvasTexture(ca); albedo.colorSpace = THREE.SRGBColorSpace;
  const rough = new THREE.CanvasTexture(cr);
  for (const t of [albedo, rough]) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.anisotropy = 8; }
  return { albedo, rough };
}

function hammerApplySteelFinish(meshes: Record<string, THREE.Mesh>, F: any): void {
  const done = new Map<string, THREE.MeshPhysicalMaterial>();
  for (const id of Object.keys(meshes)) {
    const mesh = meshes[id]; const m = mesh.material as THREE.MeshPhysicalMaterial; const sm = (m.userData as any)?.sculptMaterial;
    if (!sm || !F.materials?.[sm.id]) continue;
    const P = F.materials[sm.id];
    let mat = done.get(sm.id);
    if (!mat) {
      mat = m;
      const tex = hammerBrushedTextures(P.seed ?? 7, F.size ?? 512, P);
      mat.map = tex.albedo; mat.roughnessMap = tex.rough; mat.normalMap = null; mat.aoMap = null;
      mat.bumpMap = tex.rough; mat.bumpScale = P.bumpScale ?? 0.15;
      mat.color.setStyle(sm.baseColor, THREE.SRGBColorSpace);
      mat.roughness = 1.0; mat.metalness = 1.0;
      if (P.repeat) { for (const t of [tex.albedo, tex.rough]) t.repeat.set(P.repeat[0], P.repeat[1]); }
      mat.needsUpdate = true; done.set(sm.id, mat);
    }
    mesh.material = mat;
  }
}

// HAMMER-PATCH P9: analytic cavity AO as vertex colour (environment + hemisphere fill have no occlusion, so the
// claw slot, the under-cheek throat and the neck waist light up; CLAUDE.md lesson 12). Regions in each mesh's
// own geometry frame, parameters from spec.factoryPatch.cavityAO.
function hammerCavityAO(mesh: THREE.Mesh, rule: any): void {
  const g = mesh.geometry; const pos = g.attributes.position; if (!g.attributes.normal) g.computeVertexNormals();
  const nrm = g.attributes.normal; const col = new Float32Array(pos.count * 3);
  const ss = (e0: number, e1: number, x: number) => { const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0))); return t * t * (3 - 2 * t); };
  for (let i = 0; i < pos.count; i += 1) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    let shade = 1;
    for (const t of rule.terms) {
      let w = 1;
      if (t.normalY !== undefined) { const ax = t.normalAxis ?? 'y'; const nv = ax === 'x' ? nrm.getX(i) : ax === 'z' ? nrm.getZ(i) : nrm.getY(i); w *= ss(t.normalY[0], t.normalY[1], nv * (t.normalSign ?? 1)); }
      if (t.x) w *= ss(t.x[0], t.x[1], x); if (t.xMax) w *= 1 - ss(t.xMax[0], t.xMax[1], x);
      if (t.y) w *= ss(t.y[0], t.y[1], y); if (t.yMax) w *= 1 - ss(t.yMax[0], t.yMax[1], y);
      if (t.z) w *= ss(t.z[0], t.z[1], z); if (t.zMax) w *= 1 - ss(t.zMax[0], t.zMax[1], z);
      shade *= 1 - (1 - t.shade) * w;
    }
    col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = shade;
  }
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  (mesh.material as THREE.MeshStandardMaterial).vertexColors = true; (mesh.material as THREE.Material).needsUpdate = true;
}

// HAMMER-PATCH P10: explode + pick share ONE definition of a part: a named mesh carrying sculptComponent that is
// not surface detail (userData.explodeWithParent). Decal shells ride the part they are projected onto.
function attachHammerInteraction(root: THREE.Group, meshes: Record<string, THREE.Mesh>, P: any): void {
  const parts = Object.keys(meshes).filter((id) => !meshes[id].userData.explodeWithParent);
  const riders: Record<string, string> = {};
  for (const d of P.decals ?? []) { riders[d.component] = d.target; meshes[d.component] && (meshes[d.component].userData.explodeParent = d.target); }
  const home = new Map<THREE.Object3D, THREE.Vector3>();
  for (const id of Object.keys(meshes)) home.set(meshes[id], meshes[id].position.clone());
  const explode = (t: number, scale = P.explode?.scale ?? 2.0, clearance = P.explode?.clearance ?? 0.012) => {
    root.updateMatrixWorld(true);
    for (const id of Object.keys(meshes)) meshes[id].position.copy(home.get(meshes[id])!);
    root.updateMatrixWorld(true);
    const centre = new THREE.Box3().setFromObject(root).getCenter(new THREE.Vector3());
    const offsets: Record<string, THREE.Vector3> = {};
    for (const id of parts) {
      const c = new THREE.Box3().setFromObject(meshes[id]).getCenter(new THREE.Vector3());
      const d = c.clone().sub(centre); const len = d.length();
      offsets[id] = d.clone().multiplyScalar((scale - 1) * t).add(len > 1e-6 ? d.clone().normalize().multiplyScalar(clearance * t) : new THREE.Vector3(0, clearance * t, 0));
    }
    for (const id of Object.keys(meshes)) {
      const off = offsets[riders[id] ?? id]; if (!off) continue;
      const m = meshes[id]; const parent = m.parent!; const w = m.getWorldPosition(new THREE.Vector3()).add(off);
      m.position.copy(parent.worldToLocal(w));
    }
    root.updateMatrixWorld(true);
  };
  const partOf = (o: THREE.Object3D | null): string | null => {
    for (let x = o; x; x = x?.parent ?? null) {
      const comp = (x.userData as any).sculptComponent; if (!comp || !(x as THREE.Mesh).isMesh) continue;
      const id = comp.id as string; return riders[id] ?? (x.userData.explodeWithParent ? null : id);
    }
    return null;
  };
  const pick = (raycaster: THREE.Raycaster) => { const hit = raycaster.intersectObject(root, true)[0]; return hit ? partOf(hit.object) : null; };
  Object.assign(root.userData.sculptRuntime, { parts, explode, partOf, pick });
}

function applyHammerPatches(meshes: Record<string, THREE.Mesh>): void {
  const FP = HAMMER_PATCH_PARAMS.patch as any;
  for (const id of Object.keys(FP.sweeps ?? {})) if (meshes[id]) hammerRebuildSweep(meshes[id], FP.sweeps[id].step, FP.sweeps[id].radialSegments);
  for (const id of ['claw-prong-upper', 'claw-prong-lower']) if (meshes[id] && FP.claw) hammerRebuildClaw(meshes[id], FP.claw);
  if (meshes['grip']) hammerCarveGrip(meshes['grip']);
  if (FP.steelFinish) hammerApplySteelFinish(meshes, FP.steelFinish);
  for (const id of Object.keys(meshes)) hammerBindEnvIntensity(meshes[id]);
  for (const id of Object.keys(FP.cavityAO ?? {})) if (meshes[id]) hammerCavityAO(meshes[id], FP.cavityAO[id]);
  for (const d of FP.decals ?? []) if (meshes[d.component] && meshes[d.target]) hammerProjectDecal(meshes[d.component], meshes[d.target], d, HAMMER_PATCH_PARAMS.camera);
}

export function createClawHammerModel(options: ProceduralModelOptions = {}): THREE.Group {
  const root = new THREE.Group();
  root.name = "Claw hammer";
  root.userData.reconstructionEvidence = {"itemFamily": null, "subtype": null, "componentAdapter": null, "route": null, "exactnessTier": null, "referenceCamera": {"solved": true, "fovDegrees": 19.08, "distance": 0.7163, "target": [0.1307, 0.0221, 0.0463], "orientation": {"yaw": 3.571, "pitch": 40.265, "roll": 0}, "aspect": 1.4, "note": "fitted by authoring/fit_camera.py, silhouette IoU 0.924 at 280x200 (evidence/camera-fit.json)"}, "approximationNotes": []};
  root.userData.materialPipeline = {};
  root.userData.materialReferenceRegistry = null;

  const materialMap: Record<string, THREE.Material> = {};
  materialMap["forged-steel"] = createSculptMaterial(
    "forged-steel",
    {"id": "forged-steel", "name": "Satin forged steel (head, shank)", "type": "physical", "shaderModel": "MeshPhysicalMaterial / PBR", "baseColor": "#9c989a", "color": "#9c989a", "albedo": {"dominant": "#9c989a", "secondary": ["#6f7275", "#b4b6b8"], "samplingNotes": "sampled from reference crops (evidence/material-evidence)"}, "colorVariation": {"palette": ["#9c989a", "#6f7275", "#b4b6b8"], "pattern": "streaked", "amplitude": 0.06, "heightCorrelation": 0.2}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.1, "role": "broad forging tone shifts along the shank"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.08, "role": "longitudinal grinding streaks"}, {"id": "micro", "frequency": 64.0, "amplitude": 0.04, "role": "fine brushing scratches visible in highlights"}], "roughness": {"base": 0.38, "variation": 0.12, "map": "independent-procedural-field", "localResponse": "see localOverrides"}, "metalness": {"base": 1.0, "variation": 0.0}, "normal": {"pattern": "derived-from-independent-height-field", "strength": 0.2, "scale": 24.0, "space": "tangent"}, "bump": {"pattern": "none", "amplitude": 0.0, "scale": 1.0}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "Darken creases, seams, intersections, and recessed local features."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "shank-scuff-patch", "kind": "stain", "region": "shank, x 0.02-0.10 m, +Y and -Z faces", "description": "White mill-scale/zinc scuff patch: light grey albedo, rougher, low metalness.", "evidenceRefs": ["zone-r1c0"], "color": "#d6d6d2", "roughness": 0.75, "metalness": 0.2, "fadedMask": true}, {"id": "claw-dark-scale", "kind": "stain", "region": "claw inner faces / slot walls", "description": "Darker blackened forging scale on the claw.", "evidenceRefs": ["zone-r0c0"], "color": "#3c3e40", "roughness": 0.55, "dirtAmount": 0.5, "cavityBias": 0.6}, {"id": "cheek-stamp", "kind": "decal", "region": "cheek +Y face near the neck", "description": "Stamped monogram and text, dark ink in engraved marks.", "evidenceRefs": ["zone-r0c0"], "color": "#2a2a2a", "decal": true}, {"id": "longitudinal-brushing", "kind": "scratch", "region": "head and shank", "description": "Longitudinal grinding streaks.", "evidenceRefs": ["zone-r1c1"], "orientation": "along shank / head axis", "roughness": 0.32}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Satin forged steel, metalness 1 (bare metal), grey neutral; values from reference crop, see evidence/material-evidence.", "referencePbr": {"usable": true, "confidence": 0.749, "targetThreshold": 0.7, "maps": {"albedo": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-forged-steel/forged-steel_albedo.png", "url": "models/claw-hammer/assets/material-evidence/pbr-forged-steel/forged-steel_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-forged-steel/forged-steel_roughness.png", "url": "models/claw-hammer/assets/material-evidence/pbr-forged-steel/forged-steel_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-forged-steel/forged-steel_height.png", "url": "models/claw-hammer/assets/material-evidence/pbr-forged-steel/forged-steel_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-forged-steel/forged-steel_normal.png", "url": "models/claw-hammer/assets/material-evidence/pbr-forged-steel/forged-steel_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-forged-steel/forged-steel_ao.png", "url": "models/claw-hammer/assets/material-evidence/pbr-forged-steel/forged-steel_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "palette": ["#7E797B", "#706C6E", "#ACA8A9", "#635E60", "#918D8E"], "sourceImage": "material-evidence/crops/forged-steel.png", "limitation": "single-image PBR extraction is an estimate; 70%+ extraction confidence still needs render screenshot review"}, "analyzeTextureReadOnly": {"file": "material-evidence/analyze-forged-steel.json", "rejected": "finishClass 'worn-composite', metalness 0, roughness 0.9 -- misclassified bare steel (CLAUDE.md lessons 4/14); values above are agent-set from the crop: metalness 1 (bare forged steel), roughness from highlight spread"}},
    options
  );
  materialMap["face-steel"] = createSculptMaterial(
    "face-steel",
    {"id": "face-steel", "name": "Machined striking face steel", "type": "physical", "shaderModel": "MeshPhysicalMaterial / PBR", "baseColor": "#787272", "color": "#787272", "albedo": {"dominant": "#787272", "secondary": ["#7d7d7a", "#b8b8b4"], "samplingNotes": "sampled from reference crops (evidence/material-evidence)"}, "colorVariation": {"palette": ["#787272", "#7d7d7a", "#b8b8b4"], "pattern": "streaked", "amplitude": 0.06, "heightCorrelation": 0.2}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.1, "role": "broad forging tone shifts along the shank"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.08, "role": "longitudinal grinding streaks"}, {"id": "micro", "frequency": 64.0, "amplitude": 0.04, "role": "fine brushing scratches visible in highlights"}], "roughness": {"base": 0.48, "variation": 0.15, "map": "independent-procedural-field", "localResponse": "see localOverrides"}, "metalness": {"base": 1.0, "variation": 0.0}, "normal": {"pattern": "derived-from-independent-height-field", "strength": 0.2, "scale": 24.0, "space": "tangent"}, "bump": {"pattern": "none", "amplitude": 0.0, "scale": 1.0}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "Darken creases, seams, intersections, and recessed local features."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "face-chamfer-bright", "kind": "gloss", "region": "face chamfer ring", "description": "Bright polished chamfer ring.", "evidenceRefs": ["zone-r1c0"], "roughness": 0.25}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Machined face: lighter, rougher than the cheek.", "referencePbr": {"usable": true, "confidence": 0.735, "targetThreshold": 0.7, "maps": {"albedo": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-face-steel/face-steel_albedo.png", "url": "models/claw-hammer/assets/material-evidence/pbr-face-steel/face-steel_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-face-steel/face-steel_roughness.png", "url": "models/claw-hammer/assets/material-evidence/pbr-face-steel/face-steel_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-face-steel/face-steel_height.png", "url": "models/claw-hammer/assets/material-evidence/pbr-face-steel/face-steel_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-face-steel/face-steel_normal.png", "url": "models/claw-hammer/assets/material-evidence/pbr-face-steel/face-steel_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-face-steel/face-steel_ao.png", "url": "models/claw-hammer/assets/material-evidence/pbr-face-steel/face-steel_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "palette": ["#8B8686", "#7E7878", "#706A6A", "#5A5555", "#9C9696"], "sourceImage": "material-evidence/crops/face-steel.png", "limitation": "single-image PBR extraction is an estimate; 70%+ extraction confidence still needs render screenshot review"}, "analyzeTextureReadOnly": {"file": "material-evidence/analyze-face-steel.json", "rejected": "finishClass 'worn-composite', metalness 0, roughness 0.9 -- misclassified bare steel (CLAUDE.md lessons 4/14); values above are agent-set from the crop: metalness 1 (bare forged steel), roughness from highlight spread"}},
    options
  );
  materialMap["claw-steel"] = createSculptMaterial(
    "claw-steel",
    {"id": "claw-steel", "name": "Darker claw steel (forging scale)", "type": "physical", "shaderModel": "MeshPhysicalMaterial / PBR", "baseColor": "#7c7e82", "color": "#7c7e82", "albedo": {"dominant": "#7c7e82", "secondary": ["#27242a", "#5a5c60"], "samplingNotes": "sampled from reference crops (evidence/material-evidence)"}, "colorVariation": {"palette": ["#7c7e82", "#27242a", "#5a5c60"], "pattern": "streaked", "amplitude": 0.06, "heightCorrelation": 0.2}, "textureResolution": 1024, "textureProjection": {"mode": "uv", "repeat": [2.0, 2.0], "anisotropy": 8, "texelDensityIntent": "Preserve stable world/object-scale detail; do not stretch micro detail with component scale."}, "surfaceFrequencyBands": [{"id": "macro", "frequency": 2.0, "amplitude": 0.1, "role": "broad forging tone shifts along the shank"}, {"id": "meso", "frequency": 14.0, "amplitude": 0.08, "role": "longitudinal grinding streaks"}, {"id": "micro", "frequency": 64.0, "amplitude": 0.04, "role": "fine brushing scratches visible in highlights"}], "roughness": {"base": 0.46, "variation": 0.1, "map": "independent-procedural-field", "localResponse": "see localOverrides"}, "metalness": {"base": 1.0, "variation": 0.0}, "normal": {"pattern": "derived-from-independent-height-field", "strength": 0.2, "scale": 24.0, "space": "tangent"}, "bump": {"pattern": "none", "amplitude": 0.0, "scale": 1.0}, "displacement": {"pattern": "none", "amplitude": 0.0, "scale": 1.0, "silhouetteAffects": false}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "Darken creases, seams, intersections, and recessed local features."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "claw-edge-wear", "kind": "chip", "region": "claw top edges", "description": "Lighter worn steel on the claw's top edges.", "evidenceRefs": ["zone-r0c0"], "color": "#7c7f82"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Blackened forging scale on the claw.", "referencePbr": {"usable": true, "confidence": 0.86, "targetThreshold": 0.7, "maps": {"albedo": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-claw-steel/claw-steel_albedo.png", "url": "models/claw-hammer/assets/material-evidence/pbr-claw-steel/claw-steel_albedo.png", "channel": "albedo", "source": "reference-pixel-extraction"}, "roughness": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-claw-steel/claw-steel_roughness.png", "url": "models/claw-hammer/assets/material-evidence/pbr-claw-steel/claw-steel_roughness.png", "channel": "roughness", "source": "reference-pixel-extraction"}, "height": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-claw-steel/claw-steel_height.png", "url": "models/claw-hammer/assets/material-evidence/pbr-claw-steel/claw-steel_height.png", "channel": "height", "source": "reference-pixel-extraction"}, "normal": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-claw-steel/claw-steel_normal.png", "url": "models/claw-hammer/assets/material-evidence/pbr-claw-steel/claw-steel_normal.png", "channel": "normal", "source": "reference-pixel-extraction"}, "ao": {"path": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/pbr-claw-steel/claw-steel_ao.png", "url": "models/claw-hammer/assets/material-evidence/pbr-claw-steel/claw-steel_ao.png", "channel": "ao", "source": "reference-pixel-extraction"}}, "palette": ["#151014", "#0F090D", "#1D181B", "#070204", "#848082"], "sourceImage": "material-evidence/crops/claw-steel.png", "limitation": "single-image PBR extraction is an estimate; 70%+ extraction confidence still needs render screenshot review"}, "analyzeTextureReadOnly": {"file": "material-evidence/analyze-claw-steel.json", "rejected": "finishClass 'worn-composite', metalness 0, roughness 0.9 -- misclassified bare steel (CLAUDE.md lessons 4/14); values above are agent-set from the crop: metalness 1 (bare forged steel), roughness from highlight spread"}},
    options
  );
  materialMap["grip-elastomer"] = createSculptMaterial(
    "grip-elastomer",
    {"id": "grip-elastomer", "name": "Orange TPE grip", "type": "physical", "shaderModel": "MeshPhysicalMaterial / PBR", "baseColor": "#e64e08", "color": "#e64e08", "albedo": {"dominant": "#e64e08", "secondary": ["#dc3c0a", "#f3632e"], "samplingNotes": "sampled from reference crops (evidence/material-evidence)"}, "colorVariation": {"palette": ["#e64e08", "#dc3c0a", "#f3632e"], "pattern": "mottled", "amplitude": 0.04, "heightCorrelation": 0.2}, "roughness": {"base": 0.5, "variation": 0.08, "map": "independent-procedural-field", "localResponse": "see localOverrides"}, "metalness": {"base": 0.0, "variation": 0.0}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "Darken creases, seams, intersections, and recessed local features."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "rib-groove-cavity", "kind": "groove", "region": "grip rib grooves", "description": "Darker cavity shading inside the rib grooves (vertex AO).", "evidenceRefs": ["zone-r1c1"], "dirtAmount": 0.35, "cavityBias": 1.0}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Vivid red-orange TPE, dielectric, satin.", "envMapIntensity": 0.3, "textureless": {"declared": true, "evidence": ["material-evidence/pbr-grip-elastomer.json: valueRange 0.18 on a 100x60 crop, and that variation is the rib-groove cavity shading (carved geometry), not albedo/print", "agent vision: solid moulded orange TPE with uniform satin finish; no print/grain except the small logo panel (geometry)"]}},
    options
  );
  materialMap["stamp-ink"] = createSculptMaterial(
    "stamp-ink",
    {"id": "stamp-ink", "name": "Stamp decal carrier", "type": "physical", "shaderModel": "MeshPhysicalMaterial / PBR", "baseColor": "#2a2a2a", "color": "#2a2a2a", "albedo": {"dominant": "#2a2a2a", "secondary": [], "samplingNotes": "sampled from reference crops (evidence/material-evidence)"}, "colorVariation": {"palette": ["#2a2a2a"], "pattern": "streaked", "amplitude": 0.06, "heightCorrelation": 0.2}, "roughness": {"base": 0.5, "variation": 0.05, "map": "independent-procedural-field", "localResponse": "see localOverrides"}, "metalness": {"base": 0.6, "variation": 0.0}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "Darken creases, seams, intersections, and recessed local features."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "stamp-ink", "kind": "decal", "region": "cheek stamp", "description": "Dark stamped marks.", "evidenceRefs": ["zone-r0c0"], "color": "#262626"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Carrier for the projected stamp marks; transparent except the ink.", "textureless": {"declared": true, "evidence": ["carrier only: the stamp marks are the projected reference pixels (tools/patch_factory.py); no tiling texture"]}},
    options
  );
  materialMap["scuff-zinc"] = createSculptMaterial(
    "scuff-zinc",
    {"id": "scuff-zinc", "name": "Shank scuff patch (projected)", "type": "physical", "shaderModel": "MeshPhysicalMaterial / PBR", "baseColor": "#d2d0d1", "color": "#d2d0d1", "albedo": {"dominant": "#d2d0d1", "secondary": ["#c6c4c6", "#d9d7d9"], "samplingNotes": "sampled from reference crops (evidence/material-evidence)"}, "colorVariation": {"palette": ["#d2d0d1", "#c6c4c6", "#d9d7d9"], "pattern": "mottled", "amplitude": 0.04, "heightCorrelation": 0.2}, "roughness": {"base": 0.8, "variation": 0.05, "map": "independent-procedural-field", "localResponse": "see localOverrides"}, "metalness": {"base": 0.0, "variation": 0.0}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "Darken creases, seams, intersections, and recessed local features."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "shank-scuff-patch", "kind": "stain", "region": "shank", "description": "Projected white mill-scale scuff.", "evidenceRefs": ["zone-r1c0"], "color": "#d2d0d1", "roughness": 0.8}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Carrier for the projected scuff pixels; matte, low metalness (material-evidence/crops/shank-scuff.png median #d2d0d1).", "textureless": {"declared": true, "evidence": ["carrier only: the scuff colours are the projected reference pixels (evidence/stamp/scuff.png, authoring/make_scuff.py)"]}},
    options
  );
  materialMap["grip-marks-ink"] = createSculptMaterial(
    "grip-marks-ink",
    {"id": "grip-marks-ink", "name": "Grip marks (projected)", "type": "physical", "shaderModel": "MeshPhysicalMaterial / PBR", "baseColor": "#5a1a06", "color": "#5a1a06", "albedo": {"dominant": "#5a1a06", "secondary": [], "samplingNotes": "sampled from reference crops (evidence/material-evidence)"}, "colorVariation": {"palette": ["#5a1a06"], "pattern": "mottled", "amplitude": 0.04, "heightCorrelation": 0.2}, "roughness": {"base": 0.6, "variation": 0.05, "map": "independent-procedural-field", "localResponse": "see localOverrides"}, "metalness": {"base": 0.0, "variation": 0.0}, "ambientOcclusion": {"cavityStrength": 0.25, "contactShadowBias": 0.35, "notes": "Darken creases, seams, intersections, and recessed local features."}, "wear": {"edgeWear": 0.0, "scratches": [], "chips": []}, "dirt": {"amount": 0.0, "cavityBias": 0.0, "color": "#2F2A22"}, "localOverrides": [{"id": "lanyard-hole-shading", "kind": "hole", "region": "lanyard hole", "description": "Dark hole interior from the photo.", "evidenceRefs": ["zone-r2c2"], "color": "#5a1a06"}], "shaderNotes": ["Prefer MeshPhysicalMaterial when clearcoat, sheen, transmission, or thin-surface response is observed; otherwise use MeshStandardMaterial-compatible PBR channels.", "Generate albedo, roughness, height/normal, and AO independently; never alias albedo into roughness.", "Use normal/bump/displacement only when they map to observed surface relief.", "Use displacement geometry when the observed relief changes the close-up silhouette; texture-only relief is insufficient there."], "notes": "Carrier for projected grip logo / lanyard-hole interior pixels.", "textureless": {"declared": true, "evidence": ["carrier only: logo/hole colours are the projected reference pixels (evidence/stamp/grip_marks.png, authoring/make_grip_marks.py)"]}},
    options
  );

  const nodes: Record<string, THREE.Object3D> = { root };
  const meshes: Record<string, THREE.Mesh> = {};
  const sockets: Record<string, THREE.Object3D> = {};
  const colliders: Record<string, unknown> = {};
  const destructionGroups: Record<string, THREE.Object3D[]> = {};

  const endpoint_head_0 = makeAttachmentEndpoint(null);
  const node_head_0 = new THREE.Group();
  node_head_0.name = "Forged head (cheek and eye block)__pivot";
  node_head_0.scale.set(1, 1, 1);
  if (endpoint_head_0) {
    node_head_0.position.copy(endpoint_head_0.start);
    node_head_0.rotation.set(0.0, -0.3446, 0.0);
  } else {
    node_head_0.position.set(0.0, 0.015, 0.0);
    node_head_0.rotation.set(0.0, -0.3446, 0.0);
  }
  node_head_0.userData.sculptComponent = {"id": "head", "name": "Forged head (cheek and eye block)", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "One forged mass whose rounded-rectangle section varies smoothly from the claw root to the neck; a box/cylinder cannot follow the taper.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "rounded-section", "bevelRadius": 0.003, "segments": 4}, "deformationStack": [], "uvStrategy": "sweep u = around section, v = along head axis", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.0, -0.032], "rx": 0.006, "rz": 0.006, "squareness": 2.0}, {"position": [-0.001, 0.0, -0.024], "rx": 0.01404, "rz": 0.0105, "squareness": 3.0}, {"position": [-0.001, 0.0, -0.012], "rx": 0.01782, "rz": 0.0122, "squareness": 3.5}, {"position": [-0.001, 0.0, 0.004], "rx": 0.0189, "rz": 0.0126, "squareness": 3.5}, {"position": [-0.001, 0.0, 0.018], "rx": 0.01728, "rz": 0.0124, "squareness": 3.0}, {"position": [0.0, 0.0, 0.03], "rx": 0.0128, "rz": 0.0118, "squareness": 2.5}, {"position": [0.0, 0.0, 0.036], "rx": 0.0112, "rz": 0.011, "squareness": 2.0}], "radialSegments": 36, "capEnds": true}}, "parent": null, "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.75}, "transform": {"position": [0.0, 0.015, 0.0], "rotation": [0.0, -0.3446, 0.0]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-neck", "localPosition": [0, 0, 0.033], "axis": [0, 0, 1]}, {"id": "socket-claw-root", "localPosition": [0, 0, -0.02], "axis": [0, 0, -1]}, {"id": "socket-eye", "localPosition": [0.012, 0, 0], "axis": [1, 0, 0]}], "collider": {"type": "box", "offset": [0, 0, 0.0], "scale": [0.036, 0.026, 0.11], "isTrigger": false, "notes": "head + claw + face box proxy"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "forged-steel"}}, "material": "forged-steel", "materialLayers": ["forged-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "cheek-flat-sides", "kind": "bevel", "description": "Flat lateral cheek faces with rounded section corners (superellipse section).", "geometryEffect": "rounded-rectangle sweep section", "evidenceRefs": ["zone-r0c0"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.36, "microRoughness": 0.12, "bumpAmplitude": 0.12, "normalPattern": "longitudinal grinding streaks along the head axis (P8 brushed bump)", "displacementPattern": "none (silhouette from sweep)", "occlusionPattern": "under-cheek throat cavity AO (P9 vertex colour)", "edgeWearPattern": "brighter worn edges on the cheek top arc", "notes": "stamp marks projected (P6)"}, "evidenceRefs": ["head-region", "full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "head", "dominantAlbedo": "rgba(124, 119, 121, 1.0)", "secondaryAlbedo": "rgba(164, 160, 160, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.692, "roughnessEstimate": 0.122, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-steel-a.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.578}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-steel-a.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_head_0.userData.actionProfile = {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-neck", "localPosition": [0, 0, 0.033], "axis": [0, 0, 1]}, {"id": "socket-claw-root", "localPosition": [0, 0, -0.02], "axis": [0, 0, -1]}, {"id": "socket-eye", "localPosition": [0.012, 0, 0], "axis": [1, 0, 0]}], "collider": {"type": "box", "offset": [0, 0, 0.0], "scale": [0.036, 0.026, 0.11], "isTrigger": false, "notes": "head + claw + face box proxy"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "forged-steel"}};
  (nodes["root"] ?? root).add(node_head_0);
  nodes["head"] = node_head_0;
  const mesh_head_0Geometry = endpoint_head_0
    ? new THREE.CylinderGeometry(endpoint_head_0.endRadius, endpoint_head_0.baseRadius, endpoint_head_0.length, 32, 12)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.0, 0.0, -0.032], "rx": 0.006, "rz": 0.006, "squareness": 2.0}, {"position": [-0.001, 0.0, -0.024], "rx": 0.01404, "rz": 0.0105, "squareness": 3.0}, {"position": [-0.001, 0.0, -0.012], "rx": 0.01782, "rz": 0.0122, "squareness": 3.5}, {"position": [-0.001, 0.0, 0.004], "rx": 0.0189, "rz": 0.0126, "squareness": 3.5}, {"position": [-0.001, 0.0, 0.018], "rx": 0.01728, "rz": 0.0124, "squareness": 3.0}, {"position": [0.0, 0.0, 0.03], "rx": 0.0128, "rz": 0.0118, "squareness": 2.5}, {"position": [0.0, 0.0, 0.036], "rx": 0.0112, "rz": 0.011, "squareness": 2.0}], "radialSegments": 36, "capEnds": true});
  if (!endpoint_head_0) {
    mesh_head_0Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_head_0 = new THREE.Mesh(
    mesh_head_0Geometry,
    materialMap["forged-steel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_head_0.name = "Forged head (cheek and eye block)";
  if (endpoint_head_0) {
    mesh_head_0.position.copy(endpoint_head_0.midpoint);
    mesh_head_0.quaternion.copy(endpoint_head_0.quaternion);
  }
  mesh_head_0.castShadow = options.castShadow ?? true;
  mesh_head_0.receiveShadow = options.receiveShadow ?? true;
  mesh_head_0.userData.sculptComponent = {"id": "head", "name": "Forged head (cheek and eye block)", "level": "macro", "role": "body", "importance": 1.0, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "One forged mass whose rounded-rectangle section varies smoothly from the claw root to the neck; a box/cylinder cannot follow the taper.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "rounded-section", "bevelRadius": 0.003, "segments": 4}, "deformationStack": [], "uvStrategy": "sweep u = around section, v = along head axis", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.0, 0.0, -0.032], "rx": 0.006, "rz": 0.006, "squareness": 2.0}, {"position": [-0.001, 0.0, -0.024], "rx": 0.01404, "rz": 0.0105, "squareness": 3.0}, {"position": [-0.001, 0.0, -0.012], "rx": 0.01782, "rz": 0.0122, "squareness": 3.5}, {"position": [-0.001, 0.0, 0.004], "rx": 0.0189, "rz": 0.0126, "squareness": 3.5}, {"position": [-0.001, 0.0, 0.018], "rx": 0.01728, "rz": 0.0124, "squareness": 3.0}, {"position": [0.0, 0.0, 0.03], "rx": 0.0128, "rz": 0.0118, "squareness": 2.5}, {"position": [0.0, 0.0, 0.036], "rx": 0.0112, "rz": 0.011, "squareness": 2.0}], "radialSegments": 36, "capEnds": true}}, "parent": null, "attachment": null, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.75}, "transform": {"position": [0.0, 0.015, 0.0], "rotation": [0.0, -0.3446, 0.0]}, "actionProfile": {"animationRole": "root", "pivot": {"mode": "center", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": false, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-neck", "localPosition": [0, 0, 0.033], "axis": [0, 0, 1]}, {"id": "socket-claw-root", "localPosition": [0, 0, -0.02], "axis": [0, 0, -1]}, {"id": "socket-eye", "localPosition": [0.012, 0, 0], "axis": [1, 0, 0]}], "collider": {"type": "box", "offset": [0, 0, 0.0], "scale": [0.036, 0.026, 0.11], "isTrigger": false, "notes": "head + claw + face box proxy"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "forged-steel"}}, "material": "forged-steel", "materialLayers": ["forged-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "cheek-flat-sides", "kind": "bevel", "description": "Flat lateral cheek faces with rounded section corners (superellipse section).", "geometryEffect": "rounded-rectangle sweep section", "evidenceRefs": ["zone-r0c0"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.36, "microRoughness": 0.12, "bumpAmplitude": 0.12, "normalPattern": "longitudinal grinding streaks along the head axis (P8 brushed bump)", "displacementPattern": "none (silhouette from sweep)", "occlusionPattern": "under-cheek throat cavity AO (P9 vertex colour)", "edgeWearPattern": "brighter worn edges on the cheek top arc", "notes": "stamp marks projected (P6)"}, "evidenceRefs": ["head-region", "full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "head", "dominantAlbedo": "rgba(124, 119, 121, 1.0)", "secondaryAlbedo": "rgba(164, 160, 160, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.692, "roughnessEstimate": 0.122, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-steel-a.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.578}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-steel-a.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_head_0.add(mesh_head_0);
  meshes["head"] = mesh_head_0;
  colliders["head"] = {"type": "box", "offset": [0, 0, 0.0], "scale": [0.036, 0.026, 0.11], "isTrigger": false, "notes": "head + claw + face box proxy"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_head_0);
  const socket_head_socket_neck_0 = new THREE.Object3D();
  socket_head_socket_neck_0.name = "socket-neck";
  socket_head_socket_neck_0.position.set(0.0, 0.0, 0.033);
  socket_head_socket_neck_0.rotation.set(0, 0, 0);
  socket_head_socket_neck_0.userData.socket = {"id": "socket-neck", "localPosition": [0, 0, 0.033], "axis": [0, 0, 1]};
  node_head_0.add(socket_head_socket_neck_0);
  sockets["head:socket-neck"] = socket_head_socket_neck_0;
  const socket_head_socket_claw_root_1 = new THREE.Object3D();
  socket_head_socket_claw_root_1.name = "socket-claw-root";
  socket_head_socket_claw_root_1.position.set(0.0, 0.0, -0.02);
  socket_head_socket_claw_root_1.rotation.set(0, 0, 0);
  socket_head_socket_claw_root_1.userData.socket = {"id": "socket-claw-root", "localPosition": [0, 0, -0.02], "axis": [0, 0, -1]};
  node_head_0.add(socket_head_socket_claw_root_1);
  sockets["head:socket-claw-root"] = socket_head_socket_claw_root_1;
  const socket_head_socket_eye_2 = new THREE.Object3D();
  socket_head_socket_eye_2.name = "socket-eye";
  socket_head_socket_eye_2.position.set(0.012, 0.0, 0.0);
  socket_head_socket_eye_2.rotation.set(0, 0, 0);
  socket_head_socket_eye_2.userData.socket = {"id": "socket-eye", "localPosition": [0.012, 0, 0], "axis": [1, 0, 0]};
  node_head_0.add(socket_head_socket_eye_2);
  sockets["head:socket-eye"] = socket_head_socket_eye_2;

  const endpoint_striking_face_1 = makeAttachmentEndpoint(null);
  const node_striking_face_1 = new THREE.Group();
  node_striking_face_1.name = "Striking face, bell and waisted neck__pivot";
  node_striking_face_1.scale.set(1, 1, 1);
  if (endpoint_striking_face_1) {
    node_striking_face_1.position.copy(endpoint_striking_face_1.start);
    node_striking_face_1.rotation.set(1.5708, 0.0, 0.0);
  } else {
    node_striking_face_1.position.set(0.0, 0.0, 0.0335);
    node_striking_face_1.rotation.set(1.5708, 0.0, 0.0);
  }
  node_striking_face_1.userData.sculptComponent = {"id": "striking-face", "name": "Striking face, bell and waisted neck", "level": "meso", "role": "striking-face", "importance": 0.7, "confidence": 0.8, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Turned, rotationally symmetric form (waist, bell, chamfered face) -> lathe profile about the head axis.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "chamfer", "bevelRadius": 0.0015, "segments": 1}, "deformationStack": [], "uvStrategy": "lathe u around, v along profile", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0112, 0.0], [0.011, 0.004], [0.0112, 0.008], [0.0122, 0.012], [0.0136, 0.016], [0.0142, 0.02], [0.0143, 0.028], [0.0142, 0.0312], [0.0128, 0.033], [0.0119, 0.0333], [0.0076, 0.0337], [0.0001, 0.0339]], "segments": 48}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-neck", "localStart": [0.0, 0.0, 0.0335], "localEnd": [0.0, 0.0, 0.0674], "contactType": "butt-forged", "embedDepth": 0.004, "gapTolerance": 0.0005, "contactNormal": [0, 0, 1], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.8}, "transform": {"position": [0.0, 0.0, 0.0335], "rotation": [1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-strike", "localPosition": [0, 0.0339, 0], "axis": [0, 1, 0], "purpose": "impact point: spark/dust effect and hit test"}], "collider": {"type": "cylinder", "offset": [0, 0.02, 0], "scale": [0.03, 0.04, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "face-steel"}}, "material": "face-steel", "materialLayers": ["face-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "face-chamfer", "kind": "bevel", "description": "Bright chamfer ring around the face rim.", "geometryEffect": "two-point chamfer in lathe profile", "evidenceRefs": ["zone-r1c0"], "confidence": 0.85}, {"id": "neck-waist", "kind": "taper", "description": "Waisted neck between cheek and bell.", "geometryEffect": "lathe profile minimum radius", "evidenceRefs": ["zone-r1c0"], "confidence": 0.8}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.15, "bumpAmplitude": 0.25, "normalPattern": "mottled machined/struck face pitting", "displacementPattern": "slightly domed face in lathe profile", "occlusionPattern": "neck waist AO (P9)", "edgeWearPattern": "bright chamfer ring", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "striking-face", "dominantAlbedo": "rgba(134, 129, 129, 1.0)", "secondaryAlbedo": "rgba(98, 93, 93, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.66, "roughnessEstimate": 0.162, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-face.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.438}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-face.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_striking_face_1.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-strike", "localPosition": [0, 0.0339, 0], "axis": [0, 1, 0], "purpose": "impact point: spark/dust effect and hit test"}], "collider": {"type": "cylinder", "offset": [0, 0.02, 0], "scale": [0.03, 0.04, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "face-steel"}};
  (nodes["head"] ?? root).add(node_striking_face_1);
  nodes["striking-face"] = node_striking_face_1;
  const mesh_striking_face_1Geometry = endpoint_striking_face_1
    ? new THREE.CylinderGeometry(endpoint_striking_face_1.endRadius, endpoint_striking_face_1.baseRadius, endpoint_striking_face_1.length, 32, 12)
    : buildLatheGeometry({"points": [[0.0112, 0.0], [0.011, 0.004], [0.0112, 0.008], [0.0122, 0.012], [0.0136, 0.016], [0.0142, 0.02], [0.0143, 0.028], [0.0142, 0.0312], [0.0128, 0.033], [0.0119, 0.0333], [0.0076, 0.0337], [0.0001, 0.0339]], "segments": 48});
  if (!endpoint_striking_face_1) {
    mesh_striking_face_1Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_striking_face_1 = new THREE.Mesh(
    mesh_striking_face_1Geometry,
    materialMap["face-steel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_striking_face_1.name = "Striking face, bell and waisted neck";
  if (endpoint_striking_face_1) {
    mesh_striking_face_1.position.copy(endpoint_striking_face_1.midpoint);
    mesh_striking_face_1.quaternion.copy(endpoint_striking_face_1.quaternion);
  }
  mesh_striking_face_1.castShadow = options.castShadow ?? true;
  mesh_striking_face_1.receiveShadow = options.receiveShadow ?? true;
  mesh_striking_face_1.userData.sculptComponent = {"id": "striking-face", "name": "Striking face, bell and waisted neck", "level": "meso", "role": "striking-face", "importance": 0.7, "confidence": 0.8, "primitive": "lathe", "topologyClass": "continuous-sculpt", "topologyRationale": "Turned, rotationally symmetric form (waist, bell, chamfered face) -> lathe profile about the head axis.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "chamfer", "bevelRadius": 0.0015, "segments": 1}, "deformationStack": [], "uvStrategy": "lathe u around, v along profile", "normalStrategy": "vertex normals from generated geometry", "latheProfile": {"points": [[0.0112, 0.0], [0.011, 0.004], [0.0112, 0.008], [0.0122, 0.012], [0.0136, 0.016], [0.0142, 0.02], [0.0143, 0.028], [0.0142, 0.0312], [0.0128, 0.033], [0.0119, 0.0333], [0.0076, 0.0337], [0.0001, 0.0339]], "segments": 48}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-neck", "localStart": [0.0, 0.0, 0.0335], "localEnd": [0.0, 0.0, 0.0674], "contactType": "butt-forged", "embedDepth": 0.004, "gapTolerance": 0.0005, "contactNormal": [0, 0, 1], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.8}, "transform": {"position": [0.0, 0.0, 0.0335], "rotation": [1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-strike", "localPosition": [0, 0.0339, 0], "axis": [0, 1, 0], "purpose": "impact point: spark/dust effect and hit test"}], "collider": {"type": "cylinder", "offset": [0, 0.02, 0], "scale": [0.03, 0.04, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "face-steel"}}, "material": "face-steel", "materialLayers": ["face-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "face-chamfer", "kind": "bevel", "description": "Bright chamfer ring around the face rim.", "geometryEffect": "two-point chamfer in lathe profile", "evidenceRefs": ["zone-r1c0"], "confidence": 0.85}, {"id": "neck-waist", "kind": "taper", "description": "Waisted neck between cheek and bell.", "geometryEffect": "lathe profile minimum radius", "evidenceRefs": ["zone-r1c0"], "confidence": 0.8}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.15, "bumpAmplitude": 0.25, "normalPattern": "mottled machined/struck face pitting", "displacementPattern": "slightly domed face in lathe profile", "occlusionPattern": "neck waist AO (P9)", "edgeWearPattern": "bright chamfer ring", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "striking-face", "dominantAlbedo": "rgba(134, 129, 129, 1.0)", "secondaryAlbedo": "rgba(98, 93, 93, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.66, "roughnessEstimate": 0.162, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-face.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.438}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-face.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_striking_face_1.add(mesh_striking_face_1);
  meshes["striking-face"] = mesh_striking_face_1;
  colliders["striking-face"] = {"type": "cylinder", "offset": [0, 0.02, 0], "scale": [0.03, 0.04, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_striking_face_1);
  const socket_striking_face_socket_strike_0 = new THREE.Object3D();
  socket_striking_face_socket_strike_0.name = "socket-strike";
  socket_striking_face_socket_strike_0.position.set(0.0, 0.0339, 0.0);
  socket_striking_face_socket_strike_0.rotation.set(0, 0, 0);
  socket_striking_face_socket_strike_0.userData.socket = {"id": "socket-strike", "localPosition": [0, 0.0339, 0], "axis": [0, 1, 0], "purpose": "impact point: spark/dust effect and hit test"};
  node_striking_face_1.add(socket_striking_face_socket_strike_0);
  sockets["striking-face:socket-strike"] = socket_striking_face_socket_strike_0;

  const endpoint_claw_prong_upper_2 = makeAttachmentEndpoint(null);
  const node_claw_prong_upper_2 = new THREE.Group();
  node_claw_prong_upper_2.name = "Claw prong (upper, +Y)__pivot";
  node_claw_prong_upper_2.scale.set(1, 1, 1);
  if (endpoint_claw_prong_upper_2) {
    node_claw_prong_upper_2.position.copy(endpoint_claw_prong_upper_2.start);
    node_claw_prong_upper_2.rotation.set(-1.5408, 0.0, 0.0);
  } else {
    node_claw_prong_upper_2.position.set(0.0, 0.0008, 0.0);
    node_claw_prong_upper_2.rotation.set(-1.5408, 0.0, 0.0);
  }
  node_claw_prong_upper_2.userData.sculptComponent = {"id": "claw-prong-upper", "name": "Claw prong (upper, +Y)", "level": "meso", "role": "claw-prong", "importance": 0.7, "confidence": 0.7, "primitive": "extrude", "topologyClass": "assembled-solid", "topologyRationale": "Each forged prong has two flat parallel lateral faces and a flat-cut wedge tip (visible as crisp planar faces in the reference); the curvature is only in-plane, so an extruded outline IS the form (not a bent plane).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "extrude planar", "normalStrategy": "vertex normals from generated geometry", "profile2D": {"points": [[-0.0135, 0.01], [-0.0137, 0.026], [-0.0113, 0.04], [-0.006, 0.052], [0.0018, 0.061], [0.0118, 0.068], [0.0235, 0.0718], [0.0315, 0.073], [0.033, 0.0668], [0.0255, 0.0648], [0.0168, 0.0605], [0.0092, 0.0545], [0.0052, 0.0468], [0.0042, 0.039], [0.0058, 0.029], [0.008, 0.019], [0.004, 0.008]], "depth": 0.0105}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-claw-root", "localStart": [0.0, 0.0, -0.012], "localEnd": [0.032, 0.0, -0.07], "contactType": "embed-forged", "embedDepth": 0.008, "gapTolerance": 0.0005, "contactNormal": [0, 0, 1], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, 0.0008, 0.0], "rotation": [-1.5408, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-nail-slot", "localPosition": [0.03, 0.066, -0.0009], "axis": [0, 1, 0], "purpose": "nail-pulling contact point in the V slot (effect/IK target)"}], "collider": {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "claw-steel"}}, "material": "claw-steel", "materialLayers": ["claw-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "claw-v-slot", "kind": "hole", "description": "Tapering V slot between the two prongs, open at the tips.", "geometryEffect": "prongs offset +-Y with a small splay angle", "evidenceRefs": ["zone-r0c0"], "confidence": 0.85}], "surfaceDetail": {"macroRoughness": 0.46, "microRoughness": 0.08, "bumpAmplitude": 0.08, "normalPattern": "forging scale, fine brushing", "displacementPattern": "none", "occlusionPattern": "dark slot-facing inner face + slot root AO (P9)", "edgeWearPattern": "0.4 mm bevel catches edge highlight", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "claw-prong-upper", "dominantAlbedo": "rgba(92, 88, 89, 1.0)", "secondaryAlbedo": "rgba(106, 102, 102, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.683, "roughnessEstimate": 0.134, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/claw-faces.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.566}, "agentOverrideNote": "re-extracted at material-pass from material-evidence/crops/claw-faces.png (a fully-inside-mask window on the lit claw faces, median closest to the measured claw-top #7d7c7e); the first crop (claw-steel.png) was dominated by slot shadow (dominant rgba(13,8,11)); class metal by agent vision, extractor confidence capped <0.7"}};
  node_claw_prong_upper_2.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-nail-slot", "localPosition": [0.03, 0.066, -0.0009], "axis": [0, 1, 0], "purpose": "nail-pulling contact point in the V slot (effect/IK target)"}], "collider": {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "claw-steel"}};
  (nodes["head"] ?? root).add(node_claw_prong_upper_2);
  nodes["claw-prong-upper"] = node_claw_prong_upper_2;
  const mesh_claw_prong_upper_2Geometry = endpoint_claw_prong_upper_2
    ? new THREE.CylinderGeometry(endpoint_claw_prong_upper_2.endRadius, endpoint_claw_prong_upper_2.baseRadius, endpoint_claw_prong_upper_2.length, 32, 12)
    : buildExtrudeGeometry({"points": [[-0.0135, 0.01], [-0.0137, 0.026], [-0.0113, 0.04], [-0.006, 0.052], [0.0018, 0.061], [0.0118, 0.068], [0.0235, 0.0718], [0.0315, 0.073], [0.033, 0.0668], [0.0255, 0.0648], [0.0168, 0.0605], [0.0092, 0.0545], [0.0052, 0.0468], [0.0042, 0.039], [0.0058, 0.029], [0.008, 0.019], [0.004, 0.008]], "depth": 0.0105});
  if (!endpoint_claw_prong_upper_2) {
    mesh_claw_prong_upper_2Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_claw_prong_upper_2 = new THREE.Mesh(
    mesh_claw_prong_upper_2Geometry,
    materialMap["claw-steel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_claw_prong_upper_2.name = "Claw prong (upper, +Y)";
  if (endpoint_claw_prong_upper_2) {
    mesh_claw_prong_upper_2.position.copy(endpoint_claw_prong_upper_2.midpoint);
    mesh_claw_prong_upper_2.quaternion.copy(endpoint_claw_prong_upper_2.quaternion);
  }
  mesh_claw_prong_upper_2.castShadow = options.castShadow ?? true;
  mesh_claw_prong_upper_2.receiveShadow = options.receiveShadow ?? true;
  mesh_claw_prong_upper_2.userData.sculptComponent = {"id": "claw-prong-upper", "name": "Claw prong (upper, +Y)", "level": "meso", "role": "claw-prong", "importance": 0.7, "confidence": 0.7, "primitive": "extrude", "topologyClass": "assembled-solid", "topologyRationale": "Each forged prong has two flat parallel lateral faces and a flat-cut wedge tip (visible as crisp planar faces in the reference); the curvature is only in-plane, so an extruded outline IS the form (not a bent plane).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "extrude planar", "normalStrategy": "vertex normals from generated geometry", "profile2D": {"points": [[-0.0135, 0.01], [-0.0137, 0.026], [-0.0113, 0.04], [-0.006, 0.052], [0.0018, 0.061], [0.0118, 0.068], [0.0235, 0.0718], [0.0315, 0.073], [0.033, 0.0668], [0.0255, 0.0648], [0.0168, 0.0605], [0.0092, 0.0545], [0.0052, 0.0468], [0.0042, 0.039], [0.0058, 0.029], [0.008, 0.019], [0.004, 0.008]], "depth": 0.0105}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-claw-root", "localStart": [0.0, 0.0, -0.012], "localEnd": [0.032, 0.0, -0.07], "contactType": "embed-forged", "embedDepth": 0.008, "gapTolerance": 0.0005, "contactNormal": [0, 0, 1], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, 0.0008, 0.0], "rotation": [-1.5408, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-nail-slot", "localPosition": [0.03, 0.066, -0.0009], "axis": [0, 1, 0], "purpose": "nail-pulling contact point in the V slot (effect/IK target)"}], "collider": {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "claw-steel"}}, "material": "claw-steel", "materialLayers": ["claw-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "claw-v-slot", "kind": "hole", "description": "Tapering V slot between the two prongs, open at the tips.", "geometryEffect": "prongs offset +-Y with a small splay angle", "evidenceRefs": ["zone-r0c0"], "confidence": 0.85}], "surfaceDetail": {"macroRoughness": 0.46, "microRoughness": 0.08, "bumpAmplitude": 0.08, "normalPattern": "forging scale, fine brushing", "displacementPattern": "none", "occlusionPattern": "dark slot-facing inner face + slot root AO (P9)", "edgeWearPattern": "0.4 mm bevel catches edge highlight", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "claw-prong-upper", "dominantAlbedo": "rgba(92, 88, 89, 1.0)", "secondaryAlbedo": "rgba(106, 102, 102, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.683, "roughnessEstimate": 0.134, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/claw-faces.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.566}, "agentOverrideNote": "re-extracted at material-pass from material-evidence/crops/claw-faces.png (a fully-inside-mask window on the lit claw faces, median closest to the measured claw-top #7d7c7e); the first crop (claw-steel.png) was dominated by slot shadow (dominant rgba(13,8,11)); class metal by agent vision, extractor confidence capped <0.7"}};
  node_claw_prong_upper_2.add(mesh_claw_prong_upper_2);
  meshes["claw-prong-upper"] = mesh_claw_prong_upper_2;
  colliders["claw-prong-upper"] = {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_claw_prong_upper_2);
  const socket_claw_prong_upper_socket_nail_slot_0 = new THREE.Object3D();
  socket_claw_prong_upper_socket_nail_slot_0.name = "socket-nail-slot";
  socket_claw_prong_upper_socket_nail_slot_0.position.set(0.03, 0.066, -0.0009);
  socket_claw_prong_upper_socket_nail_slot_0.rotation.set(0, 0, 0);
  socket_claw_prong_upper_socket_nail_slot_0.userData.socket = {"id": "socket-nail-slot", "localPosition": [0.03, 0.066, -0.0009], "axis": [0, 1, 0], "purpose": "nail-pulling contact point in the V slot (effect/IK target)"};
  node_claw_prong_upper_2.add(socket_claw_prong_upper_socket_nail_slot_0);
  sockets["claw-prong-upper:socket-nail-slot"] = socket_claw_prong_upper_socket_nail_slot_0;

  const endpoint_claw_prong_lower_3 = makeAttachmentEndpoint(null);
  const node_claw_prong_lower_3 = new THREE.Group();
  node_claw_prong_lower_3.name = "Claw prong (lower, -Y)__pivot";
  node_claw_prong_lower_3.scale.set(1, 1, 1);
  if (endpoint_claw_prong_lower_3) {
    node_claw_prong_lower_3.position.copy(endpoint_claw_prong_lower_3.start);
    node_claw_prong_lower_3.rotation.set(-1.6008, 0.0, 0.0);
  } else {
    node_claw_prong_lower_3.position.set(0.0, -0.0113, 0.0);
    node_claw_prong_lower_3.rotation.set(-1.6008, 0.0, 0.0);
  }
  node_claw_prong_lower_3.userData.sculptComponent = {"id": "claw-prong-lower", "name": "Claw prong (lower, -Y)", "level": "meso", "role": "claw-prong", "importance": 0.7, "confidence": 0.7, "primitive": "extrude", "topologyClass": "assembled-solid", "topologyRationale": "Each forged prong has two flat parallel lateral faces and a flat-cut wedge tip (visible as crisp planar faces in the reference); the curvature is only in-plane, so an extruded outline IS the form (not a bent plane).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "extrude planar", "normalStrategy": "vertex normals from generated geometry", "profile2D": {"points": [[-0.0135, 0.01], [-0.0137, 0.026], [-0.0113, 0.04], [-0.006, 0.052], [0.0018, 0.061], [0.0118, 0.068], [0.0235, 0.0718], [0.0315, 0.073], [0.033, 0.0668], [0.0255, 0.0648], [0.0168, 0.0605], [0.0092, 0.0545], [0.0052, 0.0468], [0.0042, 0.039], [0.0058, 0.029], [0.008, 0.019], [0.004, 0.008]], "depth": 0.0105}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-claw-root", "localStart": [0.0, 0.0, -0.012], "localEnd": [0.032, 0.0, -0.07], "contactType": "embed-forged", "embedDepth": 0.008, "gapTolerance": 0.0005, "contactNormal": [0, 0, 1], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, -0.0113, 0.0], "rotation": [-1.6008, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "claw-steel"}}, "material": "claw-steel", "materialLayers": ["claw-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "claw-v-slot", "kind": "hole", "description": "Tapering V slot between the two prongs, open at the tips.", "geometryEffect": "prongs offset +-Y with a small splay angle", "evidenceRefs": ["zone-r0c0"], "confidence": 0.85}], "surfaceDetail": {"macroRoughness": 0.46, "microRoughness": 0.08, "bumpAmplitude": 0.08, "normalPattern": "forging scale, fine brushing", "displacementPattern": "none", "occlusionPattern": "dark slot-facing inner face + slot root AO (P9)", "edgeWearPattern": "0.4 mm bevel catches edge highlight", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "claw-prong-lower", "dominantAlbedo": "rgba(92, 88, 89, 1.0)", "secondaryAlbedo": "rgba(106, 102, 102, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.683, "roughnessEstimate": 0.134, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/claw-faces.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.566}, "agentOverrideNote": "re-extracted at material-pass from material-evidence/crops/claw-faces.png (a fully-inside-mask window on the lit claw faces, median closest to the measured claw-top #7d7c7e); the first crop (claw-steel.png) was dominated by slot shadow (dominant rgba(13,8,11)); class metal by agent vision, extractor confidence capped <0.7"}};
  node_claw_prong_lower_3.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "claw-steel"}};
  (nodes["head"] ?? root).add(node_claw_prong_lower_3);
  nodes["claw-prong-lower"] = node_claw_prong_lower_3;
  const mesh_claw_prong_lower_3Geometry = endpoint_claw_prong_lower_3
    ? new THREE.CylinderGeometry(endpoint_claw_prong_lower_3.endRadius, endpoint_claw_prong_lower_3.baseRadius, endpoint_claw_prong_lower_3.length, 32, 12)
    : buildExtrudeGeometry({"points": [[-0.0135, 0.01], [-0.0137, 0.026], [-0.0113, 0.04], [-0.006, 0.052], [0.0018, 0.061], [0.0118, 0.068], [0.0235, 0.0718], [0.0315, 0.073], [0.033, 0.0668], [0.0255, 0.0648], [0.0168, 0.0605], [0.0092, 0.0545], [0.0052, 0.0468], [0.0042, 0.039], [0.0058, 0.029], [0.008, 0.019], [0.004, 0.008]], "depth": 0.0105});
  if (!endpoint_claw_prong_lower_3) {
    mesh_claw_prong_lower_3Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_claw_prong_lower_3 = new THREE.Mesh(
    mesh_claw_prong_lower_3Geometry,
    materialMap["claw-steel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_claw_prong_lower_3.name = "Claw prong (lower, -Y)";
  if (endpoint_claw_prong_lower_3) {
    mesh_claw_prong_lower_3.position.copy(endpoint_claw_prong_lower_3.midpoint);
    mesh_claw_prong_lower_3.quaternion.copy(endpoint_claw_prong_lower_3.quaternion);
  }
  mesh_claw_prong_lower_3.castShadow = options.castShadow ?? true;
  mesh_claw_prong_lower_3.receiveShadow = options.receiveShadow ?? true;
  mesh_claw_prong_lower_3.userData.sculptComponent = {"id": "claw-prong-lower", "name": "Claw prong (lower, -Y)", "level": "meso", "role": "claw-prong", "importance": 0.7, "confidence": 0.7, "primitive": "extrude", "topologyClass": "assembled-solid", "topologyRationale": "Each forged prong has two flat parallel lateral faces and a flat-cut wedge tip (visible as crisp planar faces in the reference); the curvature is only in-plane, so an extruded outline IS the form (not a bent plane).", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "extrude planar", "normalStrategy": "vertex normals from generated geometry", "profile2D": {"points": [[-0.0135, 0.01], [-0.0137, 0.026], [-0.0113, 0.04], [-0.006, 0.052], [0.0018, 0.061], [0.0118, 0.068], [0.0235, 0.0718], [0.0315, 0.073], [0.033, 0.0668], [0.0255, 0.0648], [0.0168, 0.0605], [0.0092, 0.0545], [0.0052, 0.0468], [0.0042, 0.039], [0.0058, 0.029], [0.008, 0.019], [0.004, 0.008]], "depth": 0.0105}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-claw-root", "localStart": [0.0, 0.0, -0.012], "localEnd": [0.032, 0.0, -0.07], "contactType": "embed-forged", "embedDepth": 0.008, "gapTolerance": 0.0005, "contactNormal": [0, 0, 1], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, -0.0113, 0.0], "rotation": [-1.6008, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "claw-steel"}}, "material": "claw-steel", "materialLayers": ["claw-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "claw-v-slot", "kind": "hole", "description": "Tapering V slot between the two prongs, open at the tips.", "geometryEffect": "prongs offset +-Y with a small splay angle", "evidenceRefs": ["zone-r0c0"], "confidence": 0.85}], "surfaceDetail": {"macroRoughness": 0.46, "microRoughness": 0.08, "bumpAmplitude": 0.08, "normalPattern": "forging scale, fine brushing", "displacementPattern": "none", "occlusionPattern": "dark slot-facing inner face + slot root AO (P9)", "edgeWearPattern": "0.4 mm bevel catches edge highlight", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "claw-prong-lower", "dominantAlbedo": "rgba(92, 88, 89, 1.0)", "secondaryAlbedo": "rgba(106, 102, 102, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.683, "roughnessEstimate": 0.134, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/claw-faces.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.566}, "agentOverrideNote": "re-extracted at material-pass from material-evidence/crops/claw-faces.png (a fully-inside-mask window on the lit claw faces, median closest to the measured claw-top #7d7c7e); the first crop (claw-steel.png) was dominated by slot shadow (dominant rgba(13,8,11)); class metal by agent vision, extractor confidence capped <0.7"}};
  node_claw_prong_lower_3.add(mesh_claw_prong_lower_3);
  meshes["claw-prong-lower"] = mesh_claw_prong_lower_3;
  colliders["claw-prong-lower"] = {"type": "box", "offset": [0.012, 0.045, 0.005], "scale": [0.04, 0.07, 0.011], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_claw_prong_lower_3);

  const endpoint_shank_4 = makeAttachmentEndpoint(null);
  const node_shank_4 = new THREE.Group();
  node_shank_4.name = "Steel shank (one forging with the head)__pivot";
  node_shank_4.scale.set(1, 1, 1);
  if (endpoint_shank_4) {
    node_shank_4.position.copy(endpoint_shank_4.start);
    node_shank_4.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_shank_4.position.set(0.0, 0.0, 0.0);
    node_shank_4.rotation.set(0.0, 0.0, 0.0);
  }
  node_shank_4.userData.sculptComponent = {"id": "shank", "name": "Steel shank (one forging with the head)", "level": "macro", "role": "shank", "importance": 1.0, "confidence": 0.7, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Tapering oval/rounded-rect bar with a throat flare into the head: a sweep with varying section.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "sweep u around, v along shank", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.004, 0.0, -0.004], "rx": 0.017, "rz": 0.011, "squareness": 2.6}, {"position": [0.016, 0.0, -0.003], "rx": 0.0152, "rz": 0.0098, "squareness": 2.6}, {"position": [0.03, 0.0, -0.0015], "rx": 0.0115, "rz": 0.0086, "squareness": 2.4}, {"position": [0.05, 0.0, 0.0], "rx": 0.0092, "rz": 0.0078, "squareness": 2.3}, {"position": [0.08, 0.0, 0.0], "rx": 0.0085, "rz": 0.0074, "squareness": 2.3}, {"position": [0.115, 0.0, 0.0], "rx": 0.0081, "rz": 0.0072, "squareness": 2.3}, {"position": [0.13, 0.0, 0.0], "rx": 0.0079, "rz": 0.0071, "squareness": 2.3}], "radialSegments": 32, "capEnds": true}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-eye", "localStart": [0.004, 0.0, 0.0], "localEnd": [0.13, 0.0, 0.0], "contactType": "butt-forged", "embedDepth": 0.01, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-grip", "localPosition": [0.104, 0, 0], "axis": [1, 0, 0]}], "collider": {"type": "capsule", "offset": [0.06, 0, 0], "scale": [0.12, 0.02, 0.018], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "forged-steel"}}, "material": "forged-steel", "materialLayers": ["forged-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "throat-flare", "kind": "taper", "description": "Shank flares into the head and up under the claw (throat).", "geometryEffect": "large first-station section offset toward -Z", "evidenceRefs": ["zone-r0c0"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.36, "microRoughness": 0.12, "bumpAmplitude": 0.12, "normalPattern": "longitudinal grinding streaks", "displacementPattern": "none", "occlusionPattern": "throat fillet AO next to the eye (P9)", "edgeWearPattern": "white mill-scale scuff patch (projected decal)", "notes": ""}, "evidenceRefs": ["head-region", "full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "shank", "dominantAlbedo": "rgba(124, 119, 121, 1.0)", "secondaryAlbedo": "rgba(164, 160, 160, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.692, "roughnessEstimate": 0.122, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-steel-a.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.578}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-steel-a.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_shank_4.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-grip", "localPosition": [0.104, 0, 0], "axis": [1, 0, 0]}], "collider": {"type": "capsule", "offset": [0.06, 0, 0], "scale": [0.12, 0.02, 0.018], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "forged-steel"}};
  (nodes["head"] ?? root).add(node_shank_4);
  nodes["shank"] = node_shank_4;
  const mesh_shank_4Geometry = endpoint_shank_4
    ? new THREE.CylinderGeometry(endpoint_shank_4.endRadius, endpoint_shank_4.baseRadius, endpoint_shank_4.length, 32, 12)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.004, 0.0, -0.004], "rx": 0.017, "rz": 0.011, "squareness": 2.6}, {"position": [0.016, 0.0, -0.003], "rx": 0.0152, "rz": 0.0098, "squareness": 2.6}, {"position": [0.03, 0.0, -0.0015], "rx": 0.0115, "rz": 0.0086, "squareness": 2.4}, {"position": [0.05, 0.0, 0.0], "rx": 0.0092, "rz": 0.0078, "squareness": 2.3}, {"position": [0.08, 0.0, 0.0], "rx": 0.0085, "rz": 0.0074, "squareness": 2.3}, {"position": [0.115, 0.0, 0.0], "rx": 0.0081, "rz": 0.0072, "squareness": 2.3}, {"position": [0.13, 0.0, 0.0], "rx": 0.0079, "rz": 0.0071, "squareness": 2.3}], "radialSegments": 32, "capEnds": true});
  if (!endpoint_shank_4) {
    mesh_shank_4Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_shank_4 = new THREE.Mesh(
    mesh_shank_4Geometry,
    materialMap["forged-steel"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shank_4.name = "Steel shank (one forging with the head)";
  if (endpoint_shank_4) {
    mesh_shank_4.position.copy(endpoint_shank_4.midpoint);
    mesh_shank_4.quaternion.copy(endpoint_shank_4.quaternion);
  }
  mesh_shank_4.castShadow = options.castShadow ?? true;
  mesh_shank_4.receiveShadow = options.receiveShadow ?? true;
  mesh_shank_4.userData.sculptComponent = {"id": "shank", "name": "Steel shank (one forging with the head)", "level": "macro", "role": "shank", "importance": 1.0, "confidence": 0.7, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Tapering oval/rounded-rect bar with a throat flare into the head: a sweep with varying section.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "sweep u around, v along shank", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.004, 0.0, -0.004], "rx": 0.017, "rz": 0.011, "squareness": 2.6}, {"position": [0.016, 0.0, -0.003], "rx": 0.0152, "rz": 0.0098, "squareness": 2.6}, {"position": [0.03, 0.0, -0.0015], "rx": 0.0115, "rz": 0.0086, "squareness": 2.4}, {"position": [0.05, 0.0, 0.0], "rx": 0.0092, "rz": 0.0078, "squareness": 2.3}, {"position": [0.08, 0.0, 0.0], "rx": 0.0085, "rz": 0.0074, "squareness": 2.3}, {"position": [0.115, 0.0, 0.0], "rx": 0.0081, "rz": 0.0072, "squareness": 2.3}, {"position": [0.13, 0.0, 0.0], "rx": 0.0079, "rz": 0.0071, "squareness": 2.3}], "radialSegments": 32, "capEnds": true}}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-eye", "localStart": [0.004, 0.0, 0.0], "localEnd": [0.13, 0.0, 0.0], "contactType": "butt-forged", "embedDepth": 0.01, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-grip", "localPosition": [0.104, 0, 0], "axis": [1, 0, 0]}], "collider": {"type": "capsule", "offset": [0.06, 0, 0], "scale": [0.12, 0.02, 0.018], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "forged-steel"}}, "material": "forged-steel", "materialLayers": ["forged-steel"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "throat-flare", "kind": "taper", "description": "Shank flares into the head and up under the claw (throat).", "geometryEffect": "large first-station section offset toward -Z", "evidenceRefs": ["zone-r0c0"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.36, "microRoughness": 0.12, "bumpAmplitude": 0.12, "normalPattern": "longitudinal grinding streaks", "displacementPattern": "none", "occlusionPattern": "throat fillet AO next to the eye (P9)", "edgeWearPattern": "white mill-scale scuff patch (projected decal)", "notes": ""}, "evidenceRefs": ["head-region", "full-object"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "shank", "dominantAlbedo": "rgba(124, 119, 121, 1.0)", "secondaryAlbedo": "rgba(164, 160, 160, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.692, "roughnessEstimate": 0.122, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-steel-a.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.578}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-steel-a.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_shank_4.add(mesh_shank_4);
  meshes["shank"] = mesh_shank_4;
  colliders["shank"] = {"type": "capsule", "offset": [0.06, 0, 0], "scale": [0.12, 0.02, 0.018], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_shank_4);
  const socket_shank_socket_grip_0 = new THREE.Object3D();
  socket_shank_socket_grip_0.name = "socket-grip";
  socket_shank_socket_grip_0.position.set(0.104, 0.0, 0.0);
  socket_shank_socket_grip_0.rotation.set(0, 0, 0);
  socket_shank_socket_grip_0.userData.socket = {"id": "socket-grip", "localPosition": [0.104, 0, 0], "axis": [1, 0, 0]};
  node_shank_4.add(socket_shank_socket_grip_0);
  sockets["shank:socket-grip"] = socket_shank_socket_grip_0;

  const endpoint_grip_5 = makeAttachmentEndpoint(null);
  const node_grip_5 = new THREE.Group();
  node_grip_5.name = "Orange elastomer grip sleeve__pivot";
  node_grip_5.scale.set(1, 1, 1);
  if (endpoint_grip_5) {
    node_grip_5.position.copy(endpoint_grip_5.start);
    node_grip_5.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_grip_5.position.set(0.0, 0.0, 0.0);
    node_grip_5.rotation.set(0.0, 0.0, 0.0);
  }
  node_grip_5.userData.sculptComponent = {"id": "grip", "name": "Orange elastomer grip sleeve", "level": "macro", "role": "grip", "importance": 1.0, "confidence": 0.8, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Moulded sleeve with rounded-rect section widening to a flared butt: varying-section sweep.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "sweep u around, v along grip", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.104, 0.0, 0.0], "rx": 0.0072, "rz": 0.0066, "squareness": 2.4}, {"position": [0.11, 0.0, 0.0], "rx": 0.0113, "rz": 0.0099, "squareness": 2.6}, {"position": [0.118, 0.0, 0.0], "rx": 0.0123, "rz": 0.0107, "squareness": 2.8}, {"position": [0.15, 0.0, 0.0], "rx": 0.0138, "rz": 0.0118, "squareness": 2.8}, {"position": [0.2, 0.0, 0.0008], "rx": 0.0143, "rz": 0.0121, "squareness": 2.8}, {"position": [0.25, 0.0, 0.0018], "rx": 0.015, "rz": 0.0126, "squareness": 2.8}, {"position": [0.275, 0.0, 0.0025], "rx": 0.016, "rz": 0.0133, "squareness": 2.8}, {"position": [0.292, 0.0, 0.003], "rx": 0.0175, "rz": 0.0143, "squareness": 2.8}, {"position": [0.3, 0.0, 0.0032], "rx": 0.0182, "rz": 0.0148, "squareness": 2.8}], "radialSegments": 48, "capEnds": true}}, "parent": "shank", "attachment": {"parentId": "shank", "parentSocket": "socket-grip", "localStart": [0.104, 0.0, 0.0], "localEnd": [0.3, 0.0, 0.0], "contactType": "overlap-sleeve", "embedDepth": 0.006, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.8}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-hand", "localPosition": [0.21, 0, 0], "axis": [1, 0, 0], "purpose": "hand grip: swing pivot (wrist) for the strike animation"}], "collider": {"type": "capsule", "offset": [0.21, 0, 0], "scale": [0.2, 0.03, 0.03], "isTrigger": false, "notes": "grip capsule proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "grip", "seamRefs": ["sleeve-shank interface (overmould bond)"], "detachableFragments": ["grip", "grip-collar", "grip-butt", "grip-logo-panel", "grip-print-marks"], "breakImpulse": 120.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [{"id": "sleeve-shank interface", "between": ["grip", "shank"], "type": "overmould bond", "evidenceRefs": ["grip-region"]}], "localFeatures": [{"id": "grip-rib-grooves", "kind": "groove", "description": "~26 chevron rib grooves (repetition system grip-rib-grooves).", "geometryEffect": "carved radial displacement + cavity AO", "evidenceRefs": ["zone-r1c1"], "confidence": 0.8}, {"id": "lanyard-hole", "kind": "hole", "description": "Lanyard hole near the butt.", "geometryEffect": "real blind recess carved into the grip on both lateral faces (radius 0.0020 m, depth 0.0040 m) + cavity darkening", "evidenceRefs": ["zone-r2c2"], "confidence": 0.75, "recess": {"x": 0.2863, "z": -0.0011, "radius": 0.002, "depth": 0.004, "cavityBoost": 1.75}}, {"id": "grip-logo-panel", "kind": "seam", "description": "Recessed rectangular logo panel on the +Y face; grooves interrupted.", "geometryEffect": "groove mask + inset", "evidenceRefs": ["zone-r2c2"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "geometric rib grooves (carved), satin moulded skin", "displacementPattern": "carved S-chevron grooves 0.6 mm deep, logo-panel inset 0.4 mm, lanyard recess 3 mm, parting line 0.35 mm", "occlusionPattern": "groove/lanyard/parting-line cavity vertex colour (P2)", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "grip", "dominantAlbedo": "rgba(235, 84, 24, 1.0)", "secondaryAlbedo": "rgba(239, 97, 42, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.123, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-grip-b.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.515}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-grip-b.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_5.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-hand", "localPosition": [0.21, 0, 0], "axis": [1, 0, 0], "purpose": "hand grip: swing pivot (wrist) for the strike animation"}], "collider": {"type": "capsule", "offset": [0.21, 0, 0], "scale": [0.2, 0.03, 0.03], "isTrigger": false, "notes": "grip capsule proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "grip", "seamRefs": ["sleeve-shank interface (overmould bond)"], "detachableFragments": ["grip", "grip-collar", "grip-butt", "grip-logo-panel", "grip-print-marks"], "breakImpulse": 120.0, "debrisMaterial": "grip-elastomer"}};
  (nodes["shank"] ?? root).add(node_grip_5);
  nodes["grip"] = node_grip_5;
  const mesh_grip_5Geometry = endpoint_grip_5
    ? new THREE.CylinderGeometry(endpoint_grip_5.endRadius, endpoint_grip_5.baseRadius, endpoint_grip_5.length, 32, 12)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.104, 0.0, 0.0], "rx": 0.0072, "rz": 0.0066, "squareness": 2.4}, {"position": [0.11, 0.0, 0.0], "rx": 0.0113, "rz": 0.0099, "squareness": 2.6}, {"position": [0.118, 0.0, 0.0], "rx": 0.0123, "rz": 0.0107, "squareness": 2.8}, {"position": [0.15, 0.0, 0.0], "rx": 0.0138, "rz": 0.0118, "squareness": 2.8}, {"position": [0.2, 0.0, 0.0008], "rx": 0.0143, "rz": 0.0121, "squareness": 2.8}, {"position": [0.25, 0.0, 0.0018], "rx": 0.015, "rz": 0.0126, "squareness": 2.8}, {"position": [0.275, 0.0, 0.0025], "rx": 0.016, "rz": 0.0133, "squareness": 2.8}, {"position": [0.292, 0.0, 0.003], "rx": 0.0175, "rz": 0.0143, "squareness": 2.8}, {"position": [0.3, 0.0, 0.0032], "rx": 0.0182, "rz": 0.0148, "squareness": 2.8}], "radialSegments": 48, "capEnds": true});
  if (!endpoint_grip_5) {
    mesh_grip_5Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_grip_5 = new THREE.Mesh(
    mesh_grip_5Geometry,
    materialMap["grip-elastomer"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_grip_5.name = "Orange elastomer grip sleeve";
  if (endpoint_grip_5) {
    mesh_grip_5.position.copy(endpoint_grip_5.midpoint);
    mesh_grip_5.quaternion.copy(endpoint_grip_5.quaternion);
  }
  mesh_grip_5.castShadow = options.castShadow ?? true;
  mesh_grip_5.receiveShadow = options.receiveShadow ?? true;
  mesh_grip_5.userData.sculptComponent = {"id": "grip", "name": "Orange elastomer grip sleeve", "level": "macro", "role": "grip", "importance": 1.0, "confidence": 0.8, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Moulded sleeve with rounded-rect section widening to a flared butt: varying-section sweep.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "sweep u around, v along grip", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.104, 0.0, 0.0], "rx": 0.0072, "rz": 0.0066, "squareness": 2.4}, {"position": [0.11, 0.0, 0.0], "rx": 0.0113, "rz": 0.0099, "squareness": 2.6}, {"position": [0.118, 0.0, 0.0], "rx": 0.0123, "rz": 0.0107, "squareness": 2.8}, {"position": [0.15, 0.0, 0.0], "rx": 0.0138, "rz": 0.0118, "squareness": 2.8}, {"position": [0.2, 0.0, 0.0008], "rx": 0.0143, "rz": 0.0121, "squareness": 2.8}, {"position": [0.25, 0.0, 0.0018], "rx": 0.015, "rz": 0.0126, "squareness": 2.8}, {"position": [0.275, 0.0, 0.0025], "rx": 0.016, "rz": 0.0133, "squareness": 2.8}, {"position": [0.292, 0.0, 0.003], "rx": 0.0175, "rz": 0.0143, "squareness": 2.8}, {"position": [0.3, 0.0, 0.0032], "rx": 0.0182, "rz": 0.0148, "squareness": 2.8}], "radialSegments": 48, "capEnds": true}}, "parent": "shank", "attachment": {"parentId": "shank", "parentSocket": "socket-grip", "localStart": [0.104, 0.0, 0.0], "localEnd": [0.3, 0.0, 0.0], "contactType": "overlap-sleeve", "embedDepth": 0.006, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.8}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [{"id": "socket-hand", "localPosition": [0.21, 0, 0], "axis": [1, 0, 0], "purpose": "hand grip: swing pivot (wrist) for the strike animation"}], "collider": {"type": "capsule", "offset": [0.21, 0, 0], "scale": [0.2, 0.03, 0.03], "isTrigger": false, "notes": "grip capsule proxy"}, "constraints": [], "destruction": {"breakable": true, "fractureGroup": "grip", "seamRefs": ["sleeve-shank interface (overmould bond)"], "detachableFragments": ["grip", "grip-collar", "grip-butt", "grip-logo-panel", "grip-print-marks"], "breakImpulse": 120.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [{"id": "sleeve-shank interface", "between": ["grip", "shank"], "type": "overmould bond", "evidenceRefs": ["grip-region"]}], "localFeatures": [{"id": "grip-rib-grooves", "kind": "groove", "description": "~26 chevron rib grooves (repetition system grip-rib-grooves).", "geometryEffect": "carved radial displacement + cavity AO", "evidenceRefs": ["zone-r1c1"], "confidence": 0.8}, {"id": "lanyard-hole", "kind": "hole", "description": "Lanyard hole near the butt.", "geometryEffect": "real blind recess carved into the grip on both lateral faces (radius 0.0020 m, depth 0.0040 m) + cavity darkening", "evidenceRefs": ["zone-r2c2"], "confidence": 0.75, "recess": {"x": 0.2863, "z": -0.0011, "radius": 0.002, "depth": 0.004, "cavityBoost": 1.75}}, {"id": "grip-logo-panel", "kind": "seam", "description": "Recessed rectangular logo panel on the +Y face; grooves interrupted.", "geometryEffect": "groove mask + inset", "evidenceRefs": ["zone-r2c2"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "geometric rib grooves (carved), satin moulded skin", "displacementPattern": "carved S-chevron grooves 0.6 mm deep, logo-panel inset 0.4 mm, lanyard recess 3 mm, parting line 0.35 mm", "occlusionPattern": "groove/lanyard/parting-line cavity vertex colour (P2)", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "blockout", "colorMaterialRecipe": {"componentId": "grip", "dominantAlbedo": "rgba(235, 84, 24, 1.0)", "secondaryAlbedo": "rgba(239, 97, 42, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.123, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/c-grip-b.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.515}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/c-grip-b.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_5.add(mesh_grip_5);
  meshes["grip"] = mesh_grip_5;
  colliders["grip"] = {"type": "capsule", "offset": [0.21, 0, 0], "scale": [0.2, 0.03, 0.03], "isTrigger": false, "notes": "grip capsule proxy"};
  destructionGroups["grip"] ??= [];
  destructionGroups["grip"].push(node_grip_5);
  const socket_grip_socket_hand_0 = new THREE.Object3D();
  socket_grip_socket_hand_0.name = "socket-hand";
  socket_grip_socket_hand_0.position.set(0.21, 0.0, 0.0);
  socket_grip_socket_hand_0.rotation.set(0, 0, 0);
  socket_grip_socket_hand_0.userData.socket = {"id": "socket-hand", "localPosition": [0.21, 0, 0], "axis": [1, 0, 0], "purpose": "hand grip: swing pivot (wrist) for the strike animation"};
  node_grip_5.add(socket_grip_socket_hand_0);
  sockets["grip:socket-hand"] = socket_grip_socket_hand_0;

  const endpoint_grip_collar_6 = makeAttachmentEndpoint(null);
  const node_grip_collar_6 = new THREE.Group();
  node_grip_collar_6.name = "Grip front collar lip__pivot";
  node_grip_collar_6.scale.set(1, 1, 1);
  if (endpoint_grip_collar_6) {
    node_grip_collar_6.position.copy(endpoint_grip_collar_6.start);
    node_grip_collar_6.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_grip_collar_6.position.set(0.0, 0.0, 0.0);
    node_grip_collar_6.rotation.set(0.0, 0.0, 0.0);
  }
  node_grip_collar_6.userData.sculptComponent = {"id": "grip-collar", "name": "Grip front collar lip", "level": "meso", "role": "collar", "importance": 0.7, "confidence": 0.7, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Short rounded lip where the steel enters the sleeve.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.103, 0.0, 0.0], "rx": 0.0066, "rz": 0.006, "squareness": 2.2}, {"position": [0.109, 0.0, 0.0], "rx": 0.0116, "rz": 0.0101, "squareness": 2.6}, {"position": [0.114, 0.0, 0.0], "rx": 0.0121, "rz": 0.0105, "squareness": 2.6}, {"position": [0.118, 0.0, 0.0], "rx": 0.0121, "rz": 0.0105, "squareness": 2.6}], "radialSegments": 40, "capEnds": true}}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-grip", "localStart": [0.103, 0.0, 0.0], "localEnd": [0.118, 0.0, 0.0], "contactType": "overlap-sleeve", "embedDepth": 0.003, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "cylinder", "offset": [0.11, 0, 0], "scale": [0.012, 0.026, 0.022], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "grip-collar", "kind": "seam", "description": "Rounded lip at the sleeve mouth.", "geometryEffect": "short sweep proud of the grip", "evidenceRefs": ["zone-r1c1"], "confidence": 0.75}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "smooth moulded lip", "displacementPattern": "none", "occlusionPattern": "groove/lanyard/parting-line cavity vertex colour (P2)", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "structural-pass", "colorMaterialRecipe": {"componentId": "grip-collar", "dominantAlbedo": "rgba(231, 90, 38, 1.0)", "secondaryAlbedo": "rgba(232, 87, 26, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.87, "metalnessEstimate": 0.0, "highlightEvidence": "broad, gradually-fading highlight — supports high roughness/diffuse response", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/grip-collar.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.515}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/grip-collar.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_collar_6.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "cylinder", "offset": [0.11, 0, 0], "scale": [0.012, 0.026, 0.022], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}};
  (nodes["grip"] ?? root).add(node_grip_collar_6);
  nodes["grip-collar"] = node_grip_collar_6;
  const mesh_grip_collar_6Geometry = endpoint_grip_collar_6
    ? new THREE.CylinderGeometry(endpoint_grip_collar_6.endRadius, endpoint_grip_collar_6.baseRadius, endpoint_grip_collar_6.length, 32, 12)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.103, 0.0, 0.0], "rx": 0.0066, "rz": 0.006, "squareness": 2.2}, {"position": [0.109, 0.0, 0.0], "rx": 0.0116, "rz": 0.0101, "squareness": 2.6}, {"position": [0.114, 0.0, 0.0], "rx": 0.0121, "rz": 0.0105, "squareness": 2.6}, {"position": [0.118, 0.0, 0.0], "rx": 0.0121, "rz": 0.0105, "squareness": 2.6}], "radialSegments": 40, "capEnds": true});
  if (!endpoint_grip_collar_6) {
    mesh_grip_collar_6Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_grip_collar_6 = new THREE.Mesh(
    mesh_grip_collar_6Geometry,
    materialMap["grip-elastomer"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_grip_collar_6.name = "Grip front collar lip";
  if (endpoint_grip_collar_6) {
    mesh_grip_collar_6.position.copy(endpoint_grip_collar_6.midpoint);
    mesh_grip_collar_6.quaternion.copy(endpoint_grip_collar_6.quaternion);
  }
  mesh_grip_collar_6.castShadow = options.castShadow ?? true;
  mesh_grip_collar_6.receiveShadow = options.receiveShadow ?? true;
  mesh_grip_collar_6.userData.sculptComponent = {"id": "grip-collar", "name": "Grip front collar lip", "level": "meso", "role": "collar", "importance": 0.7, "confidence": 0.7, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Short rounded lip where the steel enters the sleeve.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.103, 0.0, 0.0], "rx": 0.0066, "rz": 0.006, "squareness": 2.2}, {"position": [0.109, 0.0, 0.0], "rx": 0.0116, "rz": 0.0101, "squareness": 2.6}, {"position": [0.114, 0.0, 0.0], "rx": 0.0121, "rz": 0.0105, "squareness": 2.6}, {"position": [0.118, 0.0, 0.0], "rx": 0.0121, "rz": 0.0105, "squareness": 2.6}], "radialSegments": 40, "capEnds": true}}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-grip", "localStart": [0.103, 0.0, 0.0], "localEnd": [0.118, 0.0, 0.0], "contactType": "overlap-sleeve", "embedDepth": 0.003, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.7}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "cylinder", "offset": [0.11, 0, 0], "scale": [0.012, 0.026, 0.022], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "grip-collar", "kind": "seam", "description": "Rounded lip at the sleeve mouth.", "geometryEffect": "short sweep proud of the grip", "evidenceRefs": ["zone-r1c1"], "confidence": 0.75}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "smooth moulded lip", "displacementPattern": "none", "occlusionPattern": "groove/lanyard/parting-line cavity vertex colour (P2)", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "structural-pass", "colorMaterialRecipe": {"componentId": "grip-collar", "dominantAlbedo": "rgba(231, 90, 38, 1.0)", "secondaryAlbedo": "rgba(232, 87, 26, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.87, "metalnessEstimate": 0.0, "highlightEvidence": "broad, gradually-fading highlight — supports high roughness/diffuse response", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/grip-collar.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.515}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/grip-collar.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_collar_6.add(mesh_grip_collar_6);
  meshes["grip-collar"] = mesh_grip_collar_6;
  colliders["grip-collar"] = {"type": "cylinder", "offset": [0.11, 0, 0], "scale": [0.012, 0.026, 0.022], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"};
  destructionGroups["grip"] ??= [];
  destructionGroups["grip"].push(node_grip_collar_6);

  const endpoint_grip_butt_7 = makeAttachmentEndpoint(null);
  const node_grip_butt_7 = new THREE.Group();
  node_grip_butt_7.name = "Grip butt flare and end cap__pivot";
  node_grip_butt_7.scale.set(1, 1, 1);
  if (endpoint_grip_butt_7) {
    node_grip_butt_7.position.copy(endpoint_grip_butt_7.start);
    node_grip_butt_7.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_grip_butt_7.position.set(0.0, 0.0, 0.0);
    node_grip_butt_7.rotation.set(0.0, 0.0, 0.0);
  }
  node_grip_butt_7.userData.sculptComponent = {"id": "grip-butt", "name": "Grip butt flare and end cap", "level": "meso", "role": "butt", "importance": 0.7, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Bulbous oval butt rounding into a flat end cap.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.294, 0.0, 0.0033], "rx": 0.0179, "rz": 0.0146, "squareness": 2.8}, {"position": [0.299, 0.0, 0.0033], "rx": 0.0184, "rz": 0.0149, "squareness": 2.8}, {"position": [0.3025, 0.0, 0.0033], "rx": 0.0178, "rz": 0.0144, "squareness": 2.6}, {"position": [0.3048, 0.0, 0.0033], "rx": 0.016, "rz": 0.0128, "squareness": 2.4}, {"position": [0.3058, 0.0, 0.0033], "rx": 0.009, "rz": 0.0072, "squareness": 2.0}], "radialSegments": 48, "capEnds": true}}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-hand", "localStart": [0.294, 0.0, 0.0], "localEnd": [0.3058, 0.0, 0.0], "contactType": "overlap-moulded", "embedDepth": 0.004, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.75}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "capsule", "offset": [0.3, 0, 0.003], "scale": [0.014, 0.037, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "butt-flare", "kind": "ridge", "description": "Flared oval butt with flat end cap.", "geometryEffect": "sweep stations widen then round off", "evidenceRefs": ["zone-r2c2"], "confidence": 0.8}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "smooth moulded end cap", "displacementPattern": "none", "occlusionPattern": "groove/lanyard/parting-line cavity vertex colour (P2)", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "structural-pass", "colorMaterialRecipe": {"componentId": "grip-butt", "dominantAlbedo": "rgba(238, 89, 28, 1.0)", "secondaryAlbedo": "rgba(230, 105, 69, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.362, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/grip-butt.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.686}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/grip-butt.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_butt_7.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "capsule", "offset": [0.3, 0, 0.003], "scale": [0.014, 0.037, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}};
  (nodes["grip"] ?? root).add(node_grip_butt_7);
  nodes["grip-butt"] = node_grip_butt_7;
  const mesh_grip_butt_7Geometry = endpoint_grip_butt_7
    ? new THREE.CylinderGeometry(endpoint_grip_butt_7.endRadius, endpoint_grip_butt_7.baseRadius, endpoint_grip_butt_7.length, 32, 12)
    : buildTaperedSweepGeometry({"stations": [{"position": [0.294, 0.0, 0.0033], "rx": 0.0179, "rz": 0.0146, "squareness": 2.8}, {"position": [0.299, 0.0, 0.0033], "rx": 0.0184, "rz": 0.0149, "squareness": 2.8}, {"position": [0.3025, 0.0, 0.0033], "rx": 0.0178, "rz": 0.0144, "squareness": 2.6}, {"position": [0.3048, 0.0, 0.0033], "rx": 0.016, "rz": 0.0128, "squareness": 2.4}, {"position": [0.3058, 0.0, 0.0033], "rx": 0.009, "rz": 0.0072, "squareness": 2.0}], "radialSegments": 48, "capEnds": true});
  if (!endpoint_grip_butt_7) {
    mesh_grip_butt_7Geometry.scale(1.0, 1.0, 1.0);
  }
  const mesh_grip_butt_7 = new THREE.Mesh(
    mesh_grip_butt_7Geometry,
    materialMap["grip-elastomer"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_grip_butt_7.name = "Grip butt flare and end cap";
  if (endpoint_grip_butt_7) {
    mesh_grip_butt_7.position.copy(endpoint_grip_butt_7.midpoint);
    mesh_grip_butt_7.quaternion.copy(endpoint_grip_butt_7.quaternion);
  }
  mesh_grip_butt_7.castShadow = options.castShadow ?? true;
  mesh_grip_butt_7.receiveShadow = options.receiveShadow ?? true;
  mesh_grip_butt_7.userData.sculptComponent = {"id": "grip-butt", "name": "Grip butt flare and end cap", "level": "meso", "role": "butt", "importance": 0.7, "confidence": 0.75, "primitive": "tapered-sweep", "topologyClass": "continuous-sculpt", "topologyRationale": "Bulbous oval butt rounding into a flat end cap.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry", "taperedSweep": {"stations": [{"position": [0.294, 0.0, 0.0033], "rx": 0.0179, "rz": 0.0146, "squareness": 2.8}, {"position": [0.299, 0.0, 0.0033], "rx": 0.0184, "rz": 0.0149, "squareness": 2.8}, {"position": [0.3025, 0.0, 0.0033], "rx": 0.0178, "rz": 0.0144, "squareness": 2.6}, {"position": [0.3048, 0.0, 0.0033], "rx": 0.016, "rz": 0.0128, "squareness": 2.4}, {"position": [0.3058, 0.0, 0.0033], "rx": 0.009, "rz": 0.0072, "squareness": 2.0}], "radialSegments": 48, "capEnds": true}}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-hand", "localStart": [0.294, 0.0, 0.0], "localEnd": [0.3058, 0.0, 0.0], "contactType": "overlap-moulded", "embedDepth": 0.004, "gapTolerance": 0.0005, "contactNormal": [1, 0, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 1.0, "height": 1.0, "depth": 1.0, "units": "metres (geometry authored in absolute units; unit scale)", "confidence": 0.75}, "transform": {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "capsule", "offset": [0.3, 0, 0.003], "scale": [0.014, 0.037, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "butt-flare", "kind": "ridge", "description": "Flared oval butt with flat end cap.", "geometryEffect": "sweep stations widen then round off", "evidenceRefs": ["zone-r2c2"], "confidence": 0.8}], "surfaceDetail": {"macroRoughness": 0.5, "microRoughness": 0.05, "bumpAmplitude": 0.0, "normalPattern": "smooth moulded end cap", "displacementPattern": "none", "occlusionPattern": "groove/lanyard/parting-line cavity vertex colour (P2)", "edgeWearPattern": "none observed", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "structural-pass", "colorMaterialRecipe": {"componentId": "grip-butt", "dominantAlbedo": "rgba(238, 89, 28, 1.0)", "secondaryAlbedo": "rgba(230, 105, 69, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.362, "metalnessEstimate": 0.0, "highlightEvidence": "moderate hotspot spread — mid-range roughness", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/grip-butt.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.686}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/grip-butt.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_butt_7.add(mesh_grip_butt_7);
  meshes["grip-butt"] = mesh_grip_butt_7;
  colliders["grip-butt"] = {"type": "capsule", "offset": [0.3, 0, 0.003], "scale": [0.014, 0.037, 0.03], "isTrigger": false, "notes": "primitive proxy (compound hammer collider)"};
  destructionGroups["grip"] ??= [];
  destructionGroups["grip"].push(node_grip_butt_7);

  const endpoint_grip_logo_panel_8 = makeAttachmentEndpoint(null);
  const node_grip_logo_panel_8 = new THREE.Group();
  node_grip_logo_panel_8.name = "Grip logo panel__pivot";
  node_grip_logo_panel_8.scale.set(1, 1, 1);
  if (endpoint_grip_logo_panel_8) {
    node_grip_logo_panel_8.position.copy(endpoint_grip_logo_panel_8.start);
    node_grip_logo_panel_8.rotation.set(0.0, 0.0, 0.0);
  } else {
    node_grip_logo_panel_8.position.set(0.245, 0.0112, 0.004);
    node_grip_logo_panel_8.rotation.set(0.0, 0.0, 0.0);
  }
  node_grip_logo_panel_8.userData.sculptComponent = {"id": "grip-logo-panel", "name": "Grip logo panel", "level": "micro", "role": "panel", "importance": 0.4, "confidence": 0.6, "primitive": "box", "topologyClass": "surface-relief", "topologyRationale": "Flat rectangular panel on the grip face; a thin plate sitting flush with the rib crests.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "chamfer", "bevelRadius": 0.0005, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-hand", "localStart": [0.2075, 0.0112, 0.004], "localEnd": [0.2825, 0.0112, 0.004], "contactType": "flush-inset", "embedDepth": 0.0004, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.075, "height": 0.0012, "depth": 0.01, "units": "metres", "confidence": 0.6}, "transform": {"position": [0.245, 0.0112, 0.004], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "grip-logo-panel", "kind": "seam", "description": "Recessed rectangular panel with small embossed logo.", "geometryEffect": "thin flush plate; grooves masked under it", "evidenceRefs": ["zone-r2c2"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "form-refinement", "colorMaterialRecipe": {"componentId": "grip-logo-panel", "dominantAlbedo": "rgba(235, 88, 26, 1.0)", "secondaryAlbedo": "rgba(254, 252, 250, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.334, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/logo-panel.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.778}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/logo-panel.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_logo_panel_8.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}};
  (nodes["grip"] ?? root).add(node_grip_logo_panel_8);
  nodes["grip-logo-panel"] = node_grip_logo_panel_8;
  const mesh_grip_logo_panel_8Geometry = endpoint_grip_logo_panel_8
    ? new THREE.CylinderGeometry(endpoint_grip_logo_panel_8.endRadius, endpoint_grip_logo_panel_8.baseRadius, endpoint_grip_logo_panel_8.length, 32, 12)
    : new THREE.BoxGeometry(1, 1, 1, 12, 12, 12);
  if (!endpoint_grip_logo_panel_8) {
    mesh_grip_logo_panel_8Geometry.scale(0.075, 0.0012, 0.01);
  }
  const mesh_grip_logo_panel_8 = new THREE.Mesh(
    mesh_grip_logo_panel_8Geometry,
    materialMap["grip-elastomer"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_grip_logo_panel_8.name = "Grip logo panel";
  if (endpoint_grip_logo_panel_8) {
    mesh_grip_logo_panel_8.position.copy(endpoint_grip_logo_panel_8.midpoint);
    mesh_grip_logo_panel_8.quaternion.copy(endpoint_grip_logo_panel_8.quaternion);
  }
  mesh_grip_logo_panel_8.castShadow = options.castShadow ?? true;
  mesh_grip_logo_panel_8.receiveShadow = options.receiveShadow ?? true;
  mesh_grip_logo_panel_8.userData.sculptComponent = {"id": "grip-logo-panel", "name": "Grip logo panel", "level": "micro", "role": "panel", "importance": 0.4, "confidence": 0.6, "primitive": "box", "topologyClass": "surface-relief", "topologyRationale": "Flat rectangular panel on the grip face; a thin plate sitting flush with the rib crests.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "chamfer", "bevelRadius": 0.0005, "segments": 1}, "deformationStack": [], "uvStrategy": "generated procedural coordinates", "normalStrategy": "vertex normals from generated geometry"}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-hand", "localStart": [0.2075, 0.0112, 0.004], "localEnd": [0.2825, 0.0112, 0.004], "contactType": "flush-inset", "embedDepth": 0.0004, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.075, "height": 0.0012, "depth": 0.01, "units": "metres", "confidence": 0.6}, "transform": {"position": [0.245, 0.0112, 0.004], "rotation": [0.0, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "grip", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-elastomer"}}, "material": "grip-elastomer", "materialLayers": ["grip-elastomer"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "grip-logo-panel", "kind": "seam", "description": "Recessed rectangular panel with small embossed logo.", "geometryEffect": "thin flush plate; grooves masked under it", "evidenceRefs": ["zone-r2c2"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "form-refinement", "colorMaterialRecipe": {"componentId": "grip-logo-panel", "dominantAlbedo": "rgba(235, 88, 26, 1.0)", "secondaryAlbedo": "rgba(254, 252, 250, 1.0)", "materialClass": "rubber", "materialClassConfidence": 0.6, "roughnessEstimate": 0.334, "metalnessEstimate": 0.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/logo-panel.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.778}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/logo-panel.png; class rubber confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_grip_logo_panel_8.add(mesh_grip_logo_panel_8);
  meshes["grip-logo-panel"] = mesh_grip_logo_panel_8;
  colliders["grip-logo-panel"] = {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"};
  destructionGroups["grip"] ??= [];
  destructionGroups["grip"].push(node_grip_logo_panel_8);

  const endpoint_cheek_stamp_9 = makeAttachmentEndpoint(null);
  const node_cheek_stamp_9 = new THREE.Group();
  node_cheek_stamp_9.name = "Stamped cheek markings decal__pivot";
  node_cheek_stamp_9.scale.set(1, 1, 1);
  if (endpoint_cheek_stamp_9) {
    node_cheek_stamp_9.position.copy(endpoint_cheek_stamp_9.start);
    node_cheek_stamp_9.rotation.set(-1.5708, 0.0, 0.0);
  } else {
    node_cheek_stamp_9.position.set(-0.003, 0.0128, 0.014);
    node_cheek_stamp_9.rotation.set(-1.5708, 0.0, 0.0);
  }
  node_cheek_stamp_9.userData.sculptComponent = {"id": "cheek-stamp", "name": "Stamped cheek markings decal", "level": "micro", "role": "decal", "importance": 0.4, "confidence": 0.5, "primitive": "plane-card", "topologyClass": "material-only", "topologyRationale": "Printed/stamped marks ride on the cheek surface; a thin carrier shell for the stamp texture.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "orthographic projection of the reference crop (patched, see tools/patch_factory.py)", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-eye", "localStart": [-0.003, 0.0128, 0.014], "localEnd": [-0.003, 0.0128, 0.034], "contactType": "decal-on-surface", "embedDepth": 0.0002, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.02, "height": 0.02, "depth": 1.0, "units": "metres", "confidence": 0.5}, "transform": {"position": [-0.003, 0.0128, 0.014], "rotation": [-1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stamp-ink"}}, "material": "stamp-ink", "materialLayers": ["stamp-ink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "cheek-stamp", "kind": "decal", "description": "Stamped monogram + '8-62633 16oz TAIWAN DK'.", "geometryEffect": "decal shell", "evidenceRefs": ["zone-r0c0"], "confidence": 0.6}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "material-pass", "colorMaterialRecipe": {"componentId": "cheek-stamp", "dominantAlbedo": "rgba(106, 102, 103, 1.0)", "secondaryAlbedo": "rgba(158, 154, 155, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.684, "roughnessEstimate": 0.133, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/stamp-ink.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.478}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/stamp-ink.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_cheek_stamp_9.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stamp-ink"}};
  (nodes["head"] ?? root).add(node_cheek_stamp_9);
  nodes["cheek-stamp"] = node_cheek_stamp_9;
  const mesh_cheek_stamp_9Geometry = endpoint_cheek_stamp_9
    ? new THREE.CylinderGeometry(endpoint_cheek_stamp_9.endRadius, endpoint_cheek_stamp_9.baseRadius, endpoint_cheek_stamp_9.length, 32, 12)
    : new THREE.PlaneGeometry(1, 1, 24, 24);
  if (!endpoint_cheek_stamp_9) {
    mesh_cheek_stamp_9Geometry.scale(0.02, 0.02, 1.0);
  }
  const mesh_cheek_stamp_9 = new THREE.Mesh(
    mesh_cheek_stamp_9Geometry,
    materialMap["stamp-ink"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_cheek_stamp_9.name = "Stamped cheek markings decal";
  if (endpoint_cheek_stamp_9) {
    mesh_cheek_stamp_9.position.copy(endpoint_cheek_stamp_9.midpoint);
    mesh_cheek_stamp_9.quaternion.copy(endpoint_cheek_stamp_9.quaternion);
  }
  mesh_cheek_stamp_9.castShadow = options.castShadow ?? true;
  mesh_cheek_stamp_9.receiveShadow = options.receiveShadow ?? true;
  mesh_cheek_stamp_9.userData.sculptComponent = {"id": "cheek-stamp", "name": "Stamped cheek markings decal", "level": "micro", "role": "decal", "importance": 0.4, "confidence": 0.5, "primitive": "plane-card", "topologyClass": "material-only", "topologyRationale": "Printed/stamped marks ride on the cheek surface; a thin carrier shell for the stamp texture.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "orthographic projection of the reference crop (patched, see tools/patch_factory.py)", "normalStrategy": "vertex normals from generated geometry"}, "parent": "head", "attachment": {"parentId": "head", "parentSocket": "socket-eye", "localStart": [-0.003, 0.0128, 0.014], "localEnd": [-0.003, 0.0128, 0.034], "contactType": "decal-on-surface", "embedDepth": 0.0002, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.02, "height": 0.02, "depth": 1.0, "units": "metres", "confidence": 0.5}, "transform": {"position": [-0.003, 0.0128, 0.014], "rotation": [-1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "stamp-ink"}}, "material": "stamp-ink", "materialLayers": ["stamp-ink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "cheek-stamp", "kind": "decal", "description": "Stamped monogram + '8-62633 16oz TAIWAN DK'.", "geometryEffect": "decal shell", "evidenceRefs": ["zone-r0c0"], "confidence": 0.6}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "material-pass", "colorMaterialRecipe": {"componentId": "cheek-stamp", "dominantAlbedo": "rgba(106, 102, 103, 1.0)", "secondaryAlbedo": "rgba(158, 154, 155, 1.0)", "materialClass": "metal", "materialClassConfidence": 0.684, "roughnessEstimate": 0.133, "metalnessEstimate": 1.0, "highlightEvidence": "sharp, tight specular hotspot — supports low roughness/high specularity", "sourceCropPath": "/mnt/c/repos/img2three-runs/claw-hammer-20261117-132359/material-evidence/crops/stamp-ink.png", "labClusterMeta": {"clusterCount": 3, "dominantClusterSharePct": 0.478}, "agentOverrideNote": "extractor class confidence capped below 0.7 on crop material-evidence/crops/stamp-ink.png; class metal confirmed by agent vision (bare forged steel / moulded TPE); accepted with --allow-low-confidence"}};
  node_cheek_stamp_9.add(mesh_cheek_stamp_9);
  meshes["cheek-stamp"] = mesh_cheek_stamp_9;
  colliders["cheek-stamp"] = {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_cheek_stamp_9);

  const endpoint_shank_scuff_10 = makeAttachmentEndpoint(null);
  const node_shank_scuff_10 = new THREE.Group();
  node_shank_scuff_10.name = "Shank mill-scale scuff patch decal__pivot";
  node_shank_scuff_10.scale.set(1, 1, 1);
  if (endpoint_shank_scuff_10) {
    node_shank_scuff_10.position.copy(endpoint_shank_scuff_10.start);
    node_shank_scuff_10.rotation.set(-1.5708, 0.0, 0.0);
  } else {
    node_shank_scuff_10.position.set(0.05, 0.009, 0.0);
    node_shank_scuff_10.rotation.set(-1.5708, 0.0, 0.0);
  }
  node_shank_scuff_10.userData.sculptComponent = {"id": "shank-scuff", "name": "Shank mill-scale scuff patch decal", "level": "micro", "role": "decal", "importance": 0.4, "confidence": 0.7, "primitive": "plane-card", "topologyClass": "material-only", "topologyRationale": "Wear mark riding on the shank surface; thin conforming carrier for the projected reference pixels.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "camera projection of the reference crop (tools/patch_factory.py P6)", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shank", "attachment": {"parentId": "shank", "parentSocket": "socket-eye", "localStart": [0.01, 0.009, 0.0], "localEnd": [0.1, 0.009, 0.0], "contactType": "decal-on-surface", "embedDepth": 0.0002, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.09, "height": 0.02, "depth": 1.0, "units": "metres", "confidence": 0.5}, "transform": {"position": [0.05, 0.009, 0.0], "rotation": [-1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "scuff-zinc"}}, "material": "scuff-zinc", "materialLayers": ["scuff-zinc"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "shank-scuff-patch", "kind": "stain", "description": "White mill-scale/zinc scuff along the upper shank.", "geometryEffect": "projected decal shell", "evidenceRefs": ["zone-r1c0"], "confidence": 0.8}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "material-pass"};
  node_shank_scuff_10.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "scuff-zinc"}};
  (nodes["shank"] ?? root).add(node_shank_scuff_10);
  nodes["shank-scuff"] = node_shank_scuff_10;
  const mesh_shank_scuff_10Geometry = endpoint_shank_scuff_10
    ? new THREE.CylinderGeometry(endpoint_shank_scuff_10.endRadius, endpoint_shank_scuff_10.baseRadius, endpoint_shank_scuff_10.length, 32, 12)
    : new THREE.PlaneGeometry(1, 1, 24, 24);
  if (!endpoint_shank_scuff_10) {
    mesh_shank_scuff_10Geometry.scale(0.09, 0.02, 1.0);
  }
  const mesh_shank_scuff_10 = new THREE.Mesh(
    mesh_shank_scuff_10Geometry,
    materialMap["scuff-zinc"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_shank_scuff_10.name = "Shank mill-scale scuff patch decal";
  if (endpoint_shank_scuff_10) {
    mesh_shank_scuff_10.position.copy(endpoint_shank_scuff_10.midpoint);
    mesh_shank_scuff_10.quaternion.copy(endpoint_shank_scuff_10.quaternion);
  }
  mesh_shank_scuff_10.castShadow = options.castShadow ?? true;
  mesh_shank_scuff_10.receiveShadow = options.receiveShadow ?? true;
  mesh_shank_scuff_10.userData.sculptComponent = {"id": "shank-scuff", "name": "Shank mill-scale scuff patch decal", "level": "micro", "role": "decal", "importance": 0.4, "confidence": 0.7, "primitive": "plane-card", "topologyClass": "material-only", "topologyRationale": "Wear mark riding on the shank surface; thin conforming carrier for the projected reference pixels.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "camera projection of the reference crop (tools/patch_factory.py P6)", "normalStrategy": "vertex normals from generated geometry"}, "parent": "shank", "attachment": {"parentId": "shank", "parentSocket": "socket-eye", "localStart": [0.01, 0.009, 0.0], "localEnd": [0.1, 0.009, 0.0], "contactType": "decal-on-surface", "embedDepth": 0.0002, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.09, "height": 0.02, "depth": 1.0, "units": "metres", "confidence": 0.5}, "transform": {"position": [0.05, 0.009, 0.0], "rotation": [-1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "scuff-zinc"}}, "material": "scuff-zinc", "materialLayers": ["scuff-zinc"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "shank-scuff-patch", "kind": "stain", "description": "White mill-scale/zinc scuff along the upper shank.", "geometryEffect": "projected decal shell", "evidenceRefs": ["zone-r1c0"], "confidence": 0.8}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["head-region"], "details": [], "fidelityTier": "material-pass"};
  node_shank_scuff_10.add(mesh_shank_scuff_10);
  meshes["shank-scuff"] = mesh_shank_scuff_10;
  colliders["shank-scuff"] = {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_shank_scuff_10);

  const endpoint_grip_print_marks_11 = makeAttachmentEndpoint(null);
  const node_grip_print_marks_11 = new THREE.Group();
  node_grip_print_marks_11.name = "Grip panel logo and lanyard-hole shading decal__pivot";
  node_grip_print_marks_11.scale.set(1, 1, 1);
  if (endpoint_grip_print_marks_11) {
    node_grip_print_marks_11.position.copy(endpoint_grip_print_marks_11.start);
    node_grip_print_marks_11.rotation.set(-1.5708, 0.0, 0.0);
  } else {
    node_grip_print_marks_11.position.set(0.245, 0.013, 0.0);
    node_grip_print_marks_11.rotation.set(-1.5708, 0.0, 0.0);
  }
  node_grip_print_marks_11.userData.sculptComponent = {"id": "grip-print-marks", "name": "Grip panel logo and lanyard-hole shading decal", "level": "micro", "role": "decal", "importance": 0.4, "confidence": 0.6, "primitive": "plane-card", "topologyClass": "material-only", "topologyRationale": "Low-relief logo and the hole's dark interior as seen in the photo; thin conforming carrier for projected reference pixels over the carved recess.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "camera projection of the reference crop (tools/patch_factory.py P6)", "normalStrategy": "vertex normals from generated geometry"}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-hand", "localStart": [0.2, 0.013, 0.0], "localEnd": [0.3, 0.013, 0.0], "contactType": "decal-on-surface", "embedDepth": 0.0002, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.1, "height": 0.03, "depth": 1.0, "units": "metres", "confidence": 0.5}, "transform": {"position": [0.245, 0.013, 0.0], "rotation": [-1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-marks-ink"}}, "material": "grip-marks-ink", "materialLayers": ["grip-marks-ink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "grip-logo", "kind": "decal", "description": "Small embossed logo at the panel's proximal end.", "geometryEffect": "projected decal", "evidenceRefs": ["zone-r2c2"], "confidence": 0.5}, {"id": "lanyard-hole-shading", "kind": "hole", "description": "Dark interior of the lanyard hole (over the carved recess).", "geometryEffect": "projected decal", "evidenceRefs": ["zone-r2c2"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "surface-pass"};
  node_grip_print_marks_11.userData.actionProfile = {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-marks-ink"}};
  (nodes["grip"] ?? root).add(node_grip_print_marks_11);
  nodes["grip-print-marks"] = node_grip_print_marks_11;
  const mesh_grip_print_marks_11Geometry = endpoint_grip_print_marks_11
    ? new THREE.CylinderGeometry(endpoint_grip_print_marks_11.endRadius, endpoint_grip_print_marks_11.baseRadius, endpoint_grip_print_marks_11.length, 32, 12)
    : new THREE.PlaneGeometry(1, 1, 24, 24);
  if (!endpoint_grip_print_marks_11) {
    mesh_grip_print_marks_11Geometry.scale(0.1, 0.03, 1.0);
  }
  const mesh_grip_print_marks_11 = new THREE.Mesh(
    mesh_grip_print_marks_11Geometry,
    materialMap["grip-marks-ink"] ?? new THREE.MeshStandardMaterial({ color: 0x888888 })
  );
  mesh_grip_print_marks_11.name = "Grip panel logo and lanyard-hole shading decal";
  if (endpoint_grip_print_marks_11) {
    mesh_grip_print_marks_11.position.copy(endpoint_grip_print_marks_11.midpoint);
    mesh_grip_print_marks_11.quaternion.copy(endpoint_grip_print_marks_11.quaternion);
  }
  mesh_grip_print_marks_11.castShadow = options.castShadow ?? true;
  mesh_grip_print_marks_11.receiveShadow = options.receiveShadow ?? true;
  mesh_grip_print_marks_11.userData.sculptComponent = {"id": "grip-print-marks", "name": "Grip panel logo and lanyard-hole shading decal", "level": "micro", "role": "decal", "importance": 0.4, "confidence": 0.6, "primitive": "plane-card", "topologyClass": "material-only", "topologyRationale": "Low-relief logo and the hole's dark interior as seen in the photo; thin conforming carrier for projected reference pixels over the carved recess.", "geometryDescriptor": {"topologyIntent": "low-poly blockout with bevel-ready edges", "edgeTreatment": {"type": "none", "bevelRadius": 0.0, "segments": 1}, "deformationStack": [], "uvStrategy": "camera projection of the reference crop (tools/patch_factory.py P6)", "normalStrategy": "vertex normals from generated geometry"}, "parent": "grip", "attachment": {"parentId": "grip", "parentSocket": "socket-hand", "localStart": [0.2, 0.013, 0.0], "localEnd": [0.3, 0.013, 0.0], "contactType": "decal-on-surface", "embedDepth": 0.0002, "gapTolerance": 0.0005, "contactNormal": [0, 1, 0], "evidenceRefs": ["full-object"]}, "dimensions": {"width": 0.1, "height": 0.03, "depth": 1.0, "units": "metres", "confidence": 0.5}, "transform": {"position": [0.245, 0.013, 0.0], "rotation": [-1.5708, 0.0, 0.0]}, "actionProfile": {"animationRole": "rigid-child", "pivot": {"mode": "socket", "localPosition": [0, 0, 0], "axis": [0, 1, 0], "confidence": 0.7}, "transformChannels": {"translate": true, "rotate": true, "scale": true, "bend": false, "twist": false, "detach": true, "visibility": true, "materialState": true}, "sockets": [], "collider": {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"}, "constraints": [], "destruction": {"breakable": false, "fractureGroup": "forged-steel", "seamRefs": [], "detachableFragments": [], "breakImpulse": 0.0, "debrisMaterial": "grip-marks-ink"}}, "material": "grip-marks-ink", "materialLayers": ["grip-marks-ink"], "deformations": [], "joints": [], "seams": [], "localFeatures": [{"id": "grip-logo", "kind": "decal", "description": "Small embossed logo at the panel's proximal end.", "geometryEffect": "projected decal", "evidenceRefs": ["zone-r2c2"], "confidence": 0.5}, {"id": "lanyard-hole-shading", "kind": "hole", "description": "Dark interior of the lanyard hole (over the carved recess).", "geometryEffect": "projected decal", "evidenceRefs": ["zone-r2c2"], "confidence": 0.7}], "surfaceDetail": {"macroRoughness": 0.0, "microRoughness": 0.0, "bumpAmplitude": 0.0, "normalPattern": "", "displacementPattern": "", "occlusionPattern": "", "edgeWearPattern": "", "notes": ""}, "evidenceRefs": ["grip-region"], "details": [], "fidelityTier": "surface-pass"};
  node_grip_print_marks_11.add(mesh_grip_print_marks_11);
  meshes["grip-print-marks"] = mesh_grip_print_marks_11;
  colliders["grip-print-marks"] = {"type": "none", "offset": [0, 0, 0], "scale": [0, 0, 0], "isTrigger": false, "notes": "surface detail rides its parent's collider"};
  destructionGroups["forged-steel"] ??= [];
  destructionGroups["forged-steel"].push(node_grip_print_marks_11);

  applyHammerPatches(meshes);  // HAMMER-PATCH P3
  root.userData.sculptRuntime = { nodes, meshes, sockets, colliders, destructionGroups } satisfies ProceduralModelRuntime;
  attachHammerInteraction(root, meshes, HAMMER_PATCH_PARAMS.patch);  // HAMMER-PATCH P10
  root.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  root.userData.actionReadiness = {
    note: 'Use root.userData.sculptRuntime.nodes for transforms, sockets for attachments, colliders for physics proxies, and destructionGroups for breakable sets.',
  };
  return root;
}

export function createClawHammerLookDevLights(
  mode: 'neutral' | 'grazing' | 'reference' = 'neutral',
): THREE.Group {
  const lights = new THREE.Group();
  lights.name = "Claw hammer look-dev lights";
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
  lights.userData.lightingFromPhoto = [{"id": "key", "type": "directional", "position": [-0.4, 2.4, 0.8], "intensity": 1.1, "color": "#ffffff", "castShadow": false, "notes": "soft overhead studio key (photo: top faces brightest, no cast shadow on the white sweep)"}, {"id": "fill", "type": "directional", "position": [1.5, 0.8, 1.5], "intensity": 0.4, "color": "#ffffff"}, {"id": "hemi", "type": "hemisphere", "skyColor": "#ffffff", "groundColor": "#d8d8d8", "intensity": 0.35}, {"id": "environment", "environmentIntensity": 1.0}, {"id": "tone", "exposure": 1.0, "notes": "ACES filmic tone mapping (viewer), exposure 1.0"}, {"id": "ground", "backgroundColor": "#ffffff", "noTable": true, "tableColor": "#ffffff", "tableRoughness": 0.9, "notes": "seamless white studio sweep, no table plane; contact shadow / ambient occlusion baked as vertex-colour cavity AO in the factory patch"}];
  lights.userData.lookDevTargets = {"qualityPriority": "reference-fidelity", "materialPass": {"albedoPaletteRequired": true, "roughnessVariationRequired": true, "normalOrBumpRequired": true, "localOverridesRequired": true, "minimumTextureResolution": 1024, "preferredTextureResolution": 2048, "independentMapChannels": ["albedo", "roughness", "height", "normal", "ambient-occlusion"], "requiredSurfaceFrequencyBands": ["macro", "meso", "micro"], "geometryReliefRequiredWhenSilhouetteAffected": true, "referencePbrExtraction": {"requiredWhenSourceImagePresent": true, "targetThreshold": 0.7, "stopOnLowConfidence": true, "script": "forge/stage1_intake/extract_pbr_evidence.py", "acceptedLimitation": "single-image extraction is reference-derived inference, not exact photogrammetry"}, "mustAvoid": ["single flat albedo per material", "uniform roughness", "albedo texture reused as roughness/height/normal/AO", "single-frequency random noise", "plastic-looking smooth bark, stone, cloth, foliage, or aged material", "local color/detail described only in prose without material masks", "claiming exact PBR recovery when confidence is below the target threshold"]}, "lightingPass": {"requiredTerms": ["key light", "fill light", "rim or environment light", "exposure", "tone mapping", "background", "contact shadow"], "mustAvoid": ["ambient-only lighting", "flat value range", "missing contact shadow", "reference lighting copied without separating material readability"]}, "screenshotReview": ["Compare albedo palette and local color zones.", "Compare roughness/normal/bump response under light.", "Compare cavity dirt, edge wear, stains, moss, scratches, or other local masks.", "Compare key/fill/rim structure, exposure, tone mapping, background, and contact shadows.", "Capture a neutral-light render to verify material readability without reference lighting.", "Capture a grazing-light close-up to expose flat normals, uniform roughness, tiling, and plastic highlights.", "Capture a reference-matched render from the same camera framing as the source."]};
  return lights;
}

// PBR materials (clearcoat/iridescence/transmission/anisotropy) need an environment
// map to visually behave as intended — call this once per renderer and assign the
// result to scene.environment before rendering. No external HDR asset required.
export function createClawHammerEnvironment(renderer: THREE.WebGLRenderer): THREE.Texture {
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
export function frameClawHammerCamera(
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
export function createClawHammerPresentationComposer(
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

export function configureClawHammerRenderer(renderer: THREE.WebGLRenderer): void {
  // Load-bearing for view-dependent finishes (anodized / Doppler): without ACES + sRGB
  // the environment reflection reads flat/washed instead of a believable metal response.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
}

export function createClawHammerInspectControls(
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
