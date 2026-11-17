// Gallery for every published model: pick a model, orbit it, explode it, compare with its photo.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import SimpleBar from 'simplebar';
import 'simplebar/dist/simplebar.css';
import './gallery.css';
import { models, type LibraryModel } from '../models/index';

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const list = $<HTMLUListElement>('#model-list');
const stage = $<HTMLDivElement>('#stage');
const title = $<HTMLHeadingElement>('#model-title');
const facts = $<HTMLDListElement>('#model-facts');
const refImg = $<HTMLImageElement>('#reference');
const explodeInput = $<HTMLInputElement>('#explode');
const explodeRow = $<HTMLLabelElement>('#explode-row');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

let current: { group: THREE.Group; lights: THREE.Object3D | null; runtime: any } | null = null;

function applyTheme() {
  const dark = matchMedia('(prefers-color-scheme: dark)').matches;
  scene.background = new THREE.Color(dark ? '#1c1d21' : '#e9e7e2');
}

function bindEnvironment(root: THREE.Object3D) {
  // three r163+ ignores material.envMapIntensity unless the material has its own envMap
  root.traverse((o: any) => {
    if (!o.isMesh) return;
    for (const m of Array.isArray(o.material) ? o.material : [o.material]) {
      if (m && typeof m.envMapIntensity === 'number' && m.envMapIntensity !== 1 && !m.envMap) {
        m.envMap = scene.environment;
        m.needsUpdate = true;
      }
    }
  });
}

function frame(object: THREE.Object3D) {
  const box = new THREE.Box3().setFromObject(object);
  const centre = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2;
  const dist = radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.1;
  camera.near = dist / 100;
  camera.far = dist * 20;
  camera.position.copy(centre).add(new THREE.Vector3(0.55, 0.45, 1).normalize().multiplyScalar(dist));
  camera.updateProjectionMatrix();
  controls.target.copy(centre);
  controls.update();
}

function fmt(v: number | null | undefined, digits = 0, unit = '') {
  return typeof v === 'number' ? `${v.toFixed(digits)}${unit}` : '–';
}

function show(model: LibraryModel) {
  if (current) {
    scene.remove(current.group);
    if (current.lights) scene.remove(current.lights);
  }
  const group = model.create({ castShadow: true, receiveShadow: true });
  bindEnvironment(group);
  const makeLights = Object.entries(model.module).find(([k, v]) => /^create\w+LookDevLights$/.test(k) && typeof v === 'function');
  const lights = makeLights ? (makeLights[1] as (mode: string) => THREE.Object3D)('neutral') : null;
  if (lights) scene.add(lights);
  else scene.add(new THREE.HemisphereLight('#ffffff', '#555555', 1.2));
  scene.add(group);
  const runtime = group.userData.sculptRuntime;
  current = { group, lights, runtime };
  frame(group);

  let triangles = 0;
  group.traverse((o: any) => {
    if (o.isMesh) triangles += o.geometry.index ? o.geometry.index.count / 3 : o.geometry.attributes.position.count / 3;
  });
  title.textContent = model.title;
  const rows: [string, string][] = [
    ['Fidelity (agent)', fmt(model.info.fidelityEstimate, 2)],
    ['Triangles', Math.round(triangles).toLocaleString('sv-SE')],
    ['Agent turns', fmt(model.info.turns)],
    ['API-equivalent cost', fmt(model.info.apiEquivalentCostUsd, 2, ' USD')],
    ['Run time', fmt(model.info.durationMinutes, 0, ' min')],
    ['Published', model.info.published ?? '–'],
  ];
  facts.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    return [dt, dd];
  }));
  refImg.hidden = !model.info.reference;
  if (model.info.reference) refImg.src = model.info.reference;
  explodeRow.hidden = typeof runtime?.explode !== 'function';
  explodeInput.value = '0';
  for (const li of list.querySelectorAll('li')) li.classList.toggle('active', li.dataset.slug === model.slug);
  history.replaceState(null, '', `#${model.slug}`);
}

explodeInput.addEventListener('input', () => {
  current?.runtime?.explode?.(Number(explodeInput.value));
});

for (const model of models) {
  const li = document.createElement('li');
  li.dataset.slug = model.slug;
  const button = document.createElement('button');
  button.type = 'button';
  button.innerHTML = `<span class="name"></span><span class="meta"></span>`;
  (button.querySelector('.name') as HTMLElement).textContent = model.title;
  (button.querySelector('.meta') as HTMLElement).textContent =
    `fidelity ${fmt(model.info.fidelityEstimate, 2)} · ${fmt(model.info.apiEquivalentCostUsd, 2, ' USD')}`;
  button.addEventListener('click', () => show(model));
  li.appendChild(button);
  list.appendChild(li);
}
new SimpleBar($('#model-scroll'), { autoHide: true });

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(1, h);
  // Centre the model in the space right of the info panel (wide layouts only).
  const panel = document.querySelector('.overlay') as HTMLElement | null;
  const shift = w > 720 && panel ? (panel.offsetLeft + panel.offsetWidth) / 2 : 0;
  if (shift > 0) camera.setViewOffset(w, h, -shift, 0, w, h);
  else camera.clearViewOffset();
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
applyTheme();
resize();

const initial = models.find((m) => m.slug === location.hash.slice(1)) ?? models[0];
if (initial) show(initial);
else title.textContent = 'No models published yet';

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
