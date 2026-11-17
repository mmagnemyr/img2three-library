// Interaction layer for the generated factory (attached by tools/patch_factory.py).
// One definition of "a part" shared by explode and picking (grimoire/build/geometry_patterns.md, assembly contract):
//   a part = a component pivot node whose mesh is NOT flagged userData.explodeWithParent.
// Surface detail (the printed decal shell, the base chip) rides its parent and a click on it resolves to the parent.
import * as THREE from 'three';

type Runtime = {
  nodes: Record<string, THREE.Object3D>;
  meshes: Record<string, THREE.Mesh>;
};

export type MugInteraction = {
  parts: string[];
  explode: (amount: number) => void;
  pickPart: (raycaster: THREE.Raycaster) => { part: string; hitMesh: string; point: [number, number, number] } | null;
};

export function attachInteraction(root: THREE.Group): MugInteraction {
  const runtime = root.userData.sculptRuntime as Runtime;
  const isDetail = (id: string) => runtime.meshes[id]?.userData.explodeWithParent === true;
  const parentOf = (id: string): string | null => {
    const comp = runtime.meshes[id]?.userData.sculptComponent as { parent?: string | null } | undefined;
    return comp?.parent ?? null;
  };
  const resolve = (id: string): string => {
    let cur: string | null = id;
    while (cur && isDetail(cur)) cur = parentOf(cur);
    return cur ?? id;
  };
  const partIds = Object.keys(runtime.nodes).filter((id) => id !== 'root' && runtime.meshes[id] && !isDetail(id));
  const home = new Map<string, THREE.Vector3>();
  for (const id of partIds) home.set(id, runtime.nodes[id].position.clone());

  root.updateMatrixWorld(true);
  const modelBox = new THREE.Box3().setFromObject(root);
  const centre = modelBox.getCenter(new THREE.Vector3());
  const partCentre = new Map<string, THREE.Vector3>();
  for (const id of partIds) partCentre.set(id, new THREE.Box3().setFromObject(runtime.meshes[id]).getCenter(new THREE.Vector3()));

  const explode = (amount: number) => {
    // Separation is a layout SCALE about the model centre (~2x at amount 1) plus a base clearance, never a uniform push.
    for (const id of partIds) runtime.nodes[id].position.copy(home.get(id)!);
    root.updateMatrixWorld(true);
    const offsets = new Map<string, THREE.Vector3>();
    for (const id of partIds) {
      const c = partCentre.get(id)!;
      const d = c.clone().sub(centre);
      const dir = d.lengthSq() > 1e-6 ? d.clone().normalize() : new THREE.Vector3(0, 1, 0);
      offsets.set(id, d.multiplyScalar(amount).add(dir.multiplyScalar(0.12 * amount)));
    }
    for (const id of partIds) {
      // nested parts already move with their parent: subtract the parent's world offset
      let worldOffset = offsets.get(id)!.clone();
      const p = parentOf(id);
      if (p && offsets.has(p)) worldOffset = worldOffset.sub(offsets.get(p)!);
      const node = runtime.nodes[id];
      const parent = node.parent ?? root;
      const a = parent.worldToLocal(node.getWorldPosition(new THREE.Vector3()).add(worldOffset));
      node.position.copy(a);
    }
    root.updateMatrixWorld(true);
  };

  const pickPart = (raycaster: THREE.Raycaster) => {
    const hits = raycaster.intersectObject(root, true);
    for (const h of hits) {
      const comp = h.object.userData.sculptComponent as { id?: string } | undefined;
      if (!comp?.id || !(h.object as THREE.Mesh).isMesh) continue;
      return { part: resolve(comp.id), hitMesh: comp.id, point: h.point.toArray() as [number, number, number] };
    }
    return null;
  };

  const api: MugInteraction = { parts: partIds, explode, pickPart };
  root.userData.interaction = api;
  return api;
}
