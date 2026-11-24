"""Blender (5.x, background) build step.

Usage: Blender -b --factory-startup --python build_blender.py -- SRC.glb PROC.npz OUT.glb [preview.png]

Imports Bill's GLB, removes the rigid "Human/Head/Hair" volumes, adds one
continuous skinned skater driven by the cleaned skeleton in PROC.npz, re-keys the
skates, and exports a GLB that keeps every node name the website player uses.
"""
import bpy, sys, json, math
import numpy as np
from mathutils import Matrix, Vector, Quaternion

argv = sys.argv[sys.argv.index('--') + 1:]
SRC, PROC, OUT = argv[:3]
PREVIEW = argv[3] if len(argv) > 3 else None

D = np.load(PROC)
J = {k[2:]: D[k] for k in D.files if k.startswith('J_')}
SKN = [str(x) for x in D['skate_names']]; SKM = D['skate_mats']
META = json.loads(str(D['meta']))
F = META['frames']

C3 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # glTF (Y up) -> Blender (Z up)
C4 = np.eye(4); C4[:3, :3] = C3
gb = lambda p: C3 @ p


def nrm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True); return v / np.maximum(n, 1e-9)


# ------------------------------------------------------------------ import
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 60          # must be set before import: glTF seconds -> frames
bpy.context.scene.render.fps_base = 1.0
bpy.ops.import_scene.gltf(filepath=SRC)
sc = bpy.context.scene
sc.frame_start, sc.frame_end = 1, F

HUMAN_PREFIX = ('Human ·', 'Head ·', 'Hair ·')
skate_objs = [bpy.data.objects[n] for n in SKN]
for o in skate_objs:
    mw = o.matrix_world.copy(); o.parent = None; o.matrix_world = mw
    if o.animation_data: o.animation_data_clear()

empty_mesh = bpy.data.meshes.new('removed rigid volume')
def in_human(o):
    while o is not None:
        if o.name.startswith(HUMAN_PREFIX): return True
        o = o.parent
    return False

doomed = {o.name for o in bpy.data.objects if in_human(o) and o.type in ('MESH', 'EMPTY')}
for o in list(bpy.data.objects):
    if o.name not in doomed:
        continue
    keep = [c for c in o.children_recursive if c.name not in doomed]
    if keep:   # e.g. the torso carries an animated follow camera: keep the node, drop the geometry
        o.data = empty_mesh
        o.hide_render = False
    else:
        bpy.data.objects.remove(o, do_unlink=True)
for m in list(bpy.data.meshes):
    if m.users == 0: bpy.data.meshes.remove(m)

# ------------------------------------------------------------------ rest skeleton (glTF coords: up +Y, forward +Z, skater's left +X)
a = math.radians(38)
dn = np.array([math.sin(a), -math.cos(a), 0.0])
REST = {'hipc': [0, 1.00, 0], 'waist': [0, 1.22, 0], 'neck': [0, 1.56, 0], 'head_top': [0, 1.82, 0]}
for s, sg in (('l', 1), ('r', -1)):
    REST[s + '_hip'] = [0.095 * sg, 1.00, 0]
    REST[s + '_kn'] = [0.095 * sg, 0.56, 0.0]
    REST[s + '_an'] = [0.095 * sg, 0.18, 0]
    REST[s + '_sh'] = [0.19 * sg, 1.50, 0]
    d = dn * np.array([sg, 1, 1])
    REST[s + '_el'] = list(np.array(REST[s + '_sh']) + d * 0.30)
    REST[s + '_wr'] = list(np.array(REST[s + '_el']) + d * 0.26)
    REST[s + '_tip'] = list(np.array(REST[s + '_wr']) + d * 0.17)
    REST[s + '_shh'] = list(np.array(REST[s + '_sh']) + d * 0.08)    # tails of the half-rotation helper bones
    REST[s + '_elh'] = list(np.array(REST[s + '_el']) + d * 0.06)
REST = {k: np.array(v, float) for k, v in REST.items()}

BONES = [('hips', 'hipc', 'waist', 'low'), ('chest', 'waist', 'neck', 'up'), ('head', 'neck', 'head_top', 'up')]
for s in 'lr':
    BONES += [(f'thigh.{s}', f'{s}_hip', f'{s}_kn', 'low'), (f'shin.{s}', f'{s}_kn', f'{s}_an', 'low'),
              (f'upperarm.{s}', f'{s}_sh', f'{s}_el', 'up'), (f'forearm.{s}', f'{s}_el', f'{s}_wr', 'up'),
              (f'hand.{s}', f'{s}_wr', f'{s}_tip', 'up'),
              # helper bones that take half of the shoulder / elbow rotation, so the skin bends in two steps and keeps its volume
              (f'shoulder.{s}', f'{s}_sh', f'{s}_shh', 'up'), (f'elbow.{s}', f'{s}_el', f'{s}_elh', 'up')]

# ------------------------------------------------------------------ armature
arm_data = bpy.data.armatures.new('Skater rig v200')
arm = bpy.data.objects.new('Skater rig v200', arm_data)
sc.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='EDIT')
for name, h, t, ref in BONES:
    eb = arm_data.edit_bones.new(name)
    eb.head = Vector(gb(REST[h])); eb.tail = Vector(gb(REST[t]))
    eb.align_roll(Vector(gb(np.array([0, 0, 1.0]))))  # bone Z -> skater forward
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------ body mesh (skin modifier skeleton -> smooth surface)
SKIN = [  # name, position, (side radius, front/back radius)
    ('hipc', REST['hipc'] + [0, -0.02, 0], (0.165, 0.120)),
    ('waist', REST['waist'], (0.140, 0.100)),
    ('chest', np.array([0, 1.38, 0.005]), (0.158, 0.118)),
    ('upper', np.array([0, 1.48, 0.0]), (0.150, 0.104)),
    ('neck', np.array([0, 1.57, 0.0]), (0.056, 0.058)),
    ('neck_top', np.array([0, 1.63, 0.0]), (0.052, 0.054)),
]
EDGES = [('hipc', 'waist'), ('waist', 'chest'), ('chest', 'upper'), ('upper', 'neck'), ('neck', 'neck_top')]
for s in 'lr':
    SKIN += [(s + '_hip', REST[s + '_hip'] + [0, -0.03, 0], (0.098, 0.100)),
             (s + '_thigh_mid', (REST[s + '_hip'] * 0.55 + REST[s + '_kn'] * 0.45), (0.092, 0.098)),
             (s + '_kn', REST[s + '_kn'], (0.065, 0.067)),
             (s + '_calf', REST[s + '_kn'] * 0.6 + REST[s + '_an'] * 0.4, (0.062, 0.071)),
             (s + '_an', REST[s + '_an'] + [0, 0.02, 0], (0.047, 0.049)),
             (s + '_sh', REST[s + '_sh'], (0.068, 0.068)),
             (s + '_uarm_mid', REST[s + '_sh'] * 0.5 + REST[s + '_el'] * 0.5, (0.058, 0.058)),
             (s + '_el', REST[s + '_el'], (0.049, 0.050)),
             (s + '_farm_mid', REST[s + '_el'] * 0.45 + REST[s + '_wr'] * 0.55, (0.048, 0.044)),
             (s + '_wr', REST[s + '_wr'], (0.033, 0.026)),
             (s + '_palm', REST[s + '_wr'] * 0.45 + REST[s + '_tip'] * 0.55, (0.034, 0.018)),
             (s + '_tip', REST[s + '_tip'] - dn * np.array([1 if s == 'l' else -1, 1, 1]) * 0.03, (0.022, 0.012))]
    EDGES += [('hipc', s + '_hip'), (s + '_hip', s + '_thigh_mid'), (s + '_thigh_mid', s + '_kn'), (s + '_kn', s + '_calf'),
              (s + '_calf', s + '_an'), ('upper', s + '_sh'), (s + '_sh', s + '_uarm_mid'), (s + '_uarm_mid', s + '_el'), (s + '_el', s + '_farm_mid'), (s + '_farm_mid', s + '_wr'),
              (s + '_wr', s + '_palm'), (s + '_palm', s + '_tip')]
names = [n for n, _, _ in SKIN]
me = bpy.data.meshes.new('Skater body · smooth skinned v200')
me.from_pydata([tuple(gb(np.asarray(p, float))) for _, p, _ in SKIN], [(names.index(x), names.index(y)) for x, y in EDGES], [])
body = bpy.data.objects.new('Skater body · smooth skinned v200', me)
sc.collection.objects.link(body)
skin = body.modifiers.new('skin', 'SKIN')
skin.use_smooth_shade = True
skin.branch_smoothing = 0.6
for i, (_, _, r) in enumerate(SKIN):
    v = me.skin_vertices[0].data[i]; v.radius = r; v.use_root = (i == 0)
sub = body.modifiers.new('subd', 'SUBSURF'); sub.levels = 2; sub.render_levels = 2
bpy.context.view_layer.objects.active = body
for o in bpy.context.selected_objects: o.select_set(False)
body.select_set(True)
bpy.ops.object.modifier_apply(modifier='skin')
bpy.ops.object.modifier_apply(modifier='subd')
# cut an edge loop at the jacket hem, so the jacket/trousers boundary is a straight line
import bmesh
HEM = 1.08
bm = bmesh.new(); bm.from_mesh(me)
near = lambda el: all(abs(v.co.x) < 0.35 for v in (el.verts if hasattr(el, 'verts') else [el]))
bmesh.ops.bisect_plane(bm, geom=[el for el in list(bm.verts) + list(bm.edges) + list(bm.faces) if near(el)],
                       plane_co=(0, 0, HEM), plane_no=(0, 0, 1), clear_inner=False, clear_outer=False)
bm.to_mesh(me); bm.free()
bpy.ops.object.shade_smooth()


def mat(name, rgb, rough=0.6, sheen=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (*rgb, 1); p.inputs['Roughness'].default_value = rough
    m.diffuse_color = (*rgb, 1)
    return m


def srgb(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c)


M_JACKET = mat('Skater jacket v200', srgb('#1f6f78'), 0.55)
M_PANTS = mat('Skater trousers v200', srgb('#1d2940'), 0.65)
M_SKIN = mat('Skater skin v200', srgb('#e2b79a'), 0.5)
M_HAIR = mat('Skater hair v200', srgb('#2b211c'), 0.8)
for m in (M_JACKET, M_PANTS, M_SKIN): me.materials.append(m)

# ------------------------------------------------------------------ head: smooth ellipsoid + nose + hair cap, parented to the head bone
def ellipsoid(name, center, radii, material, segs=32, rings=20):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings, radius=1.0, location=tuple(gb(center)))
    o = bpy.context.active_object; o.name = name; o.data.name = name + ' mesh'
    o.scale = tuple(abs(C3 @ np.array(radii)))
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.shade_smooth(); o.data.materials.append(material); return o


head_c = np.array([0, 1.715, 0.01])
parts = [ellipsoid('Skater head · smooth v200', head_c, (0.088, 0.112, 0.098), M_SKIN),
         ellipsoid('Skater nose · facing cue v200', head_c + [0, -0.01, 0.094], (0.016, 0.026, 0.022), M_SKIN, 16, 10),
         ellipsoid('Skater ear L v200', head_c + [0.086, 0.0, 0.0], (0.012, 0.026, 0.018), M_SKIN, 12, 8),
         ellipsoid('Skater ear R v200', head_c + [-0.086, 0.0, 0.0], (0.012, 0.026, 0.018), M_SKIN, 12, 8)]
hair = ellipsoid('Skater hair · cap v200', head_c + [0, 0.028, -0.012], (0.094, 0.098, 0.102), M_HAIR)
# trim the hair cap so it only covers top and back of the head
bm_v = hair.data.vertices
hair_del = [v.index for v in bm_v if v.co.z < -0.040 or (v.co.y < -0.030 and v.co.z < 0.045)]
bpy.context.view_layer.objects.active = hair
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='DESELECT'); bpy.ops.object.mode_set(mode='OBJECT')
for i in hair_del: hair.data.vertices[i].select = True
bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.delete(type='VERT'); bpy.ops.object.mode_set(mode='OBJECT')
parts.append(hair)
for o in parts:
    o.select_set(False)
    mw = o.matrix_world.copy()
    o.parent = arm; o.parent_type = 'BONE'; o.parent_bone = 'head'
    o.matrix_world = mw

# ------------------------------------------------------------------ skinning
for o in bpy.context.selected_objects: o.select_set(False)
body.select_set(True); arm.select_set(True); bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
vg = {g.index: g.name for g in body.vertex_groups}

# Arms: analytic weights. Automatic weights let the chest follow the upper arm, which crushes the shoulder when the arm swings.
# The rest arm is a straight line, so a vertex is described by t (metres along the arm from the shoulder) and rho (distance from that line).
smooth01 = lambda x: (lambda c: c * c * (3 - 2 * c))(np.clip(x, 0, 1))
P = np.array([C3.T @ np.array(v.co) for v in me.vertices])
ARM_R = ([-0.05, 0.0, 0.15, 0.30, 0.43, 0.56, 0.75], [0.068, 0.068, 0.058, 0.050, 0.047, 0.034, 0.034])
arm_w = {}
for s, sg in (('l', 1), ('r', -1)):
    d = dn * np.array([sg, 1, 1]); q = P - REST[s + '_sh']
    t = q @ d; rho = np.linalg.norm(q - np.outer(t, d), axis=1)
    m = 1 - smooth01((rho - (np.interp(t, *ARM_R) + 0.006)) / 0.026)       # 1 on the arm, 0 on the torso
    s1 = (t - 0.02) / 0.07; h1 = np.where(np.abs(s1) < 1, 1 - smooth01(np.abs(s1)), 0)   # shoulder band
    s2 = (t - 0.30) / 0.055; h2 = np.where(np.abs(s2) < 1, 1 - smooth01(np.abs(s2)), 0)  # elbow band
    wv = smooth01((t - 0.53) / 0.06)                                                    # wrist
    up = np.where(s1 <= 0, 0, np.where(s1 < 1, 1 - h1, 1)) * np.where(s2 <= -1, 1, np.where(s2 < 0, 1 - h2, 0))
    fo = np.where(s2 <= 0, 0, np.where(s2 < 1, 1 - h2, 1)) * (1 - wv)
    ha = np.where(s2 >= 1, wv, 0)
    arm_w[s] = {f'shoulder.{s}': m * h1, f'upperarm.{s}': m * up, f'elbow.{s}': m * h2, f'forearm.{s}': m * fo, f'hand.{s}': m * ha}
ARM_GROUPS = {n for s in 'lr' for n in arm_w[s]}
groups = {g.name: g for g in body.vertex_groups}
gnames = sorted(groups); gi = {n: k for k, n in enumerate(gnames)}
Wt = np.zeros((len(me.vertices), len(gnames)))
for i, v in enumerate(me.vertices):
    for g in v.groups:
        if vg[g.group] not in ARM_GROUPS: Wt[i, gi[vg[g.group]]] = g.weight
tot = Wt.sum(1); Wt[tot < 1e-4, gi['chest']] = 1.0; Wt /= Wt.sum(1, keepdims=True)
arm_sum = np.zeros(len(me.vertices))
for s in 'lr':
    for n, w in arm_w[s].items(): arm_sum += w
Wt *= np.clip(1 - arm_sum, 0, 1)[:, None]
for s in 'lr':
    for n, w in arm_w[s].items(): Wt[:, gi[n]] += w
# soften the weights round both shoulders, so the transition leaves no ridge in the shading
E = np.array([e.vertices[:] for e in me.edges])
near_sh = np.minimum(np.linalg.norm(P - REST['l_sh'], axis=1), np.linalg.norm(P - REST['r_sh'], axis=1)) < 0.17
deg = np.bincount(E.ravel(), minlength=len(P)).astype(float)
for it in range(4):
    acc = np.zeros_like(Wt); np.add.at(acc, E[:, 0], Wt[E[:, 1]]); np.add.at(acc, E[:, 1], Wt[E[:, 0]])
    Wt[near_sh] = 0.5 * Wt[near_sh] + 0.5 * (acc / np.maximum(deg, 1)[:, None])[near_sh]
Wt[Wt < 2e-3] = 0; Wt /= Wt.sum(1, keepdims=True)
for i, v in enumerate(me.vertices):
    for n in [vg[g.group] for g in v.groups]: groups[n].remove([i])
    for k in np.nonzero(Wt[i])[0]: groups[gnames[k]].add([i], float(Wt[i, k]), 'REPLACE')
for poly in me.polygons:
    best = {}
    for vi in poly.vertices:
        for g in me.vertices[vi].groups:
            best[vg[g.group]] = best.get(vg[g.group], 0) + g.weight
    top = max(best, key=best.get) if best else 'chest'
    c = poly.center
    # everything below the waistline on the torso and legs is trousers: a straight hem, not one that follows the weights
    poly.material_index = 2 if top.startswith('hand') else 1 if (c.z < HEM and abs(c.x) < 0.30) else 0

# ------------------------------------------------------------------ pose keys
ref_dir = {'low': J['f_low'] - J['hipc'], 'up': J['f_up'] - J['neck']}
up_g = nrm(J['neck'] - J['hipc'])
TORSO = {'hips', 'chest', 'head'}
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
rest_inv = {b.name: b.matrix_local.inverted() for b in arm_data.bones}
frames = {}
for name, h, t, ref in BONES:
    if name.startswith(('shoulder', 'elbow')): continue
    y = nrm(J[t] - J[h]); zr = nrm(ref_dir[ref])
    z = np.zeros_like(y)
    for f in range(F):
        cand = zr[f] - np.dot(zr[f], y[f]) * y[f]
        if np.linalg.norm(cand) < 0.3:
            cand = up_g[f] - np.dot(up_g[f], y[f]) * y[f]
        cand /= np.linalg.norm(cand)
        if name in TORSO or f == 0:
            z[f] = cand
        else:  # parallel transport with a gentle pull to the reference: no twist flips
            prev = z[f - 1] - np.dot(z[f - 1], y[f]) * y[f]
            prev /= max(np.linalg.norm(prev), 1e-9)
            zz = prev * 0.9 + cand * 0.1
            z[f] = zz / np.linalg.norm(zz)
    x = np.cross(y, z)
    frames[name] = (x, y, z, J[h])


def swing(a, b, v, frac=1.0):
    """Rotate v by frac of the shortest rotation that takes unit a to unit b (per frame)."""
    ax = np.cross(a, b); sn = np.linalg.norm(ax, axis=1); ang = np.arctan2(sn, (a * b).sum(1)) * frac
    k = ax / np.maximum(sn, 1e-9)[:, None]; co, si = np.cos(ang)[:, None], np.sin(ang)[:, None]
    out = v * co + np.cross(k, v) * si + k * (k * v).sum(1, keepdims=True) * (1 - co)
    return np.where(sn[:, None] < 1e-6, v, out)


# Arms carry no twist of their own: the upper arm swings from the chest and the forearm from the upper arm,
# and each helper bone sits halfway through that swing.
xc, yc, zc, _ = frames['chest']
for s, sg in (('l', 1), ('r', -1)):
    d0 = nrm(dn[0] * sg * xc + dn[1] * yc)                 # rest arm direction carried by the chest
    ya = nrm(J[s + '_el'] - J[s + '_sh']); yf = nrm(J[s + '_wr'] - J[s + '_el'])
    za = swing(d0, ya, zc); zf = swing(ya, yf, za)
    for name, y, z, p in ((f'upperarm.{s}', ya, za, J[s + '_sh']), (f'forearm.{s}', yf, zf, J[s + '_el']),
                          (f'shoulder.{s}', swing(d0, ya, d0, 0.5), swing(d0, ya, zc, 0.5), J[s + '_sh']),
                          (f'elbow.{s}', swing(ya, yf, ya, 0.5), swing(ya, yf, za, 0.5), J[s + '_el'])):
        y = nrm(y); z = nrm(z - (z * y).sum(1, keepdims=True) * y)
        frames[name] = (np.cross(y, z), y, z, p)

# the rig rides on the animated skater root, so three.js culling bounds travel with the skater
root_obj = bpy.data.objects['Skater root · simulated travel and 1.5-turn yaw v8']
arm.parent = root_obj; arm.matrix_parent_inverse = Matrix.Identity(4); arm.location = (0, 0, 0)
arm.rotation_euler = (0, 0, 0); arm.scale = (1, 1, 1)
ROOTB = np.einsum('ij,fjk,kl->fil', C4, D['root'], C4.T)
ROOTB_INV = np.linalg.inv(ROOTB)
for pb in arm.pose.bones: pb.rotation_mode = 'QUATERNION'
prevq = {}
for f in range(F):
    for name, *_ in BONES:
        x, y, z, p = frames[name]
        M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = C3 @ x[f], C3 @ y[f], C3 @ z[f], C3 @ p[f]
        basis = rest_inv[name] @ Matrix((ROOTB_INV[f] @ M).tolist())
        loc, q, _ = basis.decompose()
        if name in prevq and prevq[name].dot(q) < 0: q.negate()
        prevq[name] = q
        pb = arm.pose.bones[name]
        pb.location = loc; pb.rotation_quaternion = q; pb.scale = (1, 1, 1)
        for path in ('location', 'rotation_quaternion', 'scale'):
            pb.keyframe_insert(path, frame=f + 1)
bpy.ops.object.mode_set(mode='OBJECT')

# ------------------------------------------------------------------ skates (each part keyed in world space)
for n, o in zip(SKN, skate_objs):
    o.rotation_mode = 'QUATERNION'
    k = SKN.index(n)
    prevq = None
    for f in range(F):
        o.matrix_world = Matrix((C4 @ SKM[k][f] @ C4.T).tolist())
        if prevq is not None and prevq.dot(o.rotation_quaternion) < 0:
            o.rotation_quaternion = -o.rotation_quaternion
        prevq = o.rotation_quaternion.copy()
        for path in ('location', 'rotation_quaternion', 'scale'):
            o.keyframe_insert(path, frame=f + 1)

# ------------------------------------------------------------------ preview contact sheet (not exported)
if PREVIEW:
    # imported per-node actions are stashed; activate the root's action so the preview matches the web player
    for act in bpy.data.actions:
        if act.name.startswith('Skater root'):
            ad = root_obj.animation_data or root_obj.animation_data_create()
            ad.action = act
            try:
                if not ad.action_slot and len(act.slots): ad.action_slot = act.slots[0]
            except AttributeError:
                pass
            break
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'; sc.display.shading.color_type = 'MATERIAL'
    sc.display.shading.show_shadows = True
    sc.render.resolution_x, sc.render.resolution_y = 360, 540
    cam_d = bpy.data.cameras.new('preview cam'); cam_d.lens = 50
    cam = bpy.data.objects.new('preview cam', cam_d); sc.collection.objects.link(cam); sc.camera = cam
    hip = J['hipc']; travel = hip[-1] - hip[0]; travel[1] = 0
    side = np.cross(travel / max(np.linalg.norm(travel), 1e-6), [0, 1, 0])
    if np.linalg.norm(side) < 0.5: side = np.array([0, 0, 1.0])
    fl = META.get('flights') or []
    if len(argv) > 4:
        marks = [int(x) for x in argv[4].split(',')]
    elif fl:
        t0, t1 = fl[0]['takeoff'], fl[0]['landing']; n = t1 - t0
        marks = [t0 - 40, t0 - 8, t0, t0 + n // 2, t1, t1 + 5, t1 + 16, t1 + 60]
    else:
        kf = META.get('key_frames') or [int(F * k) for k in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85)]
        marks = kf[:: max(1, -(-len(kf) // 9))]
    tiles = []
    for fr in marks:
        fr = int(min(max(fr, 1), F))
        sc.frame_set(fr)
        tgt = J['hipc'][fr - 1] * np.array([1, 0, 1]) + np.array([0, 1.0, 0])
        pos = tgt + side * 4.2 + np.array([0, 0.15, 0])
        cam.location = Vector(gb(pos))
        dvec = Vector(gb(tgt - pos))
        cam.rotation_mode = 'QUATERNION'; cam.rotation_quaternion = dvec.to_track_quat('-Z', 'Y')
        path = f'{PREVIEW}-f{fr:03d}.png'; sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(path)
        tiles.append(np.array(img.pixels[:]).reshape(img.size[1], img.size[0], 4)); bpy.data.images.remove(img)
    sheet = np.concatenate(tiles, axis=1)
    out = bpy.data.images.new('sheet', sheet.shape[1], sheet.shape[0], alpha=True)
    out.pixels = sheet.ravel().tolist(); out.filepath_raw = PREVIEW + '-sheet.png'; out.file_format = 'PNG'; out.save()
    bpy.data.objects.remove(cam, do_unlink=True)

bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_animations=True, export_force_sampling=True,
                          export_frame_range=True, export_cameras=True, export_lights=False, export_apply=False,
                          export_skins=True, export_morph=False,
                          export_optimize_animation_size=False)
print('BUILD_OK', OUT)
