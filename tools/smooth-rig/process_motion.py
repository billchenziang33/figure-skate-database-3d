"""Rebuild a clean, fixed-proportion skeleton from a rigid-segment skater GLB.

Input: one of Bill's r3 GLBs (rigid body volumes posed per frame).
Output (.npz): world-space joint trajectories (glTF Y-up coords) for a new
adult-proportion skeleton, new skate world matrices, and metadata.

Pipeline
1. Read world matrices of every joint volume and of the skater root.
2. Work in root-local space (removes travel and spin yaw), smooth joints.
3. Keep only segment *directions* from the source; rebuild positions with
   fixed, left/right-symmetric adult bone lengths (no limb stretching).
4. In jump flight windows, blend arms into a tight tuck and cross the free
   leg over the landing leg (the "air position" that makes a jump readable).
5. Re-attach skates (scaled 0.85) to the new ankles and re-ground: one blade
   touches the ice whenever the skater is not in flight.
"""
import json, struct, sys
import numpy as np

ICE_Y = 0.012
SKATE_SCALE = 0.85
L = dict(hip_w=0.19, thigh=0.44, shin=0.43, spine=0.50, waist=0.22, clav_up=0.06,
         sh_w=0.38, uarm=0.30, farm=0.26, hand=0.17, head=0.26)
ROOT = 'Skater root · simulated travel and 1.5-turn yaw v8'
SRC = {'pelvis': 'Human · pelvic/waist volume', 'neck': 'Human · neck',
       'head': 'Head · detailed generic human sculpt',
       'l_sh': 'Human · left shoulder volume', 'l_el': 'Human · left elbow joint', 'l_wr': 'Human · left palm',
       'r_sh': 'Human · right shoulder volume', 'r_el': 'Human · right elbow joint', 'r_wr': 'Human · right palm',
       'l_hip': 'Human · left leg volume 3', 'l_kn': 'Human · left knee joint', 'l_an': 'Human · left ankle joint',
       'r_hip': 'Human · right leg volume 3', 'r_kn': 'Human · right knee joint', 'r_an': 'Human · right ankle joint'}
FIT = {'l': 'Left skate · exact website fit v64', 'r': 'Right skate · exact website fit v64'}
BLADE = {'l': 'Left skate · website blade v64', 'r': 'Right skate · website blade v64'}


# ---------------------------------------------------------------- glTF access
def load(p):
    b = open(p, 'rb').read()
    jl = struct.unpack('<I', b[12:16])[0]
    j = json.loads(b[20:20 + jl]); off = 20 + jl
    bl = struct.unpack('<I', b[off:off + 4])[0]
    return j, b[off + 8:off + 8 + bl]


def acc(j, bin_, i):
    a = j['accessors'][i]; bv = j['bufferViews'][a['bufferView']]
    n = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
    dt = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}[a['componentType']]
    arr = np.frombuffer(bin_, dtype=dt, count=a['count'] * n,
                        offset=bv.get('byteOffset', 0) + a.get('byteOffset', 0)).astype(np.float64)
    return arr.reshape(a['count'], n) if n > 1 else arr


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Scene:
    def __init__(self, path):
        self.j, self.b = load(path)
        N = self.j['nodes']; self.N = N
        self.par = {c: i for i, n in enumerate(N) for c in n.get('children', [])}
        self.names = {n.get('name', ''): i for i, n in enumerate(N)}
        self.anim = {}; F = 0
        for an in self.j['animations']:
            for ch in an['channels']:
                s = an['samplers'][ch['sampler']]
                t = acc(self.j, self.b, s['input'])
                self.anim.setdefault(ch['target']['node'], {})[ch['target']['path']] = acc(self.j, self.b, s['output'])
                F = max(F, len(t))
        self.F = F; self.cache = {}

    def local(self, i, f):
        n = self.N[i]; a = self.anim.get(i, {})
        g = lambda k, d: a[k][min(f, len(a[k]) - 1)] if k in a else np.array(n.get(k, d), float)
        M = np.eye(4); M[:3, :3] = qmat(g('rotation', [0, 0, 0, 1])) * g('scale', [1, 1, 1])[None, :]
        M[:3, 3] = g('translation', [0, 0, 0]); return M

    def world(self, i, f):
        k = (i, f)
        if k not in self.cache:
            self.cache[k] = (self.world(self.par[i], f) if i in self.par else np.eye(4)) @ self.local(i, f)
        return self.cache[k]

    def track(self, name):
        i = self.names[name]
        return np.array([self.world(i, f) for f in range(self.F)])

    def mesh_points(self, name, relative_to):
        """Vertices of node `name` expressed in the frame of node `relative_to` (frame 0)."""
        i = self.names[name]; m = self.j['meshes'][self.N[i]['mesh']]
        P = np.vstack([acc(self.j, self.b, pr['attributes']['POSITION']) for pr in m['primitives']])
        M = np.linalg.inv(self.world(self.names[relative_to], 0)) @ self.world(i, 0)
        return (M[:3, :3] @ P.T).T + M[:3, 3]


# ---------------------------------------------------------------- helpers
def nrm(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def ortho(v, axis):
    return v - (v * axis).sum(-1, keepdims=True) * axis


def savgol(x, win=9, poly=3):
    """Zero-phase Savitzky-Golay smoothing along axis 0 (edges reflected)."""
    h = win // 2
    A = np.vander(np.arange(-h, h + 1), poly + 1, increasing=True)
    c = np.linalg.pinv(A)[0]
    xp = np.concatenate([x[h:0:-1], x, x[-2:-h - 2:-1]], axis=0)
    out = np.zeros_like(x)
    for k in range(win):
        out += c[k] * xp[k:k + len(x)]
    return out


def smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def rot_between(a, b):
    """Rotation matrices taking unit vectors a -> b (batched)."""
    v = np.cross(a, b); c = (a * b).sum(-1); s = np.linalg.norm(v, axis=-1)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    f = np.where(s > 1e-9, (1 - c) / np.maximum(s * s, 1e-18), 0.5)[..., None, None]
    return np.eye(3) + K + K @ K * f


def flight_windows(events):
    """[(takeoff, landing, label)] in 0-based frame indices from event dict (1-based frames)."""
    if not events: return []
    out = []
    for k, v in events.items():
        if k.upper().startswith('TAKEOFF'):
            suf = k[len('TAKEOFF'):]
            for k2, v2 in events.items():
                if k2.upper() == ('LANDING' + suf).upper():
                    out.append((int(v) - 1, int(v2) - 1, suf.strip('_') or 'jump'))
    return sorted(out)


# ---------------------------------------------------------------- main
def marker_events(S):
    import re
    ev = {}
    for name in S.names:
        m = re.match(r'V102_EVENT_([A-Z]+)(?:_([A-Za-z0-9]+))?_FRAME_(\d+)$', name)
        if m: ev[m.group(1) + ('_' + m.group(2) if m.group(2) else '')] = int(m.group(3))
    return ev


def process(path, events, kind):
    S = Scene(path); F = S.F
    events = marker_events(S) or events
    R = S.track(ROOT); Rinv = np.linalg.inv(R)
    to_local = lambda Pw: np.einsum('fij,fj->fi', Rinv, np.c_[Pw, np.ones(F)])[:, :3]
    src = {k: to_local(S.track(v)[:, :3, 3]) for k, v in SRC.items()}
    src = {k: savgol(v) for k, v in src.items()}
    side_name = {'l': 'Left', 'r': 'Right'}
    parts = {s: sorted(n for n, i in S.names.items() if n.startswith(side_name[s] + ' skate · website ')
                       and 'mesh' in S.N[i]) for s in 'lr'}
    part_w = {s: {n: S.track(n) for n in parts[s]} for s in 'lr'}
    blade_w = {s: part_w[s][BLADE[s]] for s in 'lr'}
    an_w = {s: S.track(SRC[s + '_an'])[:, :3, 3] for s in 'lr'}

    hipc_s = (src['l_hip'] + src['r_hip']) / 2
    u = nrm(src['neck'] - hipc_s)
    # flight weights first: the air position also straightens the trunk
    windows = flight_windows(events) if kind in ('jump', 'combination') else []
    hip_line = S.track(SRC['r_hip'])[:, :3, 3] - S.track(SRC['l_hip'])[:, :3, 3]   # world-space body yaw (works for both rigs)
    root_yaw = np.unwrap(np.arctan2(hip_line[:, 0], hip_line[:, 2]))
    fr = np.arange(F); wins = []
    for t0, t1, label in windows:
        n = t1 - t0
        turns = abs(root_yaw[t1] - root_yaw[t0]) / (2 * np.pi)
        strength = float(np.clip((turns - 0.6) / 2.2, 0.3, 1.0))
        ramp = max(3, int(round(n * 0.22)))
        w = smoothstep((fr - t0) / ramp) * smoothstep((t1 - fr) / ramp)
        w[(fr < t0) | (fr > t1)] = 0
        wins.append((t0, t1, label, turns, strength, w * strength))
    w_trunk = np.zeros(F)
    for *_, w in wins: w_trunk = np.maximum(w_trunk, w)
    u = nrm(u * (1 - 0.85 * w_trunk[:, None]) + np.array([0, 1.0, 0]) * 0.85 * w_trunk[:, None])
    r_low = nrm(ortho(src['r_hip'] - src['l_hip'], u))
    sh = ortho(src['r_sh'] - src['l_sh'], u)
    good = np.linalg.norm(sh, axis=-1, keepdims=True) > 0.18
    r_up = nrm(np.where(good, nrm(sh), r_low))
    r_up = nrm(savgol(r_up, 11, 2))
    f_low = nrm(np.cross(u, r_low)); f_up = nrm(np.cross(u, r_up))
    # knees bend forward: fix the sign of "forward" from the data
    knee_off = src['l_kn'] - (src['l_hip'] + src['l_an']) / 2
    if (knee_off * f_low).sum(-1).mean() < 0:
        f_low, f_up = -f_low, -f_up

    d = lambda a, b: nrm(src[b] - src[a])
    dirs = {'l_thigh': d('l_hip', 'l_kn'), 'l_shin': d('l_kn', 'l_an'),
            'r_thigh': d('r_hip', 'r_kn'), 'r_shin': d('r_kn', 'r_an'),
            'l_uarm': d('l_sh', 'l_el'), 'l_farm': d('l_el', 'l_wr'),
            'r_uarm': d('r_sh', 'r_el'), 'r_farm': d('r_el', 'r_wr'),
            'head': d('neck', 'head')}
    src_shin = {s: dirs[s + '_shin'].copy() for s in 'lr'}

    # ---- air position in flight windows (jumps only)
    w_all = np.zeros(F); info = []
    wleg = {'l': np.zeros(F), 'r': np.zeros(F)}
    dirs['head'] = nrm(dirs['head'] * (1 - w_trunk[:, None]) + u * w_trunk[:, None])
    for t0, t1, label, turns, strength, w in wins:
        # landing foot = the lower ankle shortly after landing
        k = min(F - 1, t1 + 6)
        land = 'r' if an_w['r'][k, 1] <= an_w['l'][k, 1] else 'l'
        free = 'l' if land == 'r' else 'r'
        side = {'r': 1.0, 'l': -1.0}
        W = w[:, None]
        for s in 'lr':  # arms: elbows in, forearms crossing in front of the chest
            ua_t = nrm(-u * 0.80 + f_up * 0.50 - r_up * side[s] * 0.18)
            fa_t = nrm(f_up * 0.35 - r_up * side[s] * 0.92 + u * (0.10 if s == 'r' else -0.05))
            dirs[s + '_uarm'] = nrm(dirs[s + '_uarm'] * (1 - W) + ua_t * W)
            dirs[s + '_farm'] = nrm(dirs[s + '_farm'] * (1 - W) + fa_t * W)
        # landing leg long and straight; free leg crossed in front of it
        dirs[land + '_thigh'] = nrm(dirs[land + '_thigh'] * (1 - W) + nrm(-u + f_low * 0.05) * W)
        dirs[land + '_shin'] = nrm(dirs[land + '_shin'] * (1 - W) + nrm(-u - f_low * 0.02) * W)
        dirs[free + '_thigh'] = nrm(dirs[free + '_thigh'] * (1 - W) + nrm(-u + f_low * 0.18 + r_low * side[land] * 0.20) * W)
        dirs[free + '_shin'] = nrm(dirs[free + '_shin'] * (1 - W) + nrm(-u * 0.85 - f_low * 0.10 + r_low * side[land] * 0.30) * W)
        # landing position ("checkout"): free leg extended back, arms open
        later = [a for a, *_ in wins if a > t1]
        end = min([F - 1, t1 + 52] + [a - 6 for a in later])
        span = end - t1
        if span >= 12:
            hold = t1 + int(span * 0.55)
            wl = smoothstep((fr - t1) / 7.0) * smoothstep((end - fr) / max(4.0, end - hold))
            wl[(fr < t1) | (fr > end)] = 0
            WL = (wl * max(strength, 0.6))[:, None]
            for s in 'lr':
                ua_t = nrm(r_up * side[s] * 0.85 + f_up * 0.35 - u * 0.30)
                fa_t = nrm(r_up * side[s] * 0.85 + f_up * 0.40 - u * 0.12)
                dirs[s + '_uarm'] = nrm(dirs[s + '_uarm'] * (1 - WL) + ua_t * WL)
                dirs[s + '_farm'] = nrm(dirs[s + '_farm'] * (1 - WL) + fa_t * WL)
            dirs[land + '_thigh'] = nrm(dirs[land + '_thigh'] * (1 - WL) + nrm(-u + f_low * 0.28) * WL)
            dirs[land + '_shin'] = nrm(dirs[land + '_shin'] * (1 - WL) + nrm(-u - f_low * 0.18) * WL)
            dirs[free + '_thigh'] = nrm(dirs[free + '_thigh'] * (1 - WL) + nrm(-u * 0.62 - f_low * 0.70 - r_low * side[land] * 0.22) * WL)
            dirs[free + '_shin'] = nrm(dirs[free + '_shin'] * (1 - WL) + nrm(-u * 0.50 - f_low * 0.84 - r_low * side[land] * 0.18) * WL)
        wa = smoothstep((fr - (t0 - 3)) / 6.0) * smoothstep(((t1 + 3) - fr) / 6.0)
        wa[(fr < t0 - 3) | (fr > t1 + 3)] = 0
        for s_ in 'lr': wleg[s_] = np.maximum(wleg[s_], wa)
        if span >= 12: wleg[free] = np.maximum(wleg[free], WL[:, 0] / max(strength, 0.6))
        w_all = np.maximum(w_all, w)
        info.append(dict(takeoff=t0 + 1, landing=t1 + 1, label=label, turns=round(turns, 2),
                         strength=round(strength, 2), landing_foot=land))

    # ---- rebuild with fixed lengths (root-local)
    J = {}
    hipc = src['pelvis'] + (hipc_s - src['pelvis'])
    J['hipc'] = hipc
    J['waist'] = hipc + u * L['waist']
    J['neck'] = hipc + u * (L['spine'] + L['clav_up'])
    shc = hipc + u * L['spine']
    J['head_top'] = J['neck'] + dirs['head'] * L['head']
    for s, sg in (('l', -1), ('r', 1)):
        J[s + '_hip'] = hipc + r_low * sg * L['hip_w'] / 2
        J[s + '_kn'] = J[s + '_hip'] + dirs[s + '_thigh'] * L['thigh']
        J[s + '_an'] = J[s + '_kn'] + dirs[s + '_shin'] * L['shin']
        J[s + '_sh'] = shc + r_up * sg * L['sh_w'] / 2
        J[s + '_el'] = J[s + '_sh'] + dirs[s + '_uarm'] * L['uarm']
        J[s + '_wr'] = J[s + '_el'] + dirs[s + '_farm'] * L['farm']
        J[s + '_tip'] = J[s + '_wr'] + dirs[s + '_farm'] * L['hand']
    J['f_low'] = J['hipc'] + f_low; J['f_up'] = J['neck'] + f_up  # reference points, not joints

    to_world = lambda P: np.einsum('fij,fj->fi', R, np.c_[P, np.ones(F)])[:, :3]
    Jw = {k: to_world(v) for k, v in J.items()}
    blade_pts = {s: S.mesh_points(BLADE[s], BLADE[s])[::7] for s in 'lr'}
    up = np.array([0, 1.0, 0])

    def blade_low(M, s_):
        P = np.einsum('fij,nj->fni', M[:, :3, :3], blade_pts[s_]) + M[:, None, :3, 3]
        return P[..., 1].min(1)

    contact = np.ones(F, bool)
    for t0, t1, _ in flight_windows(events):
        contact[t0 + 1:t1] = False
    idx = np.arange(F)
    Jw0 = {k: v.copy() for k, v in Jw.items()}

    # ---- contact solution: keep the audited source ankle heights (blade contact), solve knees by IK
    src_w = {k: S.track(SRC[k])[:, :3, 3] for k in ('l_hip', 'r_hip', 'l_kn', 'r_kn')}
    tgt, rho, low_src = {}, {}, {}
    for s in 'lr':
        low_src[s] = blade_low(blade_w[s], s)
        a = an_w[s].copy()
        a[:, 1] = a[:, 1] + (1 - SKATE_SCALE) * (low_src[s] - a[:, 1])  # smaller skate, same blade height
        tgt[s] = a
        l_src = np.linalg.norm(src_w[s + '_kn'] - src_w[s + '_hip'], axis=1) + np.linalg.norm(an_w[s] - src_w[s + '_kn'], axis=1)
        rho[s] = np.clip(np.linalg.norm(an_w[s] - src_w[s + '_hip'], axis=1) / l_src, 0.55, 0.985)
    Lleg = L['thigh'] + L['shin']
    stance = np.where(low_src['l'] <= low_src['r'], 'l', 'r')
    c = np.zeros(F)
    for f in range(F):
        s = stance[f]
        d = Jw0[s + '_hip'][f] - tgt[s][f]
        dxz2 = d[0] ** 2 + d[2] ** 2
        c[f] = -d[1] + np.sqrt(max((rho[s][f] * Lleg) ** 2 - dxz2, 1e-6))
    c = np.where(contact, c, np.nan)
    c = np.interp(idx, idx[contact], c[contact])
    c = savgol(savgol(c, 21, 2), 9, 2)

    def ik(H, A, pole_pt):
        l1, l2 = L['thigh'], L['shin']
        dv = A - H; dist = np.clip(np.linalg.norm(dv, axis=1), 0.2, l1 + l2 - 1e-3)
        dn_ = nrm(dv); pole = nrm(ortho(pole_pt - H, dn_))
        a1 = (l1 * l1 + dist * dist - l2 * l2) / (2 * dist)
        h = np.sqrt(np.maximum(l1 * l1 - a1 * a1, 0))
        return H + dn_ * a1[:, None] + pole * h[:, None], H + dn_ * dist[:, None]

    def compose(Rm, t, M):  # (F,3,3),(F,3) applied to (F,4,4)
        out = M.copy(); out[:, :3, :3] = Rm @ M[:, :3, :3]; out[:, :3, 3] = np.einsum('fij,fj->fi', Rm, M[:, :3, 3]) + t
        return out

    def solve(c):
        J_ = {k: v + up * c[:, None] for k, v in Jw0.items()}
        T = {}
        for s in 'lr':
            H = J_[s + '_hip']
            Kc, Ac = ik(H, tgt[s], J_[s + '_kn'])
            w = wleg[s][:, None]
            K = Kc * (1 - w) + J_[s + '_kn'] * w
            A = Ac * (1 - w) + J_[s + '_an'] * w
            K = H + nrm(K - H) * L['thigh']; A = K + nrm(A - K) * L['shin']
            J_[s + '_kn'], J_[s + '_an'] = K, A
            Q = rot_between(nrm(an_w[s] - src_w[s + '_kn']), nrm(A - K))
            Rm = Q * SKATE_SCALE                                # foot transform: source skate -> new skate
            T[s] = (Rm, A - np.einsum('fij,fj->fi', Rm, an_w[s]))
        low_side = lambda s_: blade_low(compose(*T[s_], blade_w[s_]), s_)
        # never through the ice: lift a dipping blade by bending that knee
        for s in [*'lr'] * 5:
            sink = np.maximum(ICE_Y - low_side(s), 0)
            if sink.max() < 5e-4: continue
            sink = np.maximum(savgol(np.maximum.reduce([np.roll(sink, k) for k in range(-3, 4)]), 7, 2), 0) + 0.001 * (sink > 0)
            Kp, Ap = J_[s + '_kn'], J_[s + '_an'].copy(); Ap[:, 1] += sink
            Kn, An = ik(J_[s + '_hip'], Ap, Kp)
            Q = rot_between(nrm(J_[s + '_an'] - Kp), nrm(An - Kn))
            Rm, t = T[s]
            T[s] = (Q @ Rm, An + np.einsum('fij,fj->fi', Q, t - J_[s + '_an']))
            J_[s + '_kn'], J_[s + '_an'] = Kn, An
        low = np.minimum(low_side('l'), low_side('r'))
        return J_, T, low

    Jw, T, low2 = solve(c)
    # second pass: if both blades float on a ground frame (source hop artefacts in spins), lower the body
    flo = np.where(contact, np.maximum(low2 - ICE_Y - 0.004, 0), 0)
    if flo.max() > 0.006:
        flo = np.maximum.reduce([np.roll(flo, k) for k in range(-4, 5)])
        drop = np.maximum(savgol(flo, 11, 2), 0)
        c = c - drop
        for s in 'lr': tgt[s][:, 1] -= drop
        Jw, T, low2 = solve(c)
    skate_parts = {n: compose(*T[s], part_w[s][n]) for s in 'lr' for n in parts[s]}
    off = c

    steady = contact & ~(np.convolve(~contact, np.ones(11), 'same') > 0)   # ground frames away from take-off/landing
    meta = dict(frames=F, flights=info, ground_offset=[round(float(off.min()), 3), round(float(off.max()), 3)],
                min_blade_contact=round(float(low2[contact].min() - ICE_Y), 4),
                max_blade_contact=round(float(low2[steady].max() - ICE_Y), 4),
                float_frames_over_1cm=int(((low2 - ICE_Y) > 0.01)[steady].sum()),
                min_blade_flight=round(float(low2[~contact].min() - ICE_Y), 4) if (~contact).any() else None)
    return Jw, skate_parts, meta, w_all, R


if __name__ == '__main__':
    path, out, events_json, kind = sys.argv[1:5]
    Jw, parts, meta, w, R = process(path, json.loads(events_json) if events_json != 'null' else None, kind)
    names = sorted(parts)
    np.savez(out, **{'J_' + k: v for k, v in Jw.items()}, skate_names=np.array(names),
             skate_mats=np.stack([parts[n] for n in names]), tuck=w, root=R, meta=json.dumps(meta))
    print(json.dumps(meta))
