"""Author a skating motion from a pose library (round 5).

Round 4 kept the limb directions of the source animation, so its awkward poses
survived. This step keeps only what the source gets right and the player needs:
  * the travel path and jump height (pelvis trajectory),
  * the body yaw (so turn counts stay exactly as audited),
  * the event frames (take-off / landing / spin phases),
  * the skate meshes.
Every body pose is authored here from figure-skating technique: entry edge and
toe-pick per jump type, tight air position, one-foot landing with the free leg
extended, camel / sit / upright spin positions, turns and a spiral in sequences.

Usage: python3 author_motion.py SRC.glb OUT.npz EVENTS_JSON KIND ID
Output: same .npz layout as process_motion.py (world joints, skate matrices).
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_motion import Scene, SRC, ROOT, BLADE, savgol, nrm, ortho, smoothstep, marker_events, flight_windows

ICE_Y = 0.012
SKATE_SCALE = 0.85
L = dict(hip_w=0.19, thigh=0.44, shin=0.38, spine=0.50, waist=0.22, clav_up=0.06,
         sh_w=0.38, uarm=0.30, farm=0.26, hand=0.17, head=0.26)
HW = L['hip_w'] / 2
LEG = L['thigh'] + L['shin']
UP = np.array([0, 1.0, 0])
VEC = ('lean', 'head', 'l_toe', 'r_toe', 'l_knee', 'r_knee', 'l_ua', 'l_fa', 'r_ua', 'r_fa')


# ---------------------------------------------------------------- pose library
# Body frame: (right, up, forward). The skater turns counter-clockwise (to the left)
# and lands jumps on the right foot, like the source animation.
def ice(r, f, toe=(0, 0, 1), pitch=0.0, knee=None):
    """Blade on the ice at (right, forward) from the pelvis; pitch > 0 = on the toe pick."""
    return ('ice', r, f, toe, pitch, knee)


def free(dr, dy, df, toe=(0, -0.25, 1), knee=None):
    """Free foot: ankle offset from its own hip joint."""
    return ('free', dr, dy, df, toe, knee)


def mir(v):
    return (-v[0], v[1], v[2])


def Rm(arm):
    return (mir(arm[0]), mir(arm[1]))


def pose(ext, Lg, Rg, la, ra, lean=(0, 1, 0.07), twist=0.0, head=None, spinw=1.0):
    """ext = height of the hips above the stance ankle (0.80 = straight leg)."""
    p = {'ext': ext, 'lean': lean, 'twist': twist, 'spinw': spinw}
    p['head'] = head or tuple(np.array(lean, float) * 0.5 + np.array([0, 0.6, 0.08]))
    for s, sg, g in (('l', -1, Lg), ('r', 1, Rg)):
        if g[0] == 'ice':
            _, r, f, toe, pitch, knee = g
            p[s + '_an'] = (r, -ext, f); p[s + '_g'] = 1.0; p[s + '_pitch'] = pitch
        else:
            _, dr, dy, df, toe, knee = g
            p[s + '_an'] = (sg * HW + dr, dy, df); p[s + '_g'] = 0.0; p[s + '_pitch'] = 0.0
        p[s + '_toe'] = toe; p[s + '_knee'] = knee or (sg * 0.12, -0.5, 1.0)
    p['l_ua'], p['l_fa'] = la; p['r_ua'], p['r_fa'] = ra
    return {k: np.atleast_1d(np.array(v, float)) for k, v in p.items()}


def mirror(p):
    q = {}
    for k, v in p.items():
        k2 = 'r' + k[1:] if k.startswith('l_') else 'l' + k[1:] if k.startswith('r_') else k
        v = v.copy()
        if v.size == 3: v[0] = -v[0]
        if k == 'twist': v = -v
        q[k2] = v
    return q


def drop(p, *names):
    return {k: v for k, v in p.items() if k not in names}


ARMS = ('l_ua', 'l_fa', 'r_ua', 'r_fa')


def mix(a, b, t):
    return {k: a[k] * (1 - t) + b[k] * t for k in a}


def edit(p, **kw):
    q = {k: v.copy() for k, v in p.items()}
    for k, v in kw.items(): q[k] = np.atleast_1d(np.array(v, float))
    return q


# arms, written for the left arm: (upper arm, forearm) directions
SIDE = ((-0.78, -0.42, 0.28), (-0.74, -0.38, 0.50))
DOWN = ((-0.30, -0.93, 0.10), (-0.22, -0.90, 0.36))
FWD = ((-0.28, -0.20, 0.94), (-0.08, -0.10, 0.99))
BACK = ((-0.70, -0.22, -0.66), (-0.66, -0.15, -0.72))
ACROSS = ((0.05, -0.22, 0.97), (0.25, -0.10, 0.96))
LOWSIDE = ((-0.62, -0.70, -0.20), (-0.55, -0.72, 0.10))
BACKLOW = ((-0.30, -0.55, -0.78), (-0.25, -0.45, -0.85))
UPFWD = ((-0.30, 0.25, 0.92), (-0.10, 0.65, 0.75))
TUCK = ((-0.12, -0.90, 0.40), (0.82, 0.48, 0.30))
TUCK2 = ((-0.12, -0.90, 0.40), (0.80, 0.38, 0.44))
ROUND = ((-0.72, -0.42, 0.55), (0.55, -0.05, 0.83))
HALF = ((-0.42, -0.72, 0.55), (0.72, 0.30, 0.62))
TDARM = ((-0.85, -0.45, 0.25), (-0.88, -0.25, 0.35))
OPEN = ((-0.80, -0.22, 0.50), (-0.80, -0.12, 0.58))
CHECK_F = ((-0.62, -0.10, 0.78), (-0.60, -0.02, 0.80))      # checking arm, forward-side
CHECK_B = ((-0.95, -0.08, -0.28), (-0.96, 0.0, -0.25))      # checking arm, side-back
WING = ((-0.85, 0.05, -0.45), (-0.85, 0.10, -0.48))
REACH = ((-0.22, -0.30, 0.93), (0.05, -0.25, 0.97))
OVER = ((-0.22, 0.96, 0.08), (0.20, 0.97, 0.0))
WIDE = ((-0.95, 0.05, 0.10), (-0.95, 0.12, 0.12))
WIDE_UP = ((-0.85, 0.40, -0.10), (-0.80, 0.55, -0.05))
SPIRAL = ((-0.80, 0.15, -0.55), (-0.80, 0.20, -0.55))
VUP = ((-0.45, 0.88, 0.05), (-0.40, 0.91, 0.05))
ONE_UP = ((-0.25, 0.93, 0.15), (-0.05, 0.98, 0.10))

GLIDE = pose(0.76, ice(-0.11, 0.03), ice(0.11, -0.03), SIDE, Rm(SIDE), lean=(0, 1, 0.07))
READY = pose(0.79, ice(-0.10, 0.0, toe=(-0.2, 0, 1)), ice(0.10, 0.0, toe=(0.2, 0, 1)), DOWN, Rm(DOWN), lean=(0, 1, 0.02), spinw=0.3)

# ---- jump entries: (wind-up glide, reach / plant, take-off instant)
T_PREP = pose(0.70, free(-0.02, -0.66, -0.40, toe=(-0.3, -0.2, 0.9)), ice(0.03, 0.0), FWD, Rm(BACK), lean=(0, 1, 0.20), twist=-0.35)
T_PLANT = pose(0.60, ice(-0.12, -0.52, toe=(-0.25, 0, 1), pitch=0.75), ice(0.03, 0.02), FWD, Rm(LOWSIDE), lean=(0, 1, 0.28), twist=-0.45)
T_TK = pose(0.88, ice(-0.08, -0.10, toe=(-0.2, 0, 1), pitch=0.95), free(0.02, -0.60, 0.16, knee=(0.2, 0, 1)), HALF, Rm(HALF), lean=(0, 1, 0.06), twist=0.15)

F_PREP = pose(0.70, ice(-0.03, 0.0), free(0.02, -0.66, -0.40, toe=(0.3, -0.2, 0.9)), FWD, Rm(BACK), lean=(0, 1, 0.20), twist=-0.30)
F_PLANT = pose(0.60, ice(-0.03, 0.02), ice(0.12, -0.52, toe=(0.25, 0, 1), pitch=0.75), FWD, Rm(LOWSIDE), lean=(0, 1, 0.28), twist=-0.40)
F_TK = pose(0.88, free(-0.02, -0.60, 0.16, knee=(-0.2, 0, 1)), ice(0.08, -0.10, toe=(0.2, 0, 1), pitch=0.95), HALF, Rm(HALF), lean=(0, 1, 0.06), twist=0.15)

# Lutz: same pick as the flip, but from a long outside edge with the body counter-rotated
LZ_PREP = pose(0.66, ice(-0.02, 0.0), free(-0.05, -0.62, -0.46, toe=(0.3, -0.2, 0.9)), ACROSS, Rm(BACK), lean=(-0.14, 1, 0.22), twist=-0.55)
LZ_PLANT = pose(0.56, ice(-0.02, 0.02), ice(0.03, -0.60, toe=(0.2, 0, 1), pitch=0.75), ACROSS, Rm(LOWSIDE), lean=(-0.16, 1, 0.30), twist=-0.62)
LZ_TK = pose(0.88, free(-0.02, -0.60, 0.16, knee=(-0.2, 0, 1)), ice(0.06, -0.12, toe=(0.2, 0, 1), pitch=0.95), HALF, Rm(HALF), lean=(-0.03, 1, 0.06), twist=0.10)

# Salchow: back inside edge, the free leg sweeps round and the knee drives through
S_PREP = pose(0.70, ice(-0.03, 0.0), free(0.22, -0.58, -0.38, toe=(0.6, -0.2, 0.7)), FWD, Rm(BACK), lean=(0, 1, 0.15), twist=-0.35)
S_PLANT = pose(0.60, ice(-0.03, 0.02), free(0.30, -0.52, 0.02, toe=(0.5, -0.1, 0.8), knee=(0.5, 0, 1)), FWD, Rm(SIDE), lean=(0, 1, 0.22), twist=-0.10)
S_TK = pose(0.85, ice(-0.04, -0.02, pitch=0.60), free(0.02, -0.38, 0.36, knee=(0.1, 0.3, 1)), HALF, Rm(HALF), lean=(0, 1, 0.05), twist=0.25)

# Loop / Euler: both feet on the ice, sitting on the right back outside edge
LO_PREP = pose(0.68, ice(-0.07, 0.20, toe=(0.15, 0, 1)), ice(0.04, 0.0), FWD, Rm(SIDE), lean=(0, 1, 0.15), twist=-0.30)
LO_PLANT = pose(0.56, ice(-0.06, 0.22, toe=(0.2, 0, 1)), ice(0.04, 0.0), FWD, Rm(SIDE), lean=(0, 1, 0.25), twist=-0.40)
LO_TK = pose(0.85, free(0.03, -0.64, 0.14), ice(0.03, -0.02, pitch=0.60), HALF, Rm(HALF), lean=(0, 1, 0.05), twist=0.15)

# Axel: the only forward take-off; arms back, then the free knee and both arms swing through
A_PREP = pose(0.70, ice(-0.03, 0.0), free(0.04, -0.60, -0.50, toe=(0.3, -0.3, 0.9)), BACKLOW, Rm(BACKLOW), lean=(0, 1, 0.22))
A_PLANT = pose(0.58, ice(-0.03, 0.04), free(0.04, -0.62, -0.42, toe=(0.3, -0.3, 0.9)), LOWSIDE, Rm(LOWSIDE), lean=(0, 1, 0.30))
A_TK = pose(0.86, ice(-0.04, -0.04, pitch=0.65), free(0.03, -0.26, 0.40, knee=(0.0, 0.4, 1)), UPFWD, Rm(UPFWD), lean=(0, 1, 0.02), twist=0.10)

ENTRY = {'T': (T_PREP, T_PLANT, T_TK), 'F': (F_PREP, F_PLANT, F_TK), 'LZ': (LZ_PREP, LZ_PLANT, LZ_TK),
         'S': (S_PREP, S_PLANT, S_TK), 'LO': (LO_PREP, LO_PLANT, LO_TK), 'EU': (LO_PREP, LO_PLANT, LO_TK),
         'A': (A_PREP, A_PLANT, A_TK)}

# ---- air position: arms in, left leg crossed over the right
AIR_TIGHT = pose(0.80, free(0.115, -0.775, 0.085, toe=(0.2, -0.5, 1)), free(-0.03, -0.80, -0.02, toe=(0.0, -0.5, 1)),
                 TUCK, Rm(TUCK2), lean=(0, 1, 0.03), head=(0, 1, 0.03))
AIR_OPEN = pose(0.80, free(0.06, -0.76, 0.10), free(-0.01, -0.80, 0.0), ROUND, Rm(ROUND), lean=(0, 1, 0.04))

# ---- landing on the right back outside edge: knee bends, free leg extends back, arms open
TD_R = pose(0.79, free(0.06, -0.74, 0.10, toe=(0, -0.4, 1)), ice(0.02, 0.0, pitch=0.35), TDARM, Rm(TDARM), lean=(0, 1, 0.10), twist=-0.05)
LAND1_R = pose(0.72, free(-0.03, -0.66, -0.30, toe=(-0.4, -0.3, 0.8)), ice(0.02, 0.0), OPEN, Rm(OPEN), lean=(0, 1, 0.26), twist=-0.20)
LAND2_R = pose(0.62, free(-0.10, -0.38, -0.70, toe=(-0.8, -0.3, 0.5)), ice(0.02, 0.02), CHECK_F, Rm(CHECK_B),
               lean=(0, 1, 0.36), twist=-0.25, head=(0, 1, 0.12))
LAND3_R = pose(0.74, free(-0.08, -0.60, -0.50, toe=(-0.6, -0.3, 0.7)), ice(0.02, 0.0), OPEN, Rm(CHECK_B), lean=(0, 1, 0.20), twist=-0.12)
# Euler lands on the left back inside edge with the right leg free in front
TD_L = pose(0.79, ice(-0.02, 0.0, pitch=0.35), free(-0.02, -0.72, 0.14), TDARM, Rm(TDARM), lean=(0, 1, 0.08))
LAND1_L = pose(0.70, ice(-0.02, 0.0), free(0.04, -0.62, 0.22, knee=(0.3, 0, 1)), OPEN, Rm(OPEN), lean=(0, 1, 0.18), twist=-0.15)

# ---- spins (forward spin on the left foot, back spin on the right)
CAMEL_L = pose(0.79, ice(-0.01, 0.02), free(0.03, 0.03, -0.80, toe=(0.7, -0.7, 0.0)), WING, Rm(WING),
               lean=(0, 0.10, 1), head=(0, 0.75, 0.66), spinw=0.80)
SIT_L = pose(0.34, ice(-0.03, 0.08, knee=(-0.12, 0.2, 1)), free(0.03, -0.24, 0.76, toe=(0.5, 0.7, 0.3), knee=(0.1, 1, 0.2)), REACH, Rm(REACH),
             lean=(0, 0.62, 0.78), head=(0, 0.9, 0.45), spinw=1.05)
UP_OPEN_L = pose(0.79, ice(0.0, 0.0), free(-0.15, -0.62, 0.14, toe=(0.2, -0.3, 1), knee=(0.5, 0, 1)), ROUND, Rm(ROUND),
                 lean=(0, 1, 0.0), spinw=1.2)
UP_TIGHT_L = pose(0.80, ice(0.0, 0.0), free(-0.17, -0.70, 0.12, toe=(0.2, -0.3, 1), knee=(0.4, 0, 1)), OVER, Rm(OVER),
                  lean=(0, 1, 0.0), head=(0, 1, 0.0), spinw=2.1)
SPIN_ENTRY = pose(0.62, ice(-0.03, 0.04), free(0.26, -0.52, -0.50, toe=(0.7, -0.2, 0.5)), FWD, Rm(BACK),
                  lean=(0, 1, 0.24), twist=0.30, spinw=0.55)
CHANGE = pose(0.66, ice(-0.11, 0.0), ice(0.11, 0.0), OPEN, Rm(OPEN), lean=(0, 1, 0.22), spinw=0.75)
# flying camel: the right leg swings wide round the side and comes down to land; the take-off leg trails up behind
FC_TK = pose(0.86, ice(-0.04, -0.04, pitch=0.65), free(0.40, -0.42, -0.08, toe=(0.8, -0.3, 0.4), knee=(1, -0.2, 0.4)), OPEN, Rm(WIDE),
             lean=(0, 1, 0.30), twist=0.25, spinw=0.6)
FLY_CAMEL = pose(0.79, free(-0.04, 0.02, -0.74, toe=(-0.7, -0.7, 0)), free(0.14, -0.52, -0.26, toe=(0.6, -0.4, 0.6)), WING, Rm(WING),
                 lean=(0, 0.22, 0.97), head=(0, 0.8, 0.6), spinw=0.8)
FLY_SIT = pose(0.50, free(-0.02, -0.36, 0.16, knee=(-0.3, 0.3, 1)), free(0.03, -0.20, 0.70, toe=(0.5, 0.7, 0.3), knee=(0.1, 1, 0.2)), REACH, Rm(REACH),
               lean=(0, 0.75, 0.66), spinw=0.9)
FINISH = pose(0.80, ice(-0.09, 0.0, toe=(-0.25, 0, 1)), ice(0.09, -0.05, toe=(0.3, 0, 1)), ONE_UP, Rm(WIDE),
              lean=(0, 1, 0.0), head=(0, 1, 0.12), spinw=0.3)

# ---- sequences
STROKE_L = pose(0.66, ice(-0.03, 0.02), free(0.14, -0.56, -0.52, toe=(0.6, -0.3, 0.7)), ((-0.70, -0.15, 0.65), (-0.70, -0.05, 0.70)),
                Rm(((-0.85, -0.10, -0.35), (-0.88, -0.02, -0.30))), lean=(0, 1, 0.22))
THREE_PRE = pose(0.76, ice(-0.02, 0.0), free(-0.03, -0.66, -0.10, toe=(0.3, -0.2, 1)), ((-0.55, -0.10, -0.80), (-0.50, -0.05, -0.85)),
                 Rm(ACROSS), lean=(0, 1, 0.08), twist=0.55)
THREE_POST = pose(0.68, ice(-0.02, 0.0), free(0.05, -0.60, -0.40, toe=(0.5, -0.3, 0.8)), FWD, Rm(BACK), lean=(0, 1, 0.12), twist=-0.35)
BACK_R = pose(0.66, free(0.06, -0.56, 0.30, knee=(-0.3, 0, 1)), ice(0.03, 0.0), OPEN, Rm(WIDE), lean=(0, 1, 0.12), twist=0.20)
TWZ = pose(0.78, ice(-0.01, 0.0), free(-0.07, -0.62, 0.10, knee=(0.4, 0, 1)), TUCK, Rm(TUCK2), lean=(0, 1, 0.03))
TWZ_OUT = pose(0.66, ice(-0.02, 0.0), free(0.10, -0.52, -0.55, toe=(0.6, -0.3, 0.7)), WIDE, Rm(WIDE), lean=(0, 1, 0.20))
BRK = pose(0.70, free(-0.05, -0.62, 0.22), ice(0.02, 0.0), ONE_UP, Rm(((-0.90, -0.05, -0.20), (-0.92, 0.0, -0.15))),
           lean=(0.06, 1, 0.05), twist=-0.40)
TOE_L = pose(0.80, ice(-0.06, 0.05, pitch=0.70), free(0.02, -0.58, 0.16, knee=(0.2, 0, 1)), WIDE_UP, Rm(WIDE_UP), lean=(0, 1, 0.0))
SWEEP_R = pose(0.60, free(-0.16, -0.50, -0.50, toe=(-0.6, -0.3, 0.7)), ice(0.03, 0.02), ((-0.35, 0.70, 0.62), (-0.20, 0.85, 0.50)),
               Rm(((-0.80, 0.30, 0.50), (-0.75, 0.45, 0.50))), lean=(0, 1, 0.26), twist=0.25)
SPIRAL_L = pose(0.79, ice(-0.01, 0.02), free(0.03, 0.28, -0.74, toe=(0.7, -0.5, -0.3)), SPIRAL, Rm(SPIRAL),
                lean=(0, 0.32, 0.95), head=(0, 0.85, 0.5))
EAGLE = pose(0.77, ice(-0.36, 0.0, toe=(-1, 0, 0.02), knee=(-1, 0, 0.3)), ice(0.36, 0.0, toe=(1, 0, 0.02), knee=(1, 0, 0.3)),
             WIDE_UP, Rm(WIDE_UP), lean=(0, 1, -0.14), head=(0, 1, -0.10))
LUNGE = pose(0.40, ice(-0.04, 0.34, knee=(-0.1, 0, 1)), free(0.02, -0.30, -0.72, toe=(0.9, -0.2, 0.3), knee=(0.3, -1, 0.2)),
             ((-0.75, 0.25, -0.60), (-0.70, 0.35, -0.60)), Rm(((-0.75, 0.25, -0.60), (-0.70, 0.35, -0.60))),
             lean=(0, 1, 0.10), head=(0, 1, -0.10))


# ---------------------------------------------------------------- interpolation
def pchip(xk, yk, x):
    """Monotone cubic through the keys (no overshoot, flat ends)."""
    xk = np.asarray(xk, float); yk = np.asarray(yk, float).reshape(len(xk), -1)
    h = np.diff(xk); d = np.diff(yk, axis=0) / h[:, None]
    m = np.zeros_like(yk)
    for k in range(1, len(xk) - 1):
        w1 = 2 * h[k] + h[k - 1]; w2 = h[k] + 2 * h[k - 1]
        ok = d[k - 1] * d[k] > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            hm = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
        m[k] = np.where(ok, hm, 0.0)
    i = np.clip(np.searchsorted(xk, x, side='right') - 1, 0, len(xk) - 2)
    t = np.clip((x - xk[i]) / h[i], 0, 1)[:, None]
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
    return h00 * yk[i] + h10 * h[i][:, None] * m[i] + h01 * yk[i + 1] + h11 * h[i][:, None] * m[i + 1]


def clean_keys(keys, F, gap=3):
    """Sort, clamp to the clip, and drop an earlier key that sits closer than `gap` frames to the next."""
    keys = sorted(((int(min(max(f, 0), F - 1)), i, p) for i, (f, p) in enumerate(keys)), key=lambda k: (k[0], k[1]))
    out = []
    for f, _, p in keys:
        while out and f - out[-1][0] < gap: out.pop()
        out.append((f, p))
    if out[0][0] > 0: out.insert(0, (0, out[0][1]))
    if out[-1][0] < F - 1: out.append((F - 1, out[-1][1]))
    return out


def channels(keys, F):
    """Interpolate every pose channel. Two paths would otherwise cut through the body, so they get a via point:
    an arm swinging to the opposite direction passes by the side, a free leg passing the hip swings round it."""
    x = np.arange(F, dtype=float); ch = {}
    for k in keys[0][1]:
        xs = [float(f) for f, p in keys if k in p]; ys = [p[k] for _, p in keys if k in p]
        sg = -1.0 if k.startswith('l_') else 1.0
        if k.endswith(('_ua', '_fa')) or k.endswith('_an'):
            X, Y = [xs[0]], [ys[0]]
            for i in range(1, len(xs)):
                a, b = ys[i - 1], ys[i]; mid = None
                if xs[i] - xs[i - 1] >= 2:
                    if k.endswith('_an'):
                        hip = np.array([sg * HW, 0, 0]); ab = b - a; t = np.clip(np.dot(hip - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
                        near = np.linalg.norm(a + ab * t - hip)
                        if near < 0.40 and near < 0.6 * min(np.linalg.norm(a - hip), np.linalg.norm(b - hip)):
                            d = (a + b) / 2 - hip + np.array([sg * 0.45, -0.55, 0.0])
                            mid = hip + d / np.linalg.norm(d) * 0.62
                    elif np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) < -0.2:
                        d = a / np.linalg.norm(a) + b / np.linalg.norm(b) + np.array([sg * 0.9, -0.75, 0.25])
                        mid = d / np.linalg.norm(d)
                if mid is not None: X.append((xs[i - 1] + xs[i]) / 2); Y.append(mid)
                X.append(xs[i]); Y.append(ys[i])
            xs, ys = X, Y
        v = pchip(xs, ys, x)
        if k in ARMS or k in ('head', 'lean', 'twist'): v = savgol(v, 9, 2)   # nothing constrains these: round them off
        ch[k] = nrm(v) if k in VEC else v
    return ch


# ---------------------------------------------------------------- timelines
def jump_type(label):
    m = re.match(r'(\d)([A-Za-z]+)', label)
    return (m.group(2).upper(), int(m.group(1))) if m else ('T', 3)


def jump_keys(F, wins):
    """wins: [(takeoff, landing, type, strength)] in 0-based frames."""
    t0 = wins[0][0]
    K = [(0, GLIDE), (max(6, t0 - 78), GLIDE)]
    for j, (t0, t1, typ, strength) in enumerate(wins):
        prep, plant, tk = ENTRY[typ]
        gap = t0 - wins[j - 1][1] if j else 999
        if gap > 60: K += [(t0 - 44, prep), (t0 - 18, prep), (t0 - 7, plant)]
        elif gap > 34: K += [(t0 - 16, prep), (t0 - 6, plant)]
        elif gap > 16: K += [(t0 - 6, plant)]
        else: K += [(t0 - gap // 2, plant)]
        K.append((t0, tk if typ == 'A' else drop(tk, *ARMS)))   # arms pull in as one motion, plant -> air
        n = t1 - t0; ramp = max(4, round(n * 0.22))
        air = mix(AIR_OPEN, AIR_TIGHT, strength)
        left = typ == 'EU'
        K += [(t0 + ramp, air), (t1 - max(4, round(n * 0.14)), air), (t1, drop(TD_L if left else TD_R, *ARMS))]
        nxt = wins[j + 1][0] - t1 if j + 1 < len(wins) else None
        land1, land2 = (LAND1_L, LAND1_L) if left else (LAND1_R, LAND2_R)
        # before a loop or Euler the free foot comes straight down in front: no time to stretch it back first
        quick = nxt is not None and wins[j + 1][2] in ('LO', 'EU') and nxt <= 44
        if nxt is None:
            K += [(t1 + 7, land1), (t1 + 18, land2), (t1 + 52, land2), (t1 + 92, LAND3_R), (t1 + 150, GLIDE)]
        elif quick: pass
        elif nxt > 50: K += [(t1 + 7, land1), (t1 + 17, land2), (t1 + nxt - 30, land2)]
        elif nxt >= 44: K += [(t1 + 7, land1), (t1 + 16, land2)]
        elif nxt > 16: K += [(t1 + 7, land1)]
    return clean_keys(K, F)


def spin_keys(F, ev, sid):
    e = {k: v - 1 for k, v in ev.items()}
    flying = 'TAKEOFF' in e
    first_right = sid in ('FCSp', 'FCCoSp')          # a flying camel lands on the right foot (back camel)
    A = (lambda p: mirror(p)) if first_right else (lambda p: p)   # poses on the first foot
    B = (lambda p: p) if first_right else (lambda p: mirror(p))   # poses on the second foot
    K = [(0, READY)]
    if flying:
        tk, ap, ld = e['TAKEOFF'], e.get('APEX', e.get('FLIGHT_APEX')), e['LANDING']
        K += [(max(2, tk - 22), READY), (tk - 7, A_PLANT), (tk, FC_TK if 'CAMEL' in e else edit(A_TK, spinw=0.6))]
        K += [(ap, FLY_CAMEL if 'CAMEL' in e else FLY_SIT)]
        first = A(CAMEL_L) if 'CAMEL' in e else A(SIT_L)
        K += [(ld, edit(mix(first, A(UP_OPEN_L), 0.45), spinw=0.8))]
        start = ld
    else:
        en = e.get('ENTRY', 45)
        K += [(max(2, en - 14), READY), (en + 4, SPIN_ENTRY)]
        start = en + 4
    order = [(k, v) for k, v in sorted(e.items(), key=lambda kv: kv[1])
             if k in ('CAMEL', 'SIT', 'UPRIGHT', 'SIT_FIRST_FOOT', 'FOOT_CHANGE', 'CAMEL_SECOND_FOOT', 'SIT_SECOND_FOOT')]
    ex = e.get('EXIT', F - 30); spin_end = ex - 30
    foot = A; last = None; fc = e.get('FOOT_CHANGE')
    for i, (k, f) in enumerate(order):
        nf = order[i + 1][1] if i + 1 < len(order) else spin_end
        if k == 'FOOT_CHANGE':
            K += [(f - 5, CHANGE)]
            foot = B
            if not any(x.endswith('SECOND_FOOT') for x in e):
                # no marker for the second foot: sit, then (combination spins) rise into a fast upright spin
                combo = 'UPRIGHT' in e
                K += [(f + 45, foot(SIT_L))]
                if combo: K += [(f + 105, foot(SIT_L)), (f + 140, foot(UP_OPEN_L)), (f + 170, foot(UP_TIGHT_L)), (spin_end - 14, foot(UP_TIGHT_L))]
                else: K += [(spin_end - 22, foot(SIT_L))]
            continue
        if k == 'UPRIGHT' and fc is not None and f < fc:   # the source marks the change after the upright phase
            nf = fc - 20
        p = foot({'CAMEL': CAMEL_L, 'SIT': SIT_L, 'UPRIGHT': UP_OPEN_L, 'SIT_FIRST_FOOT': SIT_L,
                  'CAMEL_SECOND_FOOT': CAMEL_L, 'SIT_SECOND_FOOT': SIT_L}[k])
        hold = max(f + 6, nf - 26)
        if k == 'UPRIGHT':
            K += [(f, p), (min(f + 18, hold), foot(UP_TIGHT_L)), (hold, foot(UP_TIGHT_L))]
        else:
            K += [(f, p), (hold, p)]
    # exit: open out, step onto the right back outside edge and hold the check
    K += [(spin_end, edit(LAND1_R, spinw=0.6)), (spin_end + 16, edit(LAND2_R, spinw=0.3)),
          (min(F - 12, ex + 10), edit(LAND3_R, spinw=0.3)), (F - 1, FINISH)]
    return clean_keys(K, F)


def stsq_keys(F):
    s = F / 401.0; k = lambda f: int(round(f * s))
    P = [(0, READY, 0), (24, STROKE_L, 0), (52, THREE_PRE, 0), (66, THREE_POST, 1), (92, BACK_R, 1), (112, BACK_R, 1),
         (126, STROKE_L, 2), (146, TWZ, 2), (192, TWZ, 6), (210, TWZ_OUT, 6), (226, mirror(THREE_PRE), 6), (240, BRK, 5),
         (262, BRK, 5), (274, TOE_L, 5), (284, mirror(TOE_L), 5), (294, TOE_L, 5), (304, mirror(TOE_L), 5),
         (322, mirror(STROKE_L), 5), (338, mirror(THREE_PRE), 5), (358, SWEEP_R, 4), (380, SWEEP_R, 4), (400, GLIDE, 4)]
    return clean_keys([(k(f), p) for f, p, _ in P], F), [(k(f), o * np.pi) for f, _, o in P]


def chsq_keys(F):
    s = F / 401.0; k = lambda f: int(round(f * s))
    P = [(0, READY, 0), (22, mirror(STROKE_L), 0), (44, STROKE_L, 0), (70, SPIRAL_L, 0), (150, SPIRAL_L, 0), (172, STROKE_L, 0),
         (196, EAGLE, 0.5), (272, EAGLE, 0.5), (292, STROKE_L, 0), (312, LUNGE, 0), (352, LUNGE, 0), (382, FINISH, 0), (400, FINISH, 0)]
    return clean_keys([(k(f), p) for f, p, _ in P], F), [(k(f), o * np.pi) for f, _, o in P]


# ---------------------------------------------------------------- skates
def skate_ref(S, side, part_w, an0):
    """Foot frame of the source skate at frame 0: columns (right, up, toe), origin at the ankle."""
    Mb = part_w[BLADE[side]][0]
    Pl = S.mesh_points(BLADE[side], BLADE[side])
    Pw = Pl @ Mb[:3, :3].T + Mb[:3, 3]
    c = Pw.mean(0); Vt = np.linalg.svd(Pw - c, full_matrices=False)[2]
    a = Vt[0]
    up_ = nrm(ortho(an0 - c, a))                       # the ankle sits straight above the blade
    ext = (Pw - an0) @ a
    fwd = a if ext.max() > -ext.min() else -a          # the blade reaches further in front of the ankle than behind
    G = np.eye(4); G[:3, 0] = np.cross(fwd, up_); G[:3, 1] = up_; G[:3, 2] = fwd; G[:3, 3] = an0
    q = (Pw - an0) @ G[:3, :3]
    return G, q[::12] * SKATE_SCALE


def ik(H, A, pole_dir):
    l1, l2 = L['thigh'], L['shin']
    dv = A - H; dist = np.clip(np.linalg.norm(dv, axis=1), 0.18, (l1 + l2) * 0.995)
    dn = nrm(dv); pole = nrm(ortho(pole_dir, dn))
    a1 = (l1 * l1 + dist * dist - l2 * l2) / (2 * dist)
    hh = np.sqrt(np.maximum(l1 * l1 - a1 * a1, 0))
    return H + dn * a1[:, None] + pole * hh[:, None], H + dn * dist[:, None]


# ---------------------------------------------------------------- solve
def solve(ch, org, psi, dy, blade_q, ankle_h, align=None):
    F = len(psi)
    r = np.stack([np.sin(psi), np.zeros(F), np.cos(psi)], 1)
    f = np.stack([np.cos(psi), np.zeros(F), -np.sin(psi)], 1)
    Bv = lambda v, rr=r, ff=f: v[:, :1] * rr + v[:, 1:2] * UP + v[:, 2:3] * ff
    u = nrm(Bv(ch['lean'])); u_low = nrm(u + 0.3 * UP)
    tw = ch['twist'][:, 0]
    r_t = r * np.cos(tw)[:, None] + f * np.sin(tw)[:, None]; f_t = f * np.cos(tw)[:, None] - r * np.sin(tw)[:, None]
    r_up = nrm(ortho(r_t, u)); f_up = np.cross(u, r_up)
    r_low = nrm(ortho(r, u_low)); f_low = np.cross(u_low, r_low)
    if align is None: align = np.zeros(F)
    r_a = r * np.cos(align)[:, None] - f * np.sin(align)[:, None]; f_a = f * np.cos(align)[:, None] + r * np.sin(align)[:, None]
    g_rot = {s: np.clip(ch[s + '_g'][:, 0], 0, 1) for s in 'lr'}          # blade attitude: eases with the pose
    g = {s: g_rot[s] * (1 - smoothstep(dy / 0.06)) for s in 'lr'}        # blade height: lets go as soon as the body rises

    def feet(ph, lift):
        hipc = np.c_[org[:, 0], ph, org[:, 1]]
        out = {}
        for s, sg in (('l', -1), ('r', 1)):
            H = hipc + r_low * sg * HW
            an = hipc + Bv(ch[s + '_an']); an[:, 1] += lift[s]
            K, A = ik(H, an, Bv(ch[s + '_knee']))
            d = nrm(A - K)
            th = nrm(Bv(ch[s + '_toe'] * np.array([1, 0, 1.0]), r_a, f_a)); p = ch[s + '_pitch'][:, 0][:, None]
            toe_i = th * np.cos(p) - UP * np.sin(p); up_i = th * np.sin(p) + UP * np.cos(p)
            cand = ortho(Bv(ch[s + '_toe']), d); wc = smoothstep(np.linalg.norm(cand, axis=1) / 0.45); cand = nrm(cand)
            for i in range(1, F):
                if wc[i] < 0.999:
                    prev = cand[i - 1] - np.dot(cand[i - 1], d[i]) * d[i]
                    c = cand[i] * wc[i] + prev / max(np.linalg.norm(prev), 1e-9) * (1 - wc[i])
                    cand[i] = c / max(np.linalg.norm(c), 1e-9)
            tf = nrm(cand + 0.30 * d); uf = nrm(ortho(-d, tf))
            G = g_rot[s][:, None]
            toe = nrm(G * toe_i + (1 - G) * tf); upf = nrm(ortho(G * up_i + (1 - G) * uf, toe))
            Rf = np.stack([np.cross(toe, upf), upf, toe], 2)                    # columns right, up, toe
            low = A[:, 1] + (np.einsum('fj,nj->fn', Rf[:, 1, :], blade_q[s])).min(1)
            out[s] = dict(H=H, K=K, A=A, Rf=Rf, low=low, target=an)
        return hipc, out

    zero = {s: np.zeros(F) for s in 'lr'}
    ph = ankle_h + ch['ext'][:, 0] + dy
    hipc, ft = feet(ph, zero)
    # 1. the pelvis may not sit higher than a grounded leg can reach
    drop = np.zeros(F)
    for s in 'lr':
        gy = ft[s]['target'][:, 1] + (ICE_Y - ft[s]['low'])
        dxz = np.linalg.norm((ft[s]['H'] - ft[s]['target'])[:, [0, 2]], axis=1)
        phmax = gy + np.sqrt(np.maximum((0.985 * LEG) ** 2 - dxz ** 2, 0.01))
        drop = np.maximum(drop, g[s] * np.maximum(ph - phmax, 0))
    ph = ph - np.maximum(drop, savgol(drop, 9, 2))
    # 2. grounded blades touch the ice exactly; free blades never go through it
    lift = dict(zero)
    for it in range(8):
        hipc, ft = feet(ph, lift)
        for s in 'lr':
            gap = ICE_Y - ft[s]['low']
            lift[s] = lift[s] + g[s] * gap + (1 - g[s]) * np.maximum(gap, 0)
    for it in range(6):   # a free foot's angle follows its shin, so lifting it changes the blade's low point: close the residue
        hipc, ft = feet(ph, lift)
        for s in 'lr':
            gap = ICE_Y - ft[s]['low']
            lift[s] = lift[s] + np.where(gap > 0, gap + 0.001, 0)
    hipc, ft = feet(ph, lift)

    J = {'hipc': hipc, 'waist': hipc + u_low * L['waist']}
    shc = J['waist'] + u * (L['spine'] - L['waist'])
    J['neck'] = shc + u * L['clav_up']
    J['head_top'] = J['neck'] + nrm(Bv(ch['head'])) * L['head']
    Bt = lambda v: v[:, :1] * r_t + v[:, 1:2] * UP + v[:, 2:3] * f_t
    for s, sg in (('l', -1), ('r', 1)):
        J[s + '_hip'], J[s + '_kn'], J[s + '_an'] = ft[s]['H'], ft[s]['K'], ft[s]['A']
        J[s + '_sh'] = shc + r_up * sg * L['sh_w'] / 2
        J[s + '_el'] = J[s + '_sh'] + nrm(Bt(ch[s + '_ua'])) * L['uarm']
        fa = nrm(Bt(ch[s + '_fa']))
        J[s + '_wr'] = J[s + '_el'] + fa * L['farm']
        J[s + '_tip'] = J[s + '_wr'] + fa * L['hand']
        J[s + '_toe'] = ft[s]['A'] + ft[s]['Rf'][:, :, 2] * 0.17 - ft[s]['Rf'][:, :, 1] * ankle_h * 0.8   # preview only
        J[s + '_heel'] = ft[s]['A'] - ft[s]['Rf'][:, :, 2] * 0.07 - ft[s]['Rf'][:, :, 1] * ankle_h * 0.8
    J['f_low'] = hipc + f_low; J['f_up'] = J['neck'] + f_up
    return J, ft, g


# ---------------------------------------------------------------- main
def author(path, events, kind, mid):
    S = Scene(path); F = S.F
    events = marker_events(S) or events or {}
    R = S.track(ROOT)
    pel = S.track(SRC['pelvis'])[:, :3, 3]
    hl = S.track(SRC['r_hip'])[:, :3, 3] - S.track(SRC['l_hip'])[:, :3, 3]
    psi_src = np.unwrap(np.arctan2(hl[:, 0], hl[:, 2]))
    side_name = {'l': 'Left', 'r': 'Right'}
    parts = {s: sorted(n for n, i in S.names.items() if n.startswith(side_name[s] + ' skate · website ') and 'mesh' in S.N[i]) for s in 'lr'}
    part_w = {s: {n: S.track(n) for n in parts[s]} for s in 'lr'}
    Gref, blade_q = {}, {}
    for s in 'lr':
        Gref[s], blade_q[s] = skate_ref(S, s, part_w[s], S.world(S.names[SRC[s + '_an']], 0)[:3, 3])
    ankle_h = float(np.mean([-blade_q[s][:, 1].min() for s in 'lr']))
    fr = np.arange(F)
    dy = np.zeros(F); flights = []; yaw_keys = None; align = None
    pel_s = savgol(pel, 9, 2)
    org = pel_s[:, [0, 2]]; psi = psi_src.copy()

    if kind in ('jump', 'combination'):
        wins = []
        for t0, t1, label in flight_windows(events):
            typ, n = jump_type(label if label != 'jump' else mid)
            turns = abs(psi_src[t1] - psi_src[t0]) / (2 * np.pi)
            strength = float(np.clip((turns - 0.6) / 2.2, 0.3, 1.0))
            wins.append((t0, t1, typ, strength))
            base = np.interp(fr, [t0, t1], [pel_s[t0, 1], pel_s[t1, 1]])
            d = np.clip(pel_s[:, 1] - base, 0, None); d[(fr <= t0) | (fr >= t1)] = 0
            dy = np.maximum(dy, d)
            flights.append(dict(takeoff=t0 + 1, landing=t1 + 1, label=label, type=typ, turns=round(turns, 2),
                                strength=round(strength, 2), landing_foot='l' if typ == 'EU' else 'r'))
            if psi_src[t1] < psi_src[t0]: print('WARNING clockwise jump', mid, file=sys.stderr)
        keys = jump_keys(F, wins)
        # blades follow the direction of travel while gliding (not while the body is turning fast)
        v = np.gradient(savgol(pel, 21, 2), axis=0) * 60
        rr = np.stack([np.sin(psi), np.zeros(F), np.cos(psi)], 1); ff = np.stack([np.cos(psi), np.zeros(F), -np.sin(psi)], 1)
        phi = np.arctan2((v * rr).sum(1), (v * ff).sum(1))
        phi = np.where(np.cos(phi) < 0, phi - np.pi * np.sign(phi + 1e-9), phi)
        w = smoothstep(np.hypot(v[:, 0], v[:, 2]) / 0.4) * (1 - smoothstep(np.abs(np.gradient(psi)) * 60 / np.radians(250)))
        align = savgol(np.clip(phi, -0.7, 0.7) * w, 15, 2)
    elif kind == 'spin':
        keys = spin_keys(F, events, mid)
        ch0 = channels(keys, F)
        rate = np.gradient(psi_src) * 60
        spinning = np.abs(rate) > np.radians(100)
        a, b = np.where(spinning)[0][[0, -1]]
        w = ch0['spinw'][:, 0] * smoothstep((fr - a + 4) / 14.0) * smoothstep((b + 4 - fr) / 14.0)
        w = np.where((fr >= a) & (fr <= b), np.maximum(w, 0.12), 0)
        cs = np.cumsum(w); cs = cs / cs[-1]
        psi = np.where(fr < a, psi_src, psi_src[a] + (psi_src[b] - psi_src[a]) * cs)
        psi[fr > b] = psi_src[fr > b]
        # spin axis = pelvis path averaged over one revolution; the stance blade sits on it
        per = int(round(2 * np.pi / max(np.median(np.abs(rate[spinning])), 1e-3) * 60)) | 1
        ker = np.ones(per) / per
        pad = np.pad(pel[:, [0, 2]], ((per, per), (0, 0)), mode='edge')
        org = np.stack([np.convolve(pad[:, k], ker, 'same')[per:-per] for k in (0, 1)], 1)
        org = savgol(org, 15, 2)
        if 'TAKEOFF' in events:
            t0, t1 = events['TAKEOFF'] - 1, events['LANDING'] - 1
            base = np.interp(fr, [t0, t1], [pel_s[t0, 1], pel_s[t1, 1]])
            d = np.clip(pel_s[:, 1] - base, 0, None); d[(fr <= t0) | (fr >= t1)] = 0
            sgm = np.clip((fr - t0) / max(t1 - t0, 1), 0, 1)
            dy = np.maximum(d, 0.24 * 4 * sgm * (1 - sgm))
    else:
        keys, yk = (stsq_keys if mid == 'StSq' else chsq_keys)(F)
        v = np.gradient(savgol(pel, 31, 2), axis=0) * 60; sp = np.hypot(v[:, 0], v[:, 2])
        head = np.unwrap(np.arctan2(-v[:, 2], v[:, 0])); ok = sp > 0.25
        head = savgol(np.interp(fr, fr[ok], head[ok]), 21, 2)
        yk = sorted({f_: o for f_, o in yk}.items())
        psi = head + pchip([f_ for f_, _ in yk], [o for _, o in yk], fr.astype(float))[:, 0]

    ch = channels(keys, F)
    if kind == 'spin':   # put the grounded blade (not the pelvis) on the spin axis
        gsum = ch['l_g'][:, 0] + ch['r_g'][:, 0]
        off = (ch['l_an'] * ch['l_g'] + ch['r_an'] * ch['r_g']) / np.maximum(gsum, 1e-3)[:, None]
        off = savgol(off * np.clip(gsum, 0, 1)[:, None], 9, 2)
        rr = np.stack([np.sin(psi), np.cos(psi)], 1); ff = np.stack([np.cos(psi), -np.sin(psi)], 1)
        org = org - off[:, :1] * rr - off[:, 2:3] * ff
    J, ft, g = solve(ch, org, psi, dy, blade_q, ankle_h, align)

    skate = {}
    for s in 'lr':
        Gn = np.tile(np.eye(4), (F, 1, 1)); Gn[:, :3, :3] = ft[s]['Rf'] * SKATE_SCALE; Gn[:, :3, 3] = ft[s]['A']
        Gi = np.linalg.inv(Gref[s])
        for n in parts[s]: skate[n] = Gn @ (Gi @ part_w[s][n][0])
    low = np.minimum(ft['l']['low'], ft['r']['low'])
    grounded = np.maximum(g['l'], g['r']) > 0.98
    marks = sorted({k for k, _ in keys if 0 < k < F - 1})
    meta = dict(frames=F, flights=flights, ankle_height=round(ankle_h, 3),
                min_blade=round(float(low.min() - ICE_Y), 4),
                max_grounded_gap=round(float(np.abs(low[grounded] - ICE_Y).max()), 4) if grounded.any() else None,
                grounded_frames=int(grounded.sum()), key_frames=[int(k) + 1 for k in marks],
                lengths=L, turns_total=round(float(abs(psi[-1] - psi[0]) / (2 * np.pi)), 2))
    return J, skate, meta, R


if __name__ == '__main__':
    path, out, events_json, kind, mid = sys.argv[1:6]
    J, skate, meta, R = author(path, json.loads(events_json) if events_json != 'null' else None, kind, mid)
    names = sorted(skate)
    np.savez(out, **{'J_' + k: v for k, v in J.items()}, skate_names=np.array(names),
             skate_mats=np.stack([skate[n] for n in names]), tuck=np.zeros(meta['frames']), root=R, meta=json.dumps(meta))
    print(json.dumps(meta))
