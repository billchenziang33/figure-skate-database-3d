"""Quick stick-figure contact sheet from an authored .npz (side view + front view)."""
import sys, json, numpy as np
from PIL import Image, ImageDraw
npz, out = sys.argv[1:3]
D = np.load(npz); J = {k[2:]: D[k] for k in D.files if k.startswith('J_')}
meta = json.loads(str(D['meta'])); F = meta['frames']
frames = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else meta['key_frames']
if len(frames) > 16: frames = frames[:: int(np.ceil(len(frames) / 16))]
W, H, S = 230, 250, 110
img = Image.new('RGB', (W * len(frames), H * 2 + 16), 'white'); dr = ImageDraw.Draw(img)
BONES = [('hipc', 'waist', 'k'), ('waist', 'neck', 'k'), ('neck', 'head_top', 'k'), ('l_sh', 'r_sh', 'k'), ('l_hip', 'r_hip', 'k')]
for s, c in (('l', (40, 90, 220)), ('r', (220, 50, 40))):
    BONES += [(s + '_hip', s + '_kn', c), (s + '_kn', s + '_an', c), (s + '_sh', s + '_el', c), (s + '_el', s + '_wr', c),
              (s + '_wr', s + '_tip', c), (s + '_heel', s + '_toe', c), (s + '_an', s + '_heel', c)]
for i, fr in enumerate(frames):
    f = min(max(fr - 1, 0), F - 1)
    fw = J['f_low'][f] - J['hipc'][f]; fw[1] = 0; fw /= max(np.linalg.norm(fw), 1e-9)
    rt = np.cross(fw, [0, 1, 0])
    c0 = J['hipc'][f] * [1, 0, 1]
    for row, (ax, depth) in enumerate(((fw, rt), (rt, -fw))):      # side view (facing right), front view
        ox, oy = i * W + W // 2, row * H + H - 22
        P = lambda p: (ox + float(np.dot(p - c0, ax)) * S, oy - float(p[1]) * S)
        dr.line([(i * W, oy), (i * W + W, oy)], fill=(170, 200, 230), width=1)
        order = sorted(BONES, key=lambda b: np.dot((J[b[0]][f] + J[b[1]][f]) / 2 - c0, depth))
        for a, b, col in order:
            dr.line([P(J[a][f]), P(J[b][f])], fill=(0, 0, 0) if col == 'k' else col, width=5 if col == 'k' else 4)
        hc = (J['neck'][f] + J['head_top'][f]) / 2 + (J['head_top'][f] - J['neck'][f]) * 0.1
        x, y = P(hc); dr.ellipse([x - 10, y - 12, x + 10, y + 12], outline=(0, 0, 0), width=3)
        nose = hc + (J['f_up'][f] - J['neck'][f]) * 0.11; nx, ny = P(nose); dr.ellipse([nx - 2, ny - 2, nx + 2, ny + 2], fill=(0, 0, 0))
    dr.text((i * W + 4, 2), f'f{fr}', fill=(0, 0, 0))
img.save(out); print(out, frames)
