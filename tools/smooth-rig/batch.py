"""Rebuild every non-baseline motion: author poses (author_motion.py), then skin and export in Blender (build_blender.py)."""
import json, subprocess, sys, os
from concurrent.futures import ThreadPoolExecutor
T = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('SMOOTH_RIG_REPO') or os.path.abspath(os.path.join(T, '..', '..'))
W = os.environ.get('SMOOTH_RIG_WORK', '/tmp/smooth-rig-work')
B = '/Applications/Blender.app/Contents/MacOS/Blender'
m = json.load(open(f'{REPO}/player-motions.json'))
items = [e for e in m if e.get('kind') != 'baseline']
args = [a for a in sys.argv[1:] if not a.startswith('--')]
only = {a.partition(':')[0]: a.partition(':')[2] for a in args}
if only: items = [e for e in items if e['id'] in only]
SRC = {k: v['src'] for k, v in json.load(open(os.path.join(T, 'r4-build-report.json'))).items()}
for d in ('proc', 'out', 'sheets'): os.makedirs(f'{W}/{d}', exist_ok=True)
def run(e):
    i, k = e['id'], e['kind']
    f = SRC[i]                         # always rebuild from Bill's source GLB, never from a rebuilt output
    npz = f'{W}/proc/{i}.npz'; out = f'{W}/out/{i}-r4-smooth-web.glb'
    r = subprocess.run(['python3', os.path.join(T, 'author_motion.py'), f'{REPO}/{f}', npz, json.dumps(e.get('event_frames')), k, i],
                       capture_output=True, text=True)
    if r.returncode: return i, {'error': 'author', 'log': r.stderr[-800:]}
    meta = json.loads(r.stdout.strip().splitlines()[-1]); meta.pop('lengths', None)
    cmd = [B, '-b', '--factory-startup', '--python', os.path.join(T, 'build_blender.py'), '--', f'{REPO}/{f}', npz, out, f'{W}/sheets/{i}']
    if only.get(i): cmd.append(only[i])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if 'BUILD_OK' not in r.stdout: return i, {'error': 'blender', 'log': (r.stdout + r.stderr)[-1500:]}
    meta['out'] = os.path.basename(out); meta['src'] = f; meta['kind'] = k
    return i, meta
res = {}
with ThreadPoolExecutor(4) as ex:
    for i, meta in ex.map(run, items):
        res[i] = meta; print(i, 'ERROR ' + meta['error'] + meta['log'] if 'error' in meta else 'ok', flush=True)
json.dump(res, open(f'{W}/sheets/results.json', 'w'), indent=1, ensure_ascii=False)
print('DONE', sum('error' not in v for v in res.values()), '/', len(res))
