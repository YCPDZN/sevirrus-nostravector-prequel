"""Flatten a nested NOSTRAVECTOR character export into the flat, metre-scale layout rig_glb.py expects
(every part a direct child of the root, transforms baked in), sized to match a reference character.
usage: python3 flatten_glb.py nested.glb reference_flat.glb out.glb"""
import json, struct, sys, re
import numpy as np
def load(f):
    d = open(f, 'rb').read(); n = struct.unpack('<I', d[12:16])[0]; gl = json.loads(d[20:20 + n]); bo = 20 + n; bl = struct.unpack('<I', d[bo:bo + 4])[0]
    return gl, d[bo + 8:bo + 8 + bl]
DT = {5126: '<f4', 5125: '<u4', 5123: '<u2', 5121: 'u1'}; NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}
def reader(gl, b):
    def read(ai):
        a = gl['accessors'][ai]; bv = gl['bufferViews'][a['bufferView']]; o = bv.get('byteOffset', 0) + a.get('byteOffset', 0); n = NC[a['type']]; dt = np.dtype(DT[a['componentType']]); st = bv.get('byteStride', 0)
        if st and st != dt.itemsize * n:
            raw = np.frombuffer(b, np.uint8, st * a['count'], o).reshape(a['count'], st); return raw[:, :dt.itemsize * n].copy().view(dt).reshape(a['count'], n).astype(np.float64)
        return np.frombuffer(b, dt, a['count'] * n, o).reshape(a['count'], n).astype(np.float64)
    return read
def qm(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
def local(nd):
    if 'matrix' in nd: return np.array(nd['matrix'], float).reshape(4, 4).T
    m = np.eye(4); m[:3, :3] = qm(nd.get('rotation', [0, 0, 0, 1])) * np.array(nd.get('scale', [1, 1, 1])); m[:3, 3] = nd.get('translation', [0, 0, 0]); return m
src, ref, dst = sys.argv[1:4]
gl, b = load(src); read = reader(gl, b); N = gl['nodes']
parts = []
def walk(i, m):
    nd = N[i]; w = m @ local(nd)
    if 'mesh' in nd:
        for p in gl['meshes'][nd['mesh']]['primitives']:
            v = read(p['attributes']['POSITION']) @ w[:3, :3].T + w[:3, 3]
            nm = read(p['attributes']['NORMAL']) @ np.linalg.inv(w[:3, :3]) if 'NORMAL' in p['attributes'] else np.tile([0, 1, 0], (len(v), 1))
            nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
            ix = read(p['indices']).reshape(-1).astype(np.int64) if 'indices' in p else np.arange(len(v))
            if np.linalg.det(w[:3, :3]) < 0: ix = ix.reshape(-1, 3)[:, ::-1].reshape(-1)
            parts.append((nd.get('name', 'part'), v, nm, ix, p.get('material', 0)))
    for c in nd.get('children', []): walk(c, w)
walk(gl['scenes'][gl.get('scene', 0)]['nodes'][0], np.eye(4))
# size and place it like the reference character
rgl, rb = load(ref); rread = reader(rgl, rb)
rv = np.vstack([rread(p['attributes']['POSITION']) for m in rgl['meshes'] for p in m['primitives']])
allv = np.vstack([p[1] for p in parts]); lo, hi = allv.min(0), allv.max(0)
k = (rv[:, 1].max() - rv[:, 1].min()) / (hi[1] - lo[1]); off = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
nose = [p for p in parts if p[0] == 'Nose']
flip = nose and nose[0][1][:, 2].mean() < (lo[2] + hi[2]) / 2
out = bytearray(); views = []; accs = []; meshes = []; nodes = [{'name': N[gl['scenes'][0]['nodes'][0]].get('name', 'Character'), 'children': []}]
def add(raw, ct, cnt, typ, tgt, **kw):
    while len(out) % 4: out.append(0)
    views.append({'buffer': 0, 'byteOffset': len(out), 'byteLength': len(raw), 'target': tgt}); out.extend(raw)
    a = {'bufferView': len(views) - 1, 'componentType': ct, 'count': cnt, 'type': typ}; a.update(kw); accs.append(a); return len(accs) - 1
for name, v, nm, ix, mat in parts:
    v = (v + off) * k
    if flip: v[:, [0, 2]] *= -1; nm = nm.copy(); nm[:, [0, 2]] *= -1
    v = v.astype('<f4'); nm = nm.astype('<f4')
    pa = add(v.tobytes(), 5126, len(v), 'VEC3', 34962, min=v.min(0).tolist(), max=v.max(0).tolist()); na = add(nm.tobytes(), 5126, len(v), 'VEC3', 34962)
    ia = add(ix.astype('<u4').tobytes(), 5125, len(ix), 'SCALAR', 34963)
    meshes.append({'name': name, 'primitives': [{'attributes': {'POSITION': pa, 'NORMAL': na}, 'indices': ia, 'material': mat}]})
    nodes.append({'name': name, 'mesh': len(meshes) - 1}); nodes[0]['children'].append(len(nodes) - 1)
while len(out) % 4: out.append(0)
o = {'asset': {'version': '2.0', 'generator': 'flatten_glb'}, 'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': nodes, 'meshes': meshes, 'materials': gl.get('materials', []), 'accessors': accs, 'bufferViews': views, 'buffers': [{'byteLength': len(out)}]}
js = json.dumps(o, separators=(',', ':')).encode()
while len(js) % 4: js += b' '
open(dst, 'wb').write(struct.pack('<4sII', b'glTF', 2, 28 + len(js) + len(out)) + struct.pack('<I4s', len(js), b'JSON') + js + struct.pack('<I4s', len(out), b'BIN\x00') + bytes(out))
print(dst, len(parts), 'parts, scale', round(k, 4), 'flip', bool(flip))
