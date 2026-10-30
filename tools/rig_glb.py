"""Add a real skeleton (armature + skin weights) to the NOSTRAVECTOR character GLBs.

Every part of the model is bound to the bone it belongs to. Soft parts (sleeves, trousers,
torso, neck, collar) blend smoothly across the joint they cross, so they bend instead of
splitting. Rigid parts (armour, buckles, jewellery, hair, face) follow a single bone.

usage: python3 rig_glb.py in.glb out.glb
"""
import json, re, struct, sys
import numpy as np

# bone name, parent, rest-pose pivot (model units: metres, character faces +Z)
BONES = [
    ("Hips", None, (0, 0.95, 0)),
    ("Spine", "Hips", (0, 1.02, 0)),
    ("Neck", "Spine", (0, 1.45, 0)),
    ("Head", "Neck", (0, 1.56, 0)),
    ("Eye_L", "Head", None), ("Eye_R", "Head", None),          # placed from the eye geometry
    ("UpperArm_L", "Spine", (0.27, 1.37, 0)), ("LowerArm_L", "UpperArm_L", (0.32, 1.10, 0)), ("Hand_L", "LowerArm_L", (0.34, 0.84, 0.03)),
    ("UpperArm_R", "Spine", (-0.27, 1.37, 0)), ("LowerArm_R", "UpperArm_R", (-0.32, 1.10, 0)), ("Hand_R", "LowerArm_R", (-0.34, 0.84, 0.03)),
    ("UpperLeg_L", "Hips", (0.11, 0.87, 0)), ("LowerLeg_L", "UpperLeg_L", (0.13, 0.49, 0)), ("Foot_L", "LowerLeg_L", (0.14, 0.14, -0.01)),
    ("UpperLeg_R", "Hips", (-0.11, 0.87, 0)), ("LowerLeg_R", "UpperLeg_R", (-0.13, 0.49, 0)), ("Foot_R", "LowerLeg_R", (-0.14, 0.14, -0.01)),
]

HEAD_RX = re.compile(r"^(face|jaw|nose|eye white|ear|mouth|upper lip|lower lip|upper teeth|tongue|nostril|brow|eyebrow|hair crown|parted fringe|long hair|beard|moustache|imperial crown|crown)", re.I)
EYE_RX = re.compile(r"^(iris|pupil|eye glint)", re.I)
SOFT_RX = re.compile(r"upperArm|lowerArm|thigh|shin|Pants hips|Puffer body|^Neck|Sleeve stitch|^Collar|Waistcoat|Jacket lapel|Shirt collar", re.I)


def bone_for(name, cx):
    s = "L" if cx >= 0 else "R"
    if EYE_RX.match(name): return "Eye_" + s
    if HEAD_RX.match(name): return "Head"
    if re.match(r"^(Neck|Collar|Shirt collar)", name): return "Neck"
    if re.match(r"^(Pants hips|Jacket hem|Utility belt|Command belt buckle)", name): return "Hips"
    if re.search(r"upperArm|pauldron|shoulder inset|Sleeve stitch|epaulette", name, re.I): return "UpperArm_" + s
    if re.search(r"lowerArm|^Cuff|carapace", name, re.I): return "LowerArm_" + s
    if re.match(r"^(Palm|Finger|Thumb)", name): return "Hand_" + s
    if re.search(r"thigh", name, re.I): return "UpperLeg_" + s
    if re.search(r"shin|Boot strap|Boot buckle|Buckle inset", name, re.I): return "LowerLeg_" + s
    if re.search(r"foot boot|Boot sole|Boot toe cap", name, re.I): return "Foot_" + s
    return "Spine"


def main(src, dst):
    data = open(src, "rb").read()
    jlen = struct.unpack("<I", data[12:16])[0]
    gl = json.loads(data[20:20 + jlen])
    boff = 20 + jlen
    blen = struct.unpack("<I", data[boff:boff + 4])[0]
    binchunk = bytearray(data[boff + 8:boff + 8 + blen])

    def read_vec3(acc_i):
        a = gl["accessors"][acc_i]; bv = gl["bufferViews"][a["bufferView"]]
        o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(binchunk, dtype="<f4", count=a["count"] * 3, offset=o).reshape(-1, 3).astype(np.float64)

    root = gl["nodes"][0]
    mesh_nodes = [i for i in root["children"] if "mesh" in gl["nodes"][i]]

    # eye bones sit at the centre of each eyeball
    piv = {n: (np.array(p, float) if p is not None else None) for n, _, p in BONES}
    for i in mesh_nodes:
        nd = gl["nodes"][i]
        m = re.match(r"^Eye white (-?1)$", nd["name"])
        if m:
            v = read_vec3(gl["meshes"][nd["mesh"]]["primitives"][0]["attributes"]["POSITION"])
            c = (v.min(0) + v.max(0)) / 2
            piv["Eye_L" if c[0] >= 0 else "Eye_R"] = np.array([c[0], c[1], v[:, 2].min() + 0.005])
    for n in ("Eye_L", "Eye_R"):
        if piv[n] is None: piv[n] = np.array([0.075 if n == "Eye_L" else -0.075, 1.745, 0.15])

    names = [b[0] for b in BONES]
    parent = {b[0]: b[1] for b in BONES}
    idx = {n: k for k, n in enumerate(names)}

    # boundaries a soft part can blend across: (parent bone, child bone, point, direction toward the child side, radius)
    def nrm(v): return v / np.linalg.norm(v)
    bounds = [("Hips", "Spine", piv["Spine"], np.array([0, 1.0, 0]), 0.32),
              ("Spine", "Neck", piv["Neck"], np.array([0, 1.0, 0]), 0.14),
              ("Neck", "Head", piv["Head"], np.array([0, 1.0, 0]), 0.12)]
    for s in "LR":
        bounds += [("Spine", "UpperArm_" + s, piv["UpperArm_" + s], nrm(piv["LowerArm_" + s] - piv["UpperArm_" + s]), 0.16),
                   ("UpperArm_" + s, "LowerArm_" + s, piv["LowerArm_" + s], nrm(piv["Hand_" + s] - piv["LowerArm_" + s]), 0.12),
                   ("LowerArm_" + s, "Hand_" + s, piv["Hand_" + s], nrm(piv["Hand_" + s] - piv["LowerArm_" + s]), 0.1),
                   ("Hips", "UpperLeg_" + s, piv["UpperLeg_" + s], nrm(piv["LowerLeg_" + s] - piv["UpperLeg_" + s]), 0.13),
                   ("UpperLeg_" + s, "LowerLeg_" + s, piv["LowerLeg_" + s], nrm(piv["Foot_" + s] - piv["LowerLeg_" + s]), 0.12)]
    D = 0.05     # half-width of the blend zone
    CAP = 0.85   # a soft part never hands more than this much of a vertex to the neighbouring bone

    def pad4(b):
        while len(b) % 4: b.append(0)

    def add_view(raw, target=None):
        pad4(binchunk); off = len(binchunk); binchunk.extend(raw)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(raw)}
        if target: bv["target"] = target
        gl["bufferViews"].append(bv); return len(gl["bufferViews"]) - 1

    def add_acc(view, ctype, count, typ, **kw):
        a = {"bufferView": view, "componentType": ctype, "count": count, "type": typ}; a.update(kw)
        gl["accessors"].append(a); return len(gl["accessors"]) - 1

    report = {}
    for i in mesh_nodes:
        nd = gl["nodes"][i]; name = nd["name"]
        for prim in gl["meshes"][nd["mesh"]]["primitives"]:
            v = read_vec3(prim["attributes"]["POSITION"])
            c = (v.min(0) + v.max(0)) / 2
            b = bone_for(name, c[0])
            n = len(v)
            joints = np.zeros((n, 4), np.uint8); weights = np.zeros((n, 4), np.float32)
            joints[:, 0] = idx[b]; weights[:, 0] = 1
            if SOFT_RX.search(name):
                best_w = np.zeros(n); best_o = np.full(n, idx[b])
                for pb, cb, pt, dr, rad in bounds:
                    if b not in (pb, cb): continue
                    rel = v - pt; t = rel @ dr
                    perp = np.linalg.norm(rel - np.outer(t, dr), axis=1)
                    w = np.clip(0.5 - t / (2 * D), 0, CAP) if b == cb else np.clip(0.5 + t / (2 * D), 0, CAP)
                    w = np.where(perp < rad, w, 0)
                    other = idx[pb] if b == cb else idx[cb]
                    take = w > best_w
                    best_w = np.where(take, w, best_w); best_o = np.where(take, other, best_o)
                joints[:, 1] = best_o; weights[:, 0] = 1 - best_w; weights[:, 1] = best_w
            jv = add_view(joints.tobytes(), 34962); wv = add_view(weights.tobytes(), 34962)
            prim["attributes"]["JOINTS_0"] = add_acc(jv, 5121, n, "VEC4")
            prim["attributes"]["WEIGHTS_0"] = add_acc(wv, 5126, n, "VEC4")
            report.setdefault(b, []).append(name + (" (blended)" if SOFT_RX.search(name) else ""))
        nd["skin"] = 0

    # bone nodes
    first = len(gl["nodes"])
    for n in names:
        p = parent[n]
        t = piv[n] - (piv[p] if p else 0)
        gl["nodes"].append({"name": n, "translation": [round(float(x), 6) for x in t]})
    for n in names:
        p = parent[n]
        if p: gl["nodes"][first + idx[p]].setdefault("children", []).append(first + idx[n])
    root["children"].append(first + idx["Hips"])

    ibm = np.zeros((len(names), 16), np.float32)
    for k, n in enumerate(names):
        m = np.eye(4, dtype=np.float32); m[0:3, 3] = -piv[n]
        ibm[k] = m.T.reshape(-1)          # glTF matrices are column-major
    iv = add_view(ibm.tobytes())
    gl["skins"] = [{"name": "Armature", "joints": [first + k for k in range(len(names))], "skeleton": first + idx["Hips"],
                    "inverseBindMatrices": add_acc(iv, 5126, len(names), "MAT4")}]
    gl["asset"]["generator"] = gl["asset"].get("generator", "") + " + rigged (Armature, %d bones)" % len(names)

    pad4(binchunk)
    gl["buffers"][0]["byteLength"] = len(binchunk)
    js = json.dumps(gl, separators=(",", ":")).encode()
    while len(js) % 4: js += b" "
    out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(binchunk))
    out += struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(binchunk), b"BIN\x00") + bytes(binchunk)
    open(dst, "wb").write(out)
    for b in names:
        print(f"{b:11s} {len(report.get(b, []))} parts")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
