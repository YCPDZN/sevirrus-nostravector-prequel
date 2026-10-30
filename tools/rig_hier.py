"""Rig the nested NOSTRAVECTOR character exports (Damix Chompjak, Chompy soldier) for the game.

These models are a scaled root with body / limb / head groups, each holding many small parts.
This tool:
  1. bakes every part into world space (character stands on y = 0, faces +Z, about 1.8 m tall),
  2. binds each part to one of the game's 18 bones (soft cloth blends across the joint it crosses),
  3. merges all parts that share a material look into one primitive, carrying base colour as
     vertex colour, so a whole soldier draws in a handful of calls instead of ~160.

usage: python3 rig_hier.py in.glb out.glb [height_m]
"""
import json, re, struct, sys
import numpy as np

BONES = ["Hips", "Spine", "Neck", "Head", "Eye_L", "Eye_R",
         "UpperArm_L", "LowerArm_L", "Hand_L", "UpperArm_R", "LowerArm_R", "Hand_R",
         "UpperLeg_L", "LowerLeg_L", "Foot_L", "UpperLeg_R", "LowerLeg_R", "Foot_R"]
PARENT = {"Hips": None, "Spine": "Hips", "Neck": "Spine", "Head": "Neck", "Eye_L": "Head", "Eye_R": "Head",
          "UpperArm_L": "Spine", "LowerArm_L": "UpperArm_L", "Hand_L": "LowerArm_L",
          "UpperArm_R": "Spine", "LowerArm_R": "UpperArm_R", "Hand_R": "LowerArm_R",
          "UpperLeg_L": "Hips", "LowerLeg_L": "UpperLeg_L", "Foot_L": "LowerLeg_L",
          "UpperLeg_R": "Hips", "LowerLeg_R": "UpperLeg_R", "Foot_R": "LowerLeg_R"}
IDX = {n: i for i, n in enumerate(BONES)}
SOFT = re.compile(r"^(Sleeve|Trouser leg|Undersuit torso|Shoulder cable)", re.I)


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def local(nd):
    if "matrix" in nd: return np.array(nd["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    r = quat_mat(nd.get("rotation", [0, 0, 0, 1])); s = np.array(nd.get("scale", [1, 1, 1]), float)
    m[:3, :3] = r * s; m[:3, 3] = nd.get("translation", [0, 0, 0]); return m


def main(src, dst, height=1.8):
    data = open(src, "rb").read()
    jl = struct.unpack("<I", data[12:16])[0]; gl = json.loads(data[20:20 + jl])
    bo = 20 + jl; bl = struct.unpack("<I", data[bo:bo + 4])[0]; bin_ = data[bo + 8:bo + 8 + bl]
    N, A, BV = gl["nodes"], gl["accessors"], gl["bufferViews"]
    DT = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}; NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def read(ai):
        a = A[ai]; bv = BV[a["bufferView"]]; o = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        n = NC[a["type"]]; dt = np.dtype(DT[a["componentType"]]); st = bv.get("byteStride", 0)
        if st and st != dt.itemsize * n:
            raw = np.frombuffer(bin_, dtype=np.uint8, count=st * a["count"], offset=o).reshape(a["count"], st)
            arr = raw[:, :dt.itemsize * n].copy().view(dt).reshape(a["count"], n)
        else:
            arr = np.frombuffer(bin_, dtype=dt, count=a["count"] * n, offset=o).reshape(a["count"], n)
        return arr.astype(np.float64) if a["componentType"] == 5126 else arr.astype(np.int64)

    # gather every mesh part in world space, remembering which top-level group it hangs under
    parts = []
    def walk(i, m, group):
        nd = N[i]; w = m @ local(nd)
        if "mesh" in nd:
            for p in gl["meshes"][nd["mesh"]]["primitives"]:
                v = read(p["attributes"]["POSITION"]); v = v @ w[:3, :3].T + w[:3, 3]
                nm = read(p["attributes"]["NORMAL"]) @ np.linalg.inv(w[:3, :3]) if "NORMAL" in p["attributes"] else None
                if nm is not None: nm /= np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)
                ix = read(p["indices"]).reshape(-1) if "indices" in p else np.arange(len(v))
                if np.linalg.det(w[:3, :3]) < 0: ix = ix.reshape(-1, 3)[:, ::-1].reshape(-1)
                parts.append(dict(name=nd.get("name", ""), group=group, v=v, n=nm, i=ix, mat=gl["materials"][p.get("material", 0)] if "materials" in gl else {}))
        for c in nd.get("children", []): walk(c, w, group or N[c].get("name", ""))
    root_i = gl["scenes"][gl.get("scene", 0)]["nodes"][0]
    walk(root_i, np.eye(4), "")
    for p in parts:
        if p["group"] == N[root_i].get("name"): p["group"] = ""

    # stand on the ground, centred, scaled to height
    allv = np.vstack([p["v"] for p in parts]); lo, hi = allv.min(0), allv.max(0)
    k = height / (hi[1] - lo[1]); off = np.array([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    for p in parts: p["v"] = (p["v"] + off) * k
    # facing: the nose should be in front (+Z)
    nose = next((p for p in parts if p["name"] == "Nose"), None)
    if nose is not None and nose["v"][:, 2].mean() < 0:
        for p in parts:
            p["v"][:, [0, 2]] *= -1
            if p["n"] is not None: p["n"][:, [0, 2]] *= -1

    by = {}
    for p in parts: by.setdefault(p["name"], []).append(p)
    def box(name):
        vs = np.vstack([q["v"] for q in by[name]]); return vs.min(0), vs.max(0), (vs.min(0) + vs.max(0)) / 2
    def side_name(base, s):
        # parts are named for the model's own left/right; pick the one on the requested side of x
        cands = [n for n in by if n.startswith(base + " ")]
        for n in cands:
            if (box(n)[2][0] >= 0) == (s == "L"): return n
        raise KeyError(base)

    piv = {}
    plo, phi, pc = box("Pelvis"); piv["Hips"] = np.array([0, pc[1], 0])
    blo, bhi, bc = box("Belt"); piv["Spine"] = np.array([0, bhi[1] + 0.03, 0])
    flo, fhi, fc = box("Face"); jlo = box("Jaw")[0]
    piv["Head"] = np.array([0, jlo[1] + 0.02, fc[2] - 0.04])
    piv["Neck"] = np.array([0, piv["Head"][1] - 0.1, 0])
    for s in "LR":
        side = "leftArm" if True else ""
        sh = box(side_name("Shoulder", s) if False else [n for n in by if n.startswith("Shoulder ") and n.endswith("armour") and (box(n)[2][0] >= 0) == (s == "L")][0])
        el = box([n for n in by if n.startswith("Elbow ") and (box(n)[2][0] >= 0) == (s == "L")][0])
        gv = box([n for n in by if n.startswith("Glove ") and (box(n)[2][0] >= 0) == (s == "L")][0])
        piv["UpperArm_" + s] = np.array([sh[2][0] * 0.8, sh[2][1] - 0.02, sh[2][2] * 0.3])
        piv["LowerArm_" + s] = el[2].copy()
        piv["Hand_" + s] = np.array([gv[2][0], gv[1][1] - 0.02, gv[2][2]])
        tl = box([n for n in by if n.startswith("Trouser leg ") and (box(n)[2][0] >= 0) == (s == "L")][0])
        kn = box([n for n in by if n.startswith("Knee ") and n.endswith("armour") and (box(n)[2][0] >= 0) == (s == "L")][0])
        bt = box([n for n in by if n.startswith("Boot ") and (box(n)[2][0] >= 0) == (s == "L")][0])
        piv["UpperLeg_" + s] = np.array([tl[2][0], tl[1][1] - 0.06, 0])
        piv["LowerLeg_" + s] = np.array([kn[2][0], kn[2][1], tl[2][2]])
        piv["Foot_" + s] = np.array([bt[2][0], bt[0][1] + (bt[1][1] - bt[0][1]) * 0.55, tl[2][2]])
        ew = [n for n in by if n.startswith("Eye white") and (box(n)[2][0] >= 0) == (s == "L")]
        piv["Eye_" + s] = box(ew[0])[2] if ew else np.array([0.06 if s == "L" else -0.06, fc[1], fhi[2]])

    def nrm(v): return v / np.linalg.norm(v)
    def limb_bone(c, s, arm):
        a, b, d = (["UpperArm_", "LowerArm_", "Hand_"] if arm else ["UpperLeg_", "LowerLeg_", "Foot_"])
        p0, p1, p2 = piv[a + s], piv[b + s], piv[d + s]
        if (c - p1) @ nrm(p1 - p0) < (-0.02 if arm else -0.05): return a + s
        if (c - p2) @ nrm(p2 - p1) < 0: return b + s
        return d + s

    def bone_for(p):
        nm, g = p["name"], p["group"]; c = (p["v"].min(0) + p["v"].max(0)) / 2; s = "L" if c[0] >= 0 else "R"
        if re.match(r"^(Iris|Pupil)", nm): return "Eye_" + s
        if g in ("head", "jaw", "eyes"): return "Head"
        if g in ("leftArm", "rightArm"): return limb_bone(c, s, True)
        if g in ("leftLeg", "rightLeg"): return limb_bone(c, s, False)
        if re.match(r"^(Pelvis|Belt|Utility pouch|Holster|Sidearm)", nm): return "Hips"
        if c[1] > piv["Neck"][1] + 0.02: return "Neck"
        return "Spine"

    bounds = [("Hips", "Spine", piv["Spine"], np.array([0, 1.0, 0]), 0.4),
              ("Spine", "Neck", piv["Neck"], np.array([0, 1.0, 0]), 0.16)]
    for s in "LR":
        bounds += [("Spine", "UpperArm_" + s, piv["UpperArm_" + s], nrm(piv["LowerArm_" + s] - piv["UpperArm_" + s]), 0.16),
                   ("UpperArm_" + s, "LowerArm_" + s, piv["LowerArm_" + s], nrm(piv["Hand_" + s] - piv["LowerArm_" + s]), 0.13),
                   ("Hips", "UpperLeg_" + s, piv["UpperLeg_" + s], nrm(piv["LowerLeg_" + s] - piv["UpperLeg_" + s]), 0.15),
                   ("UpperLeg_" + s, "LowerLeg_" + s, piv["LowerLeg_" + s], nrm(piv["Foot_" + s] - piv["LowerLeg_" + s]), 0.13)]
    D, CAP = 0.05, 0.85

    def weights(p, b):
        v = p["v"]; n = len(v)
        jt = np.zeros((n, 4), np.uint8); wt = np.zeros((n, 4), np.float32); jt[:, 0] = IDX[b]; wt[:, 0] = 1
        # soft cloth and anything that straddles a limb joint gets a per-vertex split
        if SOFT.match(p["name"]) or p["group"] in ("leftArm", "rightArm", "leftLeg", "rightLeg"):
            if p["group"] in ("leftArm", "rightArm", "leftLeg", "rightLeg") and not SOFT.match(p["name"]):
                return jt, wt
            best_w = np.zeros(n); best_o = np.full(n, IDX[b]); own = np.full(n, IDX[b])
            # cloth along a limb: pick the segment per vertex, then blend at the joint
            if p["group"] in ("leftArm", "rightArm", "leftLeg", "rightLeg"):
                s = "L" if v[:, 0].mean() >= 0 else "R"; arm = p["group"].endswith("Arm")
                segs = (["UpperArm_", "LowerArm_", "Hand_"] if arm else ["UpperLeg_", "LowerLeg_", "Foot_"])
                p0, p1, p2 = (piv[x + s] for x in segs)
                t1 = (v - p1) @ nrm(p1 - p0); t2 = (v - p2) @ nrm(p2 - p1)
                own = np.where(t1 < 0, IDX[segs[0] + s], np.where(t2 < 0, IDX[segs[1] + s], IDX[segs[2] + s]))
            for pb, cb, pt, dr, rad in bounds:
                ib, ic = IDX[pb], IDX[cb]
                rel = v - pt; t = rel @ dr; perp = np.linalg.norm(rel - np.outer(t, dr), axis=1)
                is_c, is_p = own == ic, own == ib
                w = np.where(is_c, np.clip(0.5 - t / (2 * D), 0, CAP), np.where(is_p, np.clip(0.5 + t / (2 * D), 0, CAP), 0))
                w = np.where(perp < rad, w, 0); other = np.where(is_c, ib, ic)
                take = w > best_w; best_w = np.where(take, w, best_w); best_o = np.where(take, other, best_o)
            jt[:, 0] = own; jt[:, 1] = best_o; wt[:, 0] = 1 - best_w; wt[:, 1] = best_w
        return jt, wt

    # merge by material look
    groups = {}
    for p in parts:
        m = p["mat"]; pbr = m.get("pbrMetallicRoughness", {})
        col = pbr.get("baseColorFactor", [1, 1, 1, 1])
        em = tuple(round(x, 2) for x in m.get("emissiveFactor", [0, 0, 0]))
        key = (round(pbr.get("metallicFactor", 1), 1), round(pbr.get("roughnessFactor", 1), 1), em != (0, 0, 0) and em, col[3] < 0.999 and round(col[3], 2))
        b = bone_for(p); jt, wt = weights(p, b)
        nm = p["n"] if p["n"] is not None else np.tile([0, 1, 0], (len(p["v"]), 1))
        groups.setdefault(key, []).append((p["v"], nm, p["i"], np.tile(col[:3], (len(p["v"]), 1)), jt, wt, em))

    bin_out = bytearray(); views, accs, prims, mats = [], [], [], []
    def view(raw, target=None):
        while len(bin_out) % 4: bin_out.append(0)
        o = len(bin_out); bin_out.extend(raw); bv = {"buffer": 0, "byteOffset": o, "byteLength": len(raw)}
        if target: bv["target"] = target
        views.append(bv); return len(views) - 1
    def acc(raw, ctype, count, typ, target=None, **kw):
        a = {"bufferView": view(raw, target), "componentType": ctype, "count": count, "type": typ}; a.update(kw); accs.append(a); return len(accs) - 1
    for key, items in groups.items():
        V = np.vstack([x[0] for x in items]).astype(np.float32); Nn = np.vstack([x[1] for x in items]).astype(np.float32)
        C = np.vstack([x[3] for x in items]).astype(np.float32); J = np.vstack([x[4] for x in items]); W = np.vstack([x[5] for x in items])
        I, base = [], 0
        for x in items: I.append(x[2] + base); base += len(x[0])
        I = np.concatenate(I)
        W8 = np.round(W * 255).astype(np.int32); W8[:, 0] += 255 - W8.sum(1); W8 = W8.astype(np.uint8)
        itype, idt = (5123, "<u2") if len(V) < 65535 else (5125, "<u4")
        Cn = np.round(np.clip(C, 0, 1) * 65535).astype("<u2")
        attrs = {"POSITION": acc(V.tobytes(), 5126, len(V), "VEC3", 34962, min=V.min(0).tolist(), max=V.max(0).tolist()),
                 "NORMAL": acc(Nn.tobytes(), 5126, len(V), "VEC3", 34962),
                 "COLOR_0": acc(Cn.tobytes(), 5123, len(V), "VEC3", 34962, normalized=True),
                 "JOINTS_0": acc(J.astype(np.uint8).tobytes(), 5121, len(V), "VEC4", 34962),
                 "WEIGHTS_0": acc(W8.tobytes(), 5121, len(V), "VEC4", 34962, normalized=True)}
        ind = acc(I.astype(idt).tobytes(), itype, len(I), "SCALAR", 34963)
        met, rough, em, alpha = key
        mat = {"name": "m%d" % len(mats), "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, alpha or 1], "metallicFactor": met, "roughnessFactor": rough}}
        if em: mat["emissiveFactor"] = list(em)
        if alpha: mat["alphaMode"] = "BLEND"
        mats.append(mat); prims.append({"attributes": attrs, "indices": ind, "material": len(mats) - 1})

    nodes = [{"name": N[root_i].get("name", "Character"), "children": [1, 2]}, {"name": "Body", "mesh": 0, "skin": 0}]
    first = 2
    for b in BONES:
        p = PARENT[b]; t = piv[b] - (piv[p] if p else 0)
        nodes.append({"name": b, "translation": [round(float(x), 6) for x in t]})
    for b in BONES:
        if PARENT[b]: nodes[first + IDX[PARENT[b]]].setdefault("children", []).append(first + IDX[b])
    ibm = np.zeros((len(BONES), 16), np.float32)
    for k2, b in enumerate(BONES):
        m = np.eye(4, dtype=np.float32); m[0:3, 3] = -piv[b]; ibm[k2] = m.T.reshape(-1)
    ibm_a = acc(ibm.tobytes(), 5126, len(BONES), "MAT4")
    out = {"asset": {"version": "2.0", "generator": gl["asset"].get("generator", "") + " + rig_hier (merged, 18 bones)"},
           "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": [{"name": "Body", "primitives": prims}],
           "materials": mats, "skins": [{"name": "Armature", "joints": [first + k2 for k2 in range(len(BONES))], "skeleton": first, "inverseBindMatrices": ibm_a}],
           "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": 0}]}
    while len(bin_out) % 4: bin_out.append(0)
    out["buffers"][0]["byteLength"] = len(bin_out)
    js = json.dumps(out, separators=(",", ":")).encode()
    while len(js) % 4: js += b" "
    glb = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(bin_out)) + struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(bin_out), b"BIN\x00") + bytes(bin_out)
    open(dst, "wb").write(glb)
    print(f"{dst}: {len(parts)} parts -> {len(prims)} primitives, {len(glb)} bytes")
    for b in BONES: print(f"  {b:11s} {np.round(piv[b], 3)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 1.8)
