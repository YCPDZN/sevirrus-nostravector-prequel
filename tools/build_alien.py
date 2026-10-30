"""Build the Carax alien settler (spacesuit, bubble helmet, life-support pack) as a rigged GLB.

Same 18-bone armature and bone names as the rigged Sevirrus and Jakal models, so every pose
and animation works on all of them. Character faces +Z, units are metres.

usage: python3 build_alien.py out.glb
"""
import io, json, struct, sys
import numpy as np
from PIL import Image

rng = np.random.default_rng(7)

# ---------------- skeleton ----------------
PIV = {
    "Hips": (0, 0.95, 0), "Spine": (0, 1.02, 0), "Neck": (0, 1.47, 0), "Head": (0, 1.56, 0),
    "Eye_L": (0.042, 1.70, 0.08), "Eye_R": (-0.042, 1.70, 0.08),
    "UpperArm_L": (0.25, 1.40, 0), "LowerArm_L": (0.28, 1.12, 0), "Hand_L": (0.30, 0.86, 0.02),
    "UpperArm_R": (-0.25, 1.40, 0), "LowerArm_R": (-0.28, 1.12, 0), "Hand_R": (-0.30, 0.86, 0.02),
    "UpperLeg_L": (0.11, 0.90, 0), "LowerLeg_L": (0.12, 0.50, 0.01), "Foot_L": (0.13, 0.11, 0),
    "UpperLeg_R": (-0.11, 0.90, 0), "LowerLeg_R": (-0.12, 0.50, 0.01), "Foot_R": (-0.13, 0.11, 0),
}
PARENT = {"Hips": None, "Spine": "Hips", "Neck": "Spine", "Head": "Neck", "Eye_L": "Head", "Eye_R": "Head",
          "UpperArm_L": "Spine", "LowerArm_L": "UpperArm_L", "Hand_L": "LowerArm_L",
          "UpperArm_R": "Spine", "LowerArm_R": "UpperArm_R", "Hand_R": "LowerArm_R",
          "UpperLeg_L": "Hips", "LowerLeg_L": "UpperLeg_L", "Foot_L": "LowerLeg_L",
          "UpperLeg_R": "Hips", "LowerLeg_R": "UpperLeg_R", "Foot_R": "LowerLeg_R"}
BONES = list(PIV.keys())
P = {k: np.array(v, float) for k, v in PIV.items()}

# ---------------- geometry helpers (all return verts, uvs, tris) ----------------
def rot_to(d):
    """rotation matrix taking +Y onto direction d"""
    d = d / np.linalg.norm(d); y = np.array([0, 1.0, 0]); v = np.cross(y, d); c = y @ d
    if np.linalg.norm(v) < 1e-9: return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * (1 / (1 + c))

def euler(x=0, y=0, z=0):
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

def lathe(profile, n=24, sx=1.0, sz=1.0, uscale=1.0):
    prof = np.array(profile, float); m = len(prof)
    th = np.linspace(0, 2 * np.pi, n + 1)
    V = []; UV = []
    L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(prof, axis=0), axis=1))]
    for i, (r, y) in enumerate(prof):
        for j, t in enumerate(th):
            V.append([r * np.cos(t) * sx, y, r * np.sin(t) * sz]); UV.append([j / n * uscale, L[i] / max(L[-1], 1e-6)])
    T = []
    for i in range(m - 1):
        for j in range(n):
            a, b = i * (n + 1) + j, i * (n + 1) + j + 1; c, d = a + n + 1, b + n + 1
            T += [[a, c, b], [b, c, d]]
    return np.array(V), np.array(UV), np.array(T)

def capsule(A, B, r, sx=1.0, sz=1.0, n=20):
    A, B = np.array(A, float), np.array(B, float); L = np.linalg.norm(B - A)
    prof = [(r * np.cos(a), r * np.sin(a)) for a in np.linspace(-np.pi / 2, 0, 7)] + [(r * np.cos(a), L + r * np.sin(a)) for a in np.linspace(0, np.pi / 2, 7)]
    V, UV, T = lathe(prof, n, sx, sz, uscale=2)
    return V @ rot_to(B - A).T + A, UV, T

def ellipsoid(c, radii, R=None, n=22, lat=(-np.pi / 2, np.pi / 2), rings=14):
    prof = [(np.cos(a), np.sin(a)) for a in np.linspace(lat[0], lat[1], rings)]
    V, UV, T = lathe(prof, n)
    V = V * np.array(radii)
    if R is not None: V = V @ R.T
    return V + np.array(c), UV, T

def cylinder(c, r, h, R=None, n=18, cap=True):
    prof = ([(0, -h / 2)] if cap else []) + [(r, -h / 2), (r, h / 2)] + ([(0, h / 2)] if cap else [])
    V, UV, T = lathe(prof, n)
    if R is not None: V = V @ R.T
    return V + np.array(c), UV, T

def torus(c, Rr, r, R=None, n=28, m=10, sz=1.0):
    V = []; UV = []
    for i in range(n + 1):
        u = 2 * np.pi * i / n
        for j in range(m + 1):
            v = 2 * np.pi * j / m
            V.append([(Rr + r * np.cos(v)) * np.cos(u), r * np.sin(v), (Rr + r * np.cos(v)) * np.sin(u) * sz]); UV.append([i / n * 4, j / m])
    T = []
    for i in range(n):
        for j in range(m):
            a = i * (m + 1) + j; b = a + 1; cc = a + m + 1; d = cc + 1
            T += [[a, b, cc], [b, d, cc]]
    V = np.array(V)
    if R is not None: V = V @ R.T
    return V + np.array(c), np.array(UV), np.array(T)

def box(c, size, R=None, bevel=0.0):
    sx, sy, sz = np.array(size) / 2
    faces = [((1, 0, 0), (0, 0, -1), (0, 1, 0)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
             ((0, -1, 0), (1, 0, 0), (0, 0, 1)), ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (-1, 0, 0), (0, 1, 0))]
    V = []; UV = []; T = []
    for nrm, u, v in faces:
        nrm, u, v = map(np.array, (nrm, u, v)); base = len(V)
        for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            p = nrm * [sx, sy, sz] + u * a * [sx, sy, sz] + v * b * [sx, sy, sz]
            V.append(p); UV.append([(a + 1) / 2, (b + 1) / 2])
        T += [[base, base + 1, base + 2], [base, base + 2, base + 3]]
    V = np.array(V, float)
    if R is not None: V = V @ R.T
    return V + np.array(c), np.array(UV, float), np.array(T)

# ---------------- materials ----------------
def camo_png(seed, palette):
    g = np.random.default_rng(seed); S = 256
    yy, xx = np.mgrid[0:S, 0:S] / S
    img = np.zeros((S, S, 3)); img[:] = palette[0]
    for layer, col in enumerate(palette[1:]):
        f = np.zeros((S, S))
        for _ in range(28):
            cx, cy, r = g.random(), g.random(), 0.04 + g.random() * 0.09
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    f += np.exp(-(((xx - cx - ox) / (r * (0.6 + g.random()))) ** 2 + ((yy - cy - oy) / r) ** 2))
        img[f > 0.55 + layer * 0.08] = col
    img += g.normal(0, 4, img.shape)                   # fabric grain
    buf = io.BytesIO(); Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, "PNG"); return buf.getvalue()

TEX = [camo_png(3, [(124, 118, 98), (98, 93, 76), (150, 143, 118), (78, 75, 63)]),
       camo_png(11, [(104, 99, 82), (84, 80, 66), (128, 122, 100), (66, 63, 53)])]
MATS = [
    ("Suit", dict(pbrMetallicRoughness=dict(baseColorTexture=dict(index=0), metallicFactor=0.0, roughnessFactor=0.92))),
    ("Pack", dict(pbrMetallicRoughness=dict(baseColorTexture=dict(index=1), metallicFactor=0.0, roughnessFactor=0.9))),
    ("Strap", dict(pbrMetallicRoughness=dict(baseColorFactor=[0.2, 0.18, 0.13, 1], metallicFactor=0.0, roughnessFactor=0.85))),
    ("Metal", dict(pbrMetallicRoughness=dict(baseColorFactor=[0.55, 0.57, 0.6, 1], metallicFactor=0.85, roughnessFactor=0.35))),
    ("Device", dict(pbrMetallicRoughness=dict(baseColorFactor=[0.35, 0.33, 0.28, 1], metallicFactor=0.3, roughnessFactor=0.6))),
    ("Dark", dict(pbrMetallicRoughness=dict(baseColorFactor=[0.04, 0.04, 0.045, 1], metallicFactor=0.2, roughnessFactor=0.55))),
    ("Glove", dict(pbrMetallicRoughness=dict(baseColorFactor=[0.12, 0.11, 0.1, 1], metallicFactor=0.0, roughnessFactor=0.8))),
    ("Skin", dict(pbrMetallicRoughness=dict(baseColorFactor=[0.21, 0.25, 0.3, 1], metallicFactor=0.1, roughnessFactor=0.45))),
    ("Eye", dict(pbrMetallicRoughness=dict(baseColorFactor=[0.005, 0.006, 0.01, 1], metallicFactor=0.3, roughnessFactor=0.08))),
    ("Glass", dict(alphaMode="BLEND", doubleSided=True, pbrMetallicRoughness=dict(baseColorFactor=[0.75, 0.85, 1.0, 0.16], metallicFactor=0.0, roughnessFactor=0.04))),
    ("Red light", dict(emissiveFactor=[1.0, 0.08, 0.05], pbrMetallicRoughness=dict(baseColorFactor=[0.6, 0.05, 0.03, 1]))),
    ("Blue screen", dict(emissiveFactor=[0.12, 0.45, 1.0], pbrMetallicRoughness=dict(baseColorFactor=[0.1, 0.3, 0.7, 1], roughnessFactor=0.2))),
    ("Glint", dict(emissiveFactor=[0.8, 0.85, 0.9], pbrMetallicRoughness=dict(baseColorFactor=[1, 1, 1, 1]))),
]
MI = {n: i for i, (n, _) in enumerate(MATS)}

# ---------------- parts: (name, mesh, material, bone, soft) ----------------
parts = []
def add(name, mesh, mat, bone, soft=False): parts.append((name, mesh, mat, bone, soft))

for s, sg in (("L", 1), ("R", -1)):
    sh, el, wr = P["UpperArm_" + s], P["LowerArm_" + s], P["Hand_" + s]
    hp, kn, an = P["UpperLeg_" + s], P["LowerLeg_" + s], P["Foot_" + s]
    add(f"Upper sleeve {s}", capsule(sh + [0, 0.02, 0], el, 0.07), "Suit", "UpperArm_" + s, True)
    add(f"Shoulder pad {s}", ellipsoid(sh + [sg * 0.025, 0.035, 0], (0.088, 0.06, 0.092), euler(z=-sg * 0.35), lat=(-0.3, np.pi / 2)), "Pack", "UpperArm_" + s)
    add(f"Lower sleeve {s}", capsule(el, wr + [0, 0.03, 0], 0.06), "Suit", "LowerArm_" + s, True)
    add(f"Elbow pad {s}", ellipsoid(el + [0, -0.01, -0.045], (0.052, 0.06, 0.035)), "Device", "LowerArm_" + s)
    add(f"Wrist ring {s}", cylinder(wr + [0, 0.035, 0], 0.062, 0.035, rot_to(el - wr)), "Metal", "LowerArm_" + s)
    add(f"Glove {s}", ellipsoid(wr + [0, -0.07, 0.012], (0.044, 0.072, 0.034)), "Glove", "Hand_" + s)
    add(f"Thumb {s}", capsule(wr + [-sg * 0.005, -0.03, 0.035], wr + [-sg * 0.02, -0.09, 0.055], 0.016), "Glove", "Hand_" + s)
    add(f"Cuff {s}", capsule(wr + [0, 0.005, 0], wr + [0, -0.02, 0], 0.052), "Glove", "Hand_" + s)
    add(f"Thigh suit {s}", capsule(hp + [0, 0.02, 0], kn, 0.09), "Suit", "UpperLeg_" + s, True)
    add(f"Thigh strap {s}", torus(hp + [0, -0.12, 0], 0.092, 0.013), "Strap", "UpperLeg_" + s)
    add(f"Shin suit {s}", capsule(kn, an + [0, 0.07, 0], 0.074), "Suit", "LowerLeg_" + s, True)
    add(f"Knee pad {s}", ellipsoid(kn + [0, 0.0, 0.06], (0.06, 0.07, 0.035)), "Device", "LowerLeg_" + s)
    add(f"Boot shaft {s}", capsule(an + [0, 0.0, 0], an + [0, 0.13, 0], 0.08), "Glove", "Foot_" + s)
    add(f"Boot {s}", ellipsoid(an + [0, -0.035, 0.055], (0.068, 0.07, 0.14)), "Glove", "Foot_" + s)
    add(f"Boot sole {s}", box(an + [0, -0.095, 0.05], (0.13, 0.03, 0.29)), "Dark", "Foot_" + s)

# torso and hips
add("Torso suit", capsule(P["Spine"] + [0, 0.02, 0], [0, 1.36, 0], 0.175, 1.12, 0.82), "Suit", "Spine", True)
add("Abdomen suit", capsule([0, 0.9, 0], [0, 1.06, 0], 0.15, 1.08, 0.85), "Suit", "Spine", True)
add("Pelvis suit", capsule([-0.085, 0.88, 0], [0.085, 0.88, 0], 0.125), "Suit", "Hips", True)
add("Belt", torus([0, 0.98, 0.005], 0.168, 0.024, sz=0.8), "Strap", "Hips")
add("Belt buckle", box([0, 0.98, 0.145], (0.07, 0.05, 0.02)), "Metal", "Hips")
for k, x in enumerate((-0.14, 0.13)):
    add(f"Belt pouch {k}", box([x, 0.94, 0.11], (0.07, 0.08, 0.05), euler(y=-np.sign(x) * 0.5)), "Strap", "Hips")
# harness and chest control unit
for sg in (1, -1):
    add(f"Chest strap {'L' if sg > 0 else 'R'}", box([sg * 0.1, 1.3, 0.142], (0.045, 0.34, 0.02), euler(x=-0.12, z=sg * 0.08)), "Strap", "Spine")
add("Cross strap", box([0, 1.15, 0.148], (0.3, 0.04, 0.02)), "Strap", "Spine")
add("Chest unit", box([0.045, 1.27, 0.175], (0.16, 0.13, 0.065)), "Device", "Spine")
add("Chest unit panel", box([0.045, 1.28, 0.209], (0.12, 0.07, 0.005)), "Dark", "Spine")
add("Chest unit light", ellipsoid([0.1, 1.31, 0.21], (0.011, 0.011, 0.008)), "Red light", "Spine")
for k in range(3):
    add(f"Chest unit knob {k}", cylinder([0.0 + k * 0.035, 1.225, 0.212], 0.009, 0.012, euler(x=np.pi / 2)), "Metal", "Spine")
add("Chest hose", torus([0.11, 1.33, 0.12], 0.07, 0.012, euler(x=0.3, z=1.2)), "Dark", "Spine")
# life-support backpack
add("Backpack", box([0, 1.2, -0.25], (0.34, 0.48, 0.17)), "Pack", "Spine")
add("Backpack lid", box([0, 1.45, -0.25], (0.32, 0.05, 0.15)), "Device", "Spine")
add("Backpack roll", cylinder([0, 1.5, -0.26], 0.055, 0.36, euler(z=np.pi / 2)), "Pack", "Spine")
for sg in (1, -1):
    add(f"Air tank {'L' if sg > 0 else 'R'}", capsule([sg * 0.205, 0.98, -0.25], [sg * 0.205, 1.36, -0.25], 0.045), "Metal", "Spine")
    add(f"Tank valve {'L' if sg > 0 else 'R'}", cylinder([sg * 0.205, 1.42, -0.25], 0.018, 0.04), "Dark", "Spine")
add("Backpack pocket", box([0, 1.1, -0.345], (0.26, 0.2, 0.04)), "Strap", "Spine")
# collar and helmet
add("Collar ring", torus([0, 1.47, 0], 0.145, 0.03), "Metal", "Neck")
add("Neck seal", capsule([0, 1.44, 0], [0, 1.54, 0], 0.07), "Dark", "Neck")
add("Helmet ring", torus([0, 1.505, 0.0], 0.152, 0.022), "Metal", "Head")
add("Helmet glass", ellipsoid([0, 1.69, 0.012], (0.205, 0.215, 0.205), lat=(-1.0, np.pi / 2), rings=18, n=32), "Glass", "Head")
add("Helmet comms", box([0.17, 1.6, -0.03], (0.035, 0.07, 0.06)), "Device", "Head")
add("Helmet lamp", cylinder([0.17, 1.64, 0.03], 0.016, 0.03, euler(x=np.pi / 2)), "Metal", "Head")
# the alien
add("Alien head", ellipsoid([0, 1.68, 0.01], (0.094, 0.13, 0.112)), "Skin", "Head")
add("Alien cranium", ellipsoid([0, 1.755, -0.02], (0.075, 0.075, 0.1)), "Skin", "Head")
add("Alien jaw", ellipsoid([0, 1.6, 0.045], (0.058, 0.055, 0.07)), "Skin", "Head")
for sg, s in ((1, "L"), (-1, "R")):
    add(f"Cheekbone {s}", ellipsoid([sg * 0.06, 1.66, 0.07], (0.035, 0.022, 0.04)), "Skin", "Head")
    add(f"Brow ridge {s}", ellipsoid([sg * 0.04, 1.735, 0.085], (0.04, 0.016, 0.03), euler(z=sg * 0.25)), "Skin", "Head")
    add(f"Eye {s}", ellipsoid([sg * 0.046, 1.702, 0.09], (0.041, 0.025, 0.026), euler(z=sg * 0.42)), "Eye", "Eye_" + s)
    add(f"Eye glint {s}", ellipsoid([sg * 0.055, 1.713, 0.114], (0.006, 0.005, 0.002)), "Glint", "Eye_" + s)
    add(f"Nostril {s}", box([sg * 0.008, 1.645, 0.112], (0.004, 0.012, 0.004)), "Dark", "Head")
add("Mouth", box([0, 1.608, 0.105], (0.04, 0.003, 0.006)), "Dark", "Head")
# kit
add("Holster", box([-0.2, 0.7, 0.02], (0.06, 0.2, 0.1)), "Strap", "UpperLeg_R")
add("Pistol grip", box([-0.2, 0.83, 0.0], (0.035, 0.08, 0.05), euler(x=-0.3)), "Dark", "UpperLeg_R")
add("Thigh pouch", box([0.2, 0.7, 0.03], (0.055, 0.14, 0.09)), "Strap", "UpperLeg_L")
add("Wrist display", box([0.29, 0.98, 0.055], (0.045, 0.07, 0.012), euler(x=-0.1)), "Blue screen", "LowerArm_L")
add("Tablet", box([0.25, 0.8, 0.13], (0.2, 0.012, 0.14), euler(x=0.9, y=0.3)), "Device", "Hand_L")

# ---------------- skin weights ----------------
def nrm(v): return v / np.linalg.norm(v)
BOUNDS = [("Hips", "Spine", P["Spine"], np.array([0, 1.0, 0]), 0.32), ("Spine", "Neck", P["Neck"], np.array([0, 1.0, 0]), 0.16)]
for s in "LR":
    BOUNDS += [("Spine", "UpperArm_" + s, P["UpperArm_" + s], nrm(P["LowerArm_" + s] - P["UpperArm_" + s]), 0.16),
               ("UpperArm_" + s, "LowerArm_" + s, P["LowerArm_" + s], nrm(P["Hand_" + s] - P["LowerArm_" + s]), 0.12),
               ("Hips", "UpperLeg_" + s, P["UpperLeg_" + s], nrm(P["LowerLeg_" + s] - P["UpperLeg_" + s]), 0.14),
               ("UpperLeg_" + s, "LowerLeg_" + s, P["LowerLeg_" + s], nrm(P["Foot_" + s] - P["LowerLeg_" + s]), 0.12),
               ("LowerLeg_" + s, "Foot_" + s, P["Foot_" + s] + [0, 0.07, 0], np.array([0, -1.0, 0]), 0.12)]
D, CAP = 0.05, 0.85
BI = {b: i for i, b in enumerate(BONES)}

def weights(V, bone, soft):
    n = len(V); J = np.zeros((n, 4), np.uint8); W = np.zeros((n, 4), np.float32)
    J[:, 0] = BI[bone]; W[:, 0] = 1
    if soft:
        bw = np.zeros(n); bo = np.full(n, BI[bone])
        for pb, cb, pt, dr, rad in BOUNDS:
            if bone not in (pb, cb): continue
            rel = V - pt; t = rel @ dr; perp = np.linalg.norm(rel - np.outer(t, dr), axis=1)
            w = np.clip(0.5 - t / (2 * D), 0, CAP) if bone == cb else np.clip(0.5 + t / (2 * D), 0, CAP)
            w = np.where(perp < rad, w, 0); o = BI[pb] if bone == cb else BI[cb]
            take = w > bw; bw = np.where(take, w, bw); bo = np.where(take, o, bo)
        J[:, 1] = bo; W[:, 0] = 1 - bw; W[:, 1] = bw
    return J, W

def normals(V, T):
    N = np.zeros_like(V); f = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3): np.add.at(N, T[:, k], f)
    l = np.linalg.norm(N, axis=1, keepdims=True); l[l == 0] = 1
    return N / l

# ---------------- write GLB ----------------
def main(out):
    blob = bytearray(); views = []; accs = []
    def view(raw, target=None):
        while len(blob) % 4: blob.append(0)
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)}
        if target: v["target"] = target
        blob.extend(raw); views.append(v); return len(views) - 1
    def acc(arr, ctype, typ, target=None, minmax=False):
        a = {"bufferView": view(arr.tobytes(), target), "componentType": ctype, "count": len(arr), "type": typ}
        if minmax: a["min"] = arr.min(0).tolist(); a["max"] = arr.max(0).tolist()
        accs.append(a); return len(accs) - 1

    meshes = []; nodes = [{"name": "ALIEN SETTLER — Carax", "children": []}]
    for name, (V, UV, T), mat, bone, soft in parts:
        V = V.astype(np.float32); N = normals(V.astype(float), T).astype(np.float32)
        J, W = weights(V.astype(float), bone, soft)
        prim = {"attributes": {"POSITION": acc(V, 5126, "VEC3", 34962, True), "NORMAL": acc(N, 5126, "VEC3", 34962),
                               "TEXCOORD_0": acc(UV.astype(np.float32), 5126, "VEC2", 34962),
                               "JOINTS_0": acc(J, 5121, "VEC4", 34962), "WEIGHTS_0": acc(W, 5126, "VEC4", 34962)},
                "indices": acc(T.astype(np.uint16).reshape(-1), 5123, "SCALAR", 34963), "material": MI[mat]}
        meshes.append({"name": name, "primitives": [prim]})
        nodes.append({"name": name, "mesh": len(meshes) - 1, "skin": 0}); nodes[0]["children"].append(len(nodes) - 1)
    first = len(nodes)
    for b in BONES:
        p = PARENT[b]; t = P[b] - (P[p] if p else 0)
        nodes.append({"name": b, "translation": [round(float(x), 6) for x in t]})
    for b in BONES:
        if PARENT[b]: nodes[first + BI[PARENT[b]]].setdefault("children", []).append(first + BI[b])
    nodes[0]["children"].append(first + BI["Hips"])
    ibm = np.zeros((len(BONES), 16), np.float32)
    for k, b in enumerate(BONES):
        m = np.eye(4, dtype=np.float32); m[0:3, 3] = -P[b]; ibm[k] = m.T.reshape(-1)
    ibm_acc = acc(ibm, 5126, "MAT4")
    images = [{"bufferView": view(png), "mimeType": "image/png"} for png in TEX]
    gl = {"asset": {"version": "2.0", "generator": "NOSTRAVECTOR alien settler builder (rigged, %d bones)" % len(BONES)},
          "scene": 0, "scenes": [{"nodes": [0]}], "nodes": nodes, "meshes": meshes,
          "materials": [dict(name=n, **m) for n, m in MATS],
          "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
          "images": images, "textures": [{"sampler": 0, "source": i} for i in range(len(images))],
          "skins": [{"name": "Armature", "joints": [first + k for k in range(len(BONES))], "skeleton": first + BI["Hips"], "inverseBindMatrices": ibm_acc}],
          "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": 0}]}
    while len(blob) % 4: blob.append(0)
    gl["buffers"][0]["byteLength"] = len(blob)
    js = json.dumps(gl, separators=(",", ":")).encode()
    while len(js) % 4: js += b" "
    out_b = struct.pack("<4sII", b"glTF", 2, 28 + len(js) + len(blob)) + struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\x00") + bytes(blob)
    open(out, "wb").write(out_b)
    print(f"{len(parts)} parts, {sum(len(p[1][0]) for p in parts)} vertices, {len(out_b)//1024} KB")

if __name__ == "__main__":
    main(sys.argv[1])
