"""'Fields of Carax' — original battle cue for the Sevirrus duel.

140 BPM, D minor, 18 bars (30.86 s) plus a reverb tail. Every hit is placed on the same
beat grid as the duel choreography, so clashes, the knock-back, the leap, the impale and the
rip land on the music. Rendered entirely from synthesis (no samples).

usage: python3 compose_duel.py out.wav
"""
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 44100
BPM = 140
BEAT = 60 / BPM
BAR = BEAT * 4
LEN = 18 * BAR + 2.5
N = int(LEN * SR)
rng = np.random.default_rng(1790)

def bar(b, beat=0.0): return (b - 1) * BAR + beat * BEAT          # bar numbers start at 1
def hz(m): return 440 * 2 ** ((m - 69) / 12)

# --- event times shared with the duel choreography (seconds from the start of the duel) ---
CHARGE = bar(3); CLASH1 = bar(4); SHOVE = bar(4, 3)
PARRIES = [bar(5, k) for k in (1, 2, 3)] + [bar(6, k) for k in (0, 1, 2)]   # 7.29 .. 9.43
COUNTER = bar(6, 3); HEAVY = bar(7, 2); RISE = bar(8); DASH = bar(8, 3)
FLURRY = [bar(9) + k * BEAT / 2 for k in range(7)]                           # 13.71 .. 15.0
LEGCUT = bar(9, 4); DUCK = bar(10, 1.5); LEAP = bar(11, 1); CATCH = bar(11, 3)
DISARM = bar(12, 2); STICK = bar(13); CATCH_WRIST = bar(14, 1); IMPALE = bar(14, 2)
RIP = bar(16); FLICK = bar(17, 2); BLACK = bar(18, 1.4)

tracks = {k: np.zeros((N, 2)) for k in ("drums", "strings", "brass", "choir", "fx", "bass")}

def put(track, sig, t, pan=0.0, gain=1.0):
    i = int(t * SR)
    if i >= N: return
    sig = sig[: N - i] * gain
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    tracks[track][i:i + len(sig), 0] += sig * l * 1.414
    tracks[track][i:i + len(sig), 1] += sig * r * 1.414

def env(n, a=0.005, d=None, rel=0.05):
    t = np.arange(n) / SR; e = np.minimum(1, t / max(a, 1e-4))
    if d: e *= np.exp(-t / d)
    r = int(rel * SR)
    if r and n > r: e[-r:] *= np.linspace(1, 0, r)
    return e

def saw(f, dur, detune=(-7, 0, 7), vib=0.0):
    t = np.arange(int(dur * SR)) / SR; out = np.zeros_like(t)
    for c in detune:
        ff = f * 2 ** (c / 1200) * (1 + vib * 0.006 * np.sin(2 * np.pi * 5.2 * t + rng.random() * 6))
        ph = np.cumsum(ff) / SR + rng.random()
        out += 2 * (ph % 1) - 1
    return out / len(detune)

# ---------------- instruments ----------------
def taiko(t, amp=1.0, pitch=1.0, pan=0.0):
    n = int(0.9 * SR); tt = np.arange(n) / SR
    f = (95 * np.exp(-tt * 18) + 48) * pitch
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 5.5)
    skin = sosfilt(butter(2, 900, 'low', fs=SR, output='sos'), rng.normal(0, 1, n)) * np.exp(-tt * 40) * 0.6
    put("drums", (body + skin) * amp, t, pan)

def boom(t, amp=1.0):
    n = int(3.0 * SR); tt = np.arange(n) / SR
    sub = np.sin(2 * np.pi * np.cumsum(55 * np.exp(-tt * 3) + 28) / SR) * np.exp(-tt * 1.4)
    crack = sosfilt(butter(2, [200, 4000], 'band', fs=SR, output='sos'), rng.normal(0, 1, n)) * np.exp(-tt * 9)
    put("fx", (sub * 1.3 + crack * 0.35) * amp, t)
    taiko(t, amp * 1.2, 0.8)

def cymbal(t, dur=1.6, amp=0.5, swell=False):
    n = int(dur * SR); tt = np.arange(n) / SR
    s = sosfilt(butter(2, 4500, 'high', fs=SR, output='sos'), rng.normal(0, 1, n))
    e = (tt / dur) ** 2.5 if swell else np.exp(-tt * 2.2)
    put("fx", s * e * amp * 0.4, t - (dur if swell else 0), 0.25)
    put("fx", np.roll(s, 997) * e * amp * 0.35, t - (dur if swell else 0), -0.25)

def string_note(t, m, dur, amp=0.3, pan=0.0):
    s = saw(hz(m), dur) * env(int(dur * SR), 0.004, 0.18, 0.02)
    put("strings", s * amp, t, pan)

def pad_strings(t, notes, dur, amp=0.12):
    for m in notes:
        s = saw(hz(m), dur, (-9, -3, 3, 9), vib=1.0) * env(int(dur * SR), 0.35, None, 0.6)
        put("strings", s * amp, t, rng.uniform(-0.5, 0.5))

def brass(t, notes, dur=0.35, amp=0.35, swell=False):
    for m in notes:
        n = int(dur * SR); s = saw(hz(m), dur, (-6, 0, 6))
        e = env(n, 0.25 if swell else 0.012, None if swell else dur * 0.6, 0.08)
        put("brass", np.tanh(s * 1.8) * e * amp, t, rng.uniform(-0.3, 0.3))

def choir(t, notes, dur, amp=0.16):
    for m in notes:
        n = int(dur * SR); s = saw(hz(m), dur, (-11, -4, 4, 11), vib=1.4)
        put("choir", s * env(n, 0.5, None, 0.8) * amp, t, rng.uniform(-0.6, 0.6))

def bass(t, m, dur, amp=0.4):
    n = int(dur * SR); tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * hz(m) * tt) + 0.4 * saw(hz(m), dur, (0,))
    put("bass", s * env(n, 0.02, None, 0.3) * amp, t)

# ---------------- harmony (MIDI roots per bar) ----------------
D2, F2, G2, A2, Bb1, C2, Eb2 = 38, 41, 43, 45, 34, 36, 39
CH = {"Dm": (D2, [62, 65, 69]), "Bb": (Bb1 + 12, [58, 62, 65]), "C": (C2 + 12, [60, 64, 67]), "A": (A2, [61, 64, 69]),
      "Gm": (G2, [62, 67, 70]), "Eb": (Eb2, [58, 63, 67])}
HARM = {1: "Dm", 2: "Dm", 3: "Bb", 4: "Dm", 5: "Bb", 6: "C", 7: "A", 8: "Dm", 9: "Bb", 10: "C", 11: "Dm",
        12: "Gm", 13: "Eb", 14: "A", 15: "Dm", 16: "Bb", 17: "Dm", 18: "Dm"}

# ---------------- the score ----------------
# bars 1-2: the black hole. Low drone, choir rising, a taiko roll into the charge
for b in (1, 2):
    root, tri = CH[HARM[b]]
    bass(bar(b), root, BAR + 0.1, 0.35)
choir(0, [50, 57, 62, 65], BAR * 2 + 0.3, 0.13)
pad_strings(0, [38, 45, 50], BAR * 2, 0.1)
for k in range(12): taiko(bar(2, 1) + k * BEAT / 4, 0.25 + k * 0.05, 1.1, (-1) ** k * 0.3)
cymbal(CHARGE, 1.6, 0.35, swell=True)

# bar 3: the charge — everything hits
boom(CHARGE, 0.9); cymbal(CHARGE, 2.0, 0.5)
brass(CHARGE, [46, 53, 58, 62, 65], BAR * 0.9, 0.3, swell=True)
for k in range(8): taiko(bar(3) + k * BEAT / 2, 0.55 if k % 2 == 0 else 0.35, 1.0 if k % 2 == 0 else 1.25, (-1) ** k * 0.4)
cymbal(CLASH1, 1.0, 0.4, swell=True)

# driving section bars 4-7 and 9-11: sixteenth-note string ostinato, taiko groove, stabs on every clash
OST = [0, 0, 12, 0, 0, 7, 0, 12, 0, 0, 12, 0, 3, 0, 7, 0]
GROOVE = [1.0, 0, 0.5, 0, 0.8, 0.4, 0.5, 0.3]
for b in (4, 5, 6, 7, 9, 10, 11):
    root, tri = CH[HARM[b]]
    for k in range(16):
        string_note(bar(b) + k * BEAT / 4, root + 12 + OST[k], BEAT / 4 + 0.02, 0.22 if k % 4 == 0 else 0.15, (-0.35, 0.35)[k % 2])
    for k in range(8):
        if GROOVE[k]: taiko(bar(b) + k * BEAT / 2, 0.6 * GROOVE[k], 1.0 if k % 4 == 0 else 1.3, (-1) ** k * 0.35)
    bass(bar(b), root, BAR, 0.4)
    pad_strings(bar(b), [m - 12 for m in tri], BAR, 0.07)
    choir(bar(b), [m for m in tri], BAR + 0.2, 0.06)

boom(CLASH1, 1.0); cymbal(CLASH1, 2.2, 0.5)
brass(CLASH1, [50, 57, 62, 65, 69], BAR * 0.8, 0.28, swell=False)
brass(SHOVE, [50, 57, 62], 0.3, 0.25)
for i, t in enumerate(PARRIES):                     # the leader's six blows, each parried
    tri = CH[HARM[5 if t < bar(6) else 6]][1]
    brass(t, [m - 12 for m in tri] + [tri[0]], 0.26, 0.26)
    taiko(t, 0.5, 1.4, (-1) ** i * 0.5)
brass(COUNTER, [57, 61, 64, 69], 0.5, 0.32)
cymbal(COUNTER, 1.2, 0.35)

# bar 7: the heavy blow and the knock-back
cymbal(HEAVY, 1.4, 0.45, swell=True)
boom(HEAVY, 1.2); cymbal(HEAVY, 2.6, 0.55)
brass(HEAVY, [45, 52, 57, 61, 64], BEAT * 2, 0.34)
choir(HEAVY, [57, 61, 64, 69], BEAT * 2.2, 0.14)

# bar 8: he rises, grinning — the floor drops out to choir and a pulse
choir(RISE, [50, 57, 62, 65, 69], BAR, 0.15)
bass(RISE, D2, BAR, 0.45)
for k in range(4): taiko(RISE + k * BEAT, 0.35, 0.85)
for k in range(8): taiko(DASH - BEAT + k * BEAT / 8, 0.2 + k * 0.06, 1.3, (-1) ** k * 0.4)
cymbal(bar(9), 0.9, 0.4, swell=True)

# bars 9-10: the flurry — stabs on every eighth note
boom(bar(9), 0.8)
for i, t in enumerate(FLURRY):
    brass(t, [58, 62, 65] if i % 2 == 0 else [57, 60, 65], 0.18, 0.22)
brass(LEGCUT, [48, 55, 60, 64], 0.6, 0.3); cymbal(LEGCUT, 1.4, 0.4)
brass(DUCK, [55, 60, 64], 0.25, 0.25)

# bar 11: the leap and the catch
cymbal(CATCH, 1.3, 0.55, swell=True)
for k in range(8): taiko(LEAP + k * BEAT / 4, 0.3 + k * 0.07, 1.2, (-1) ** k * 0.3)
boom(CATCH, 1.3); cymbal(CATCH, 3.0, 0.6)
brass(CATCH, [38, 50, 57, 62, 65, 69], BEAT * 3, 0.32)
choir(CATCH, [62, 65, 69, 74], BEAT * 3, 0.16)

# bar 12: the disarm
boom(DISARM, 0.9); brass(DISARM, [43, 50, 55, 58, 62], BEAT * 2, 0.3); cymbal(DISARM, 2.0, 0.45)
for k in range(4): taiko(bar(12) + k * BEAT, 0.45, 1.0)

# bars 13-14: dread. He walks forward; a slow climb, then silence before the blade goes in
choir(STICK, [51, 58, 63, 67], BAR + BEAT, 0.14)
bass(STICK, Eb2, BAR, 0.45)
pad_strings(STICK, [39, 46, 51], BAR, 0.12)
for k in range(4): taiko(STICK + k * BEAT, 0.3 + k * 0.08, 0.8)
rise_n = int((IMPALE - 0.25 - bar(14)) * SR)
riser = saw(hz(57), rise_n / SR, (-15, 0, 15)); riser = riser * np.linspace(0, 1, len(riser)) ** 2
put("strings", riser * 0.25, bar(14))
cymbal(IMPALE - 0.25, 1.5, 0.5, swell=True)
# (0.25 s of silence before the impale)

# bar 14 beat 3: the impale
boom(IMPALE, 1.5); cymbal(IMPALE, 3.5, 0.6)
brass(IMPALE, [38, 50, 57, 62, 65, 69, 74], BEAT * 2.5, 0.34)
choir(IMPALE, [62, 65, 69, 74, 77], BAR * 2.2, 0.17)

# bar 15: lament while he hangs on the blade
pad_strings(bar(15), [38, 50, 57, 62], BAR, 0.12)
bass(bar(15), D2, BAR, 0.4)
for k in range(4): taiko(bar(15) + k * BEAT, 0.3, 0.75)

# bar 16: the rip
cymbal(RIP, 0.9, 0.5, swell=True)
boom(RIP, 1.4); cymbal(RIP, 3.0, 0.6)
brass(RIP, [34, 46, 53, 58, 62, 65], BAR, 0.34)
choir(RIP, [58, 62, 65, 70], BAR, 0.16)
for k in range(8): taiko(RIP + k * BEAT / 2, 0.45 * (1 - k / 10), 1.0, (-1) ** k * 0.35)

# bars 17-18: he stands over the body. A final minor chord that will not resolve
choir(bar(17), [50, 57, 62, 65, 69], BAR * 2 + 1.0, 0.16)
pad_strings(bar(17), [38, 45, 50, 53], BAR * 2 + 1.0, 0.11)
bass(bar(17), D2, BAR * 2, 0.45)
brass(FLICK, [50, 57, 62], 0.4, 0.22)
boom(BLACK, 1.1); cymbal(BLACK, 2.5, 0.4)

# ---------------- mix ----------------
def lp(x, f): return sosfilt(butter(2, f, 'low', fs=SR, output='sos'), x, axis=0)
def hp(x, f): return sosfilt(butter(2, f, 'high', fs=SR, output='sos'), x, axis=0)
mix = (lp(tracks["strings"], 3200) * 0.9 + lp(tracks["brass"], 2600) * 0.85 +
       lp(hp(tracks["choir"], 180), 3400) * 1.0 + tracks["drums"] * 1.0 + tracks["fx"] * 0.9 + lp(tracks["bass"], 400) * 0.9)
# ---------------- dialogue holds ----------------
# The fight pauses for the spoken lines. At each hold (fight time, bars) the score sustains the
# current chord with a low pulse, then picks up exactly where it left off.
HOLDS = [(bar(2, 2), 4), (CLASH1 + 0.36, 2), (PARRIES[5] + 0.15, 1), (COUNTER + 0.35, 2), (RISE + 0.5, 3),
         (LEGCUT + 0.3, 2), (STICK + 0.3, 2), (CATCH_WRIST + 0.2, 1), (IMPALE + 1.0, 2), (RIP + 1.0, 1), (FLICK + 0.4, 1)]
def hold_segment(hf, nbars):
    global tracks, N
    saved, savedN = tracks, N
    dur = nbars * BAR; N = int(dur * SR); tracks = {k: np.zeros((N, 2)) for k in saved}
    b = min(18, int(hf // BAR) + 1); root, tri = CH[HARM[b]]
    bass(0, root, dur, 0.38); pad_strings(0, [m - 12 for m in tri], dur, 0.09); choir(0, tri, dur, 0.08)
    for k in range(int(nbars * 4)):
        taiko(k * BEAT, 0.28 if k % 4 == 0 else 0.16, 0.8, 0)
    seg = (lp(tracks["strings"], 3200) * 0.9 + lp(tracks["brass"], 2600) * 0.85 + lp(hp(tracks["choir"], 180), 3400) +
           tracks["drums"] + tracks["fx"] * 0.9 + lp(tracks["bass"], 400) * 0.9)
    tracks, N = saved, savedN
    return seg
pieces, last, XF = [], 0, int(0.03 * SR)
for hf, nb in HOLDS:
    i = int(hf * SR); cut = mix[last:i].copy(); cut[-XF:] *= np.linspace(1, 0, XF)[:, None]
    seg = hold_segment(hf, nb); seg[:XF] *= np.linspace(0, 1, XF)[:, None]; seg[-XF:] *= np.linspace(1, 0, XF)[:, None]
    pieces += [cut, seg]; last = i
tail = mix[last:].copy(); tail[:XF] *= np.linspace(0, 1, XF)[:, None]; pieces.append(tail)
mix = np.concatenate(pieces); N = len(mix)
tracks["bass"] = np.zeros_like(mix)

# hall reverb: decorrelated exponential noise tails
ir_n = int(2.6 * SR); tt = np.arange(ir_n) / SR
ir = np.stack([rng.normal(0, 1, ir_n) * np.exp(-tt * 2.4), rng.normal(0, 1, ir_n) * np.exp(-tt * 2.4)], 1)
ir[:, 0] = lp(ir[:, 0], 5000); ir[:, 1] = lp(ir[:, 1], 5000)
ir /= np.abs(ir).sum(0) ** 0.5 * 30
send = hp(mix - tracks["bass"] * 0.9, 250)
wet = np.stack([fftconvolve(send[:, c], ir[:, c])[:N] for c in range(2)], 1)
out = mix + wet * 0.55
out = np.tanh(out / np.abs(out).max() * 1.6) * 0.92            # glue + soft clip
fade = int(0.8 * SR); out[-fade:] *= np.linspace(1, 0, fade)[:, None]

import wave
pcm = (np.clip(out, -1, 1) * 32767).astype("<i2")
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print("rendered %.2f s with %d dialogue holds" % (N / SR, len(HOLDS)))
