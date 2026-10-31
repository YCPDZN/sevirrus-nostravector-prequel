# SEVIRRUS

*A prequel to NOSTRAVECTOR.*

Aridaxx, 1790 AD. Hak Chompjak, heir to Emperor Jakal Chompjak XVI, stands one step below a throne he has already decided is his. On the mining world of Carax, at the rim of Sagittarius A*, the one called Sevirrus kills a settler leader and executes his people. His father swears to find the monster, without knowing who wears the mask.

A browser game built with [three.js](https://threejs.org). The whole game is one self-contained page: models, voices and music are embedded, and only three.js loads from a CDN.

## Play

**https://ycpdzn.github.io/sevirrus-nostravector-prequel/** (GitHub Pages: Settings → Pages → Deploy from a branch → `main` / root).

`index.html` is the game; it loads its models and voices from `assets/` when it starts, so it needs to be served over the web (GitHub Pages, or `python3 -m http.server` in this folder, then open http://localhost:8000). Only three.js comes from a CDN.

## What's in it

- **Main menu**: Sevirrus as a live, rigged 3D model; subject file (Chaotic Evil); a chapter box with pictures, where each chapter unlocks when the one before it is cleared; lore archive; Save / Load with three slots and Continue.
- **Intro cutscene** (Begin): a duel with the settlers' leader on Carax, scored to *Fields of Carax* and fully voiced, then the executions under Sagittarius A*.
- **Alphinos Castle**: Jakal Chompjak XVI's speech to Aridaxx, voiced, with lip sync.
- **Chapter I: Investigation** (playable):
  1. Examine six clues at the massacre site; a recording on the leader's wrist display reveals the killer's voice.
  2. Drive the royal rover five miles to the Bastion of Carax, to *Demons of Laniakea*.
  3. Fight up five floors of the bastion against the Cult of Zark.
- **Chapter II: The Siege of Aridaxx** (cutscene): a month later, Sevirrus breaks the gate of Alphinos Castle, beheads the Chompy guard, holds one head up for the rest to see, cuts down ten more and demands the head of Jakal Chompjak XVI. Damix Chompjak, Hak's cousin, watches in horror, then laughs, mocks him and throws him back down the steps again and again, until Sevirrus simply rolls his neck. Then, playable, as Damix: run the burning road to Alphinos Keep, cutting through the Cult of Zark, and seal the gates before Sevirrus reaches the Emperor. Sevirrus walks behind you the whole way and has no health bar: strike him, or let him catch you, and he stops your heart.
- **Chapter III: Assassination** (cutscene): the throne hall of Alphinos Keep. Sevirrus kills Damix, then his father, Emperor Jakal Chompjak XVI, and takes the crown. After it, the main menu belongs to the Emperor.

The game saves only when the player chooses to (pause menu, or at the end of a chapter). Saves are stored in the browser.

### Controls

| | Keyboard and mouse | Touch |
|---|---|---|
| Move / drive | WASD or arrow keys, Shift to run | on-screen stick |
| Look | drag | drag on the right side |
| Examine / enter / climb | E | Examine / Drive / Climb button |
| Strike | click or F | Strike button |
| Dodge | Space | Dodge button |
| Force blast (Damix) | Q | Blast button |
| Seal the gates (Damix) | hold E | hold Seal |
| Journal / Pause | J / Esc | buttons, top right |
| Skip dialogue line | Enter | |

## Repository

| Path | Contents |
|---|---|
| `index.html` | the game (website build) |
| `assets/` | the game's models (`.glb`) and voice lines and music (`.mp3`), loaded by `index.html` |
| `tools/build_site.py` | builds `index.html` and `assets/` from the single-file game (every asset embedded) |
| `models/` | original concept models (Sevirrus, Jakal, Damix Chompjak, Chompy soldier), rigged versions (18-bone armature: Hips, Spine, Neck, Head, Eye_L/R, UpperArm, LowerArm, Hand, UpperLeg, LowerLeg, Foot), and the alien settler |
| `music/fields_of_carax.mp3` | the duel score, synthesized from scratch |
| `tools/rig_glb.py` | adds the armature and skin weights to the character models |
| `tools/rig_hier.py` | rigs the nested exports (Damix Chompjak, Chompy soldier) and merges each into a few draw calls |
| `tools/build_alien.py` | builds and rigs the alien settler model |
| `tools/compose_duel.py` | renders *Fields of Carax* on the duel's beat grid |

The tools need Python 3 with `numpy`, `scipy` and `Pillow`.
