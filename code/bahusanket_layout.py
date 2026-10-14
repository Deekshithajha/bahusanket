"""Which MediaPipe points our poses keep, and where each body part sits in the array.

Two layouts exist (both lists of indices into the 543-point MediaPipe Holistic layout:
body 0-32, face 33-500, left hand 501-521, right hand 522-542):
  * v1, model_point_indices.json: 103 points = 7 body + 54 face + 42 hands.
    Face = jaw outline, both eyebrows, lower lip only.
  * v2, model_point_indices_v2.json: 123 points = 7 body + 74 face + 42 hands.
    Adds 20 face points: upper lip (7 outer, 4 inner), both eyes (corners plus
    upper and lower lid, 4 each) and the nose bridge, so mouth opening, blinks and
    eye direction become visible to the model.

Usage in a notebook, right after loading the index file:
    import bahusanket_layout as L
    LAY = L.apply(point_idx, B, V)   # sets B.N_POINTS, B.PART_SLICES (and V.IN_DIM)
Order inside each part is ascending MediaPipe index, as ISL.ipynb built it.
"""

# Face-mesh ids (0-467) of the 20 points v2 adds.
V2_NEW_FACE = [185, 40, 37, 0, 267, 270, 409,        # upper lip, outer edge
               81, 13, 311, 308,                     # upper lip, inner edge (308 = inner corner)
               33, 133, 159, 145,                    # right eye: corners, upper lid, lower lid
               362, 263, 386, 374,                   # left eye
               168]                                  # nose bridge, between the eyes

# Contours for drawing, as face-mesh ids. Points a layout lacks are skipped.
FACE_CONTOURS = {
    "jaw": [234, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 454],
    "brow_r": [70, 63, 105, 66, 107, 55, 65],
    "brow_l": [300, 293, 334, 296, 336, 285, 295],
    "lip_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 267, 0, 37, 40, 185, 61],
    "lip_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 311, 13, 81, 78],
    "eye_r": [33, 159, 133, 145, 33],
    "eye_l": [362, 386, 263, 374, 362],
}
FACE_OFF, LHAND_OFF, RHAND_OFF = 33, 501, 522


def layout(point_idx):
    """Describe a point list: n_points, part slices, and positions used for drawing."""
    idx = [int(i) for i in point_idx]
    pos = {m: k for k, m in enumerate(idx)}
    body = [k for k, m in enumerate(idx) if m < FACE_OFF]
    face = [k for k, m in enumerate(idx) if FACE_OFF <= m < LHAND_OFF]
    hands = [k for k, m in enumerate(idx) if m >= LHAND_OFF]
    for name, ks in (("body", body), ("face", face), ("hands", hands)):
        assert ks == list(range(ks[0], ks[-1] + 1)), f"{name} points are not contiguous"
    lines = {}
    for name, ids in FACE_CONTOURS.items():
        ks = [pos[FACE_OFF + m] for m in ids if FACE_OFF + m in pos]
        if len(ks) >= 2:
            lines[name] = ks
    return {
        "n_points": len(idx),
        "parts": {"body": slice(body[0], body[-1] + 1), "face": slice(face[0], face[-1] + 1),
                  "hands": slice(hands[0], hands[-1] + 1)},
        "body": {n: pos[m] for n, m in (("nose", 0), ("l_shoulder", 11), ("r_shoulder", 12), ("l_elbow", 13),
                                        ("r_elbow", 14), ("l_wrist", 15), ("r_wrist", 16)) if m in pos},
        "lhand": pos.get(LHAND_OFF), "rhand": pos.get(RHAND_OFF),
        "face_lines": lines,
        "chin": pos.get(FACE_OFF + 152),
    }


def columns_of(subset_idx, point_idx):
    """Positions of the points in subset_idx inside point_idx (e.g. the 103 v1 points inside v2),
    so scores can be compared on the same points across layouts."""
    pos = {int(m): k for k, m in enumerate(point_idx)}
    return [pos[int(m)] for m in subset_idx]


def apply(point_idx, *modules):
    """Point bahusanket_lib (and bahusanket_vq) at this layout. Returns layout(point_idx)."""
    lay = layout(point_idx)
    for mod in modules:
        if hasattr(mod, "N_POINTS"):
            mod.N_POINTS = lay["n_points"]
        if hasattr(mod, "PART_SLICES"):
            mod.PART_SLICES = dict(lay["parts"])
        if hasattr(mod, "IN_DIM"):
            mod.IN_DIM = lay["n_points"] * 2
    return lay
