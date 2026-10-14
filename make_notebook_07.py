"""Generates 07_face_points_v2.ipynb: rebuild training_ready/ with 123 points (20 more face points).
Run:  python make_notebook_07.py   (embeds the current code/ files, so regenerate after editing them)"""
import json, os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 07 · Add 20 face points (103 → 123)

**Why.** Our 54 face points covered only the jaw, eyebrows and the *lower* lip. With no upper lip and no eyes, a model
could never show the mouth opening (mouthing words) or blinks and eye direction, which are part of the non-manual
features (face and head signals) we promised.

**What changes.** 20 face points are added: upper lip (7 on the outer edge, 4 on the inner edge), both eyes
(two corners, upper lid and lower lid, 4 each) and the nose bridge. New layout: **123 points = 7 body + 74 face + 42 hands**,
saved as `model_point_indices_v2.json`. The old 103-point files are **not touched**, so notebook 06 and older results keep working.

**What this notebook does (no GPU needed, about 20-30 minutes):**
1. Saves the new point list into your Drive `BahuSanket` folder and makes a 2-line update to `code/bahusanket_vq.py`.
2. Reads the raw pose batches in `pose_data/` (they still hold all 543 MediaPipe points, so nothing is re-downloaded).
3. Writes `training_ready_v2/train.npz`, `val.npz`, `test.npz`: the **same clips, same order, same sentences** as `training_ready/`, with 123 points per frame.
4. Checks that the 103 old points come out **exactly** as before, so only the new points are new.
5. Draws one test clip with the new face, into `viz/`.

**How to run:** Runtime → Change runtime type → **CPU** is fine. Then Runtime → **Run all**, and paste the last cell's output back to Claude.
""")

code(r'''
# 1. Setup: mount Drive and find the BahuSanket folder
import os, sys, json, glob, time
if os.environ.get("BAHU_ROOT"):                     # (used only for Claude's local test run)
    ROOT = os.environ["BAHU_ROOT"]
else:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/BahuSanket"
    if not os.path.isdir(ROOT):
        hits = glob.glob("/content/drive/MyDrive/**/BahuSanket", recursive=True)
        assert hits, "Could not find the BahuSanket folder in your Drive"
        ROOT = hits[0]
print("Drive folder:", ROOT)
CODE_DIR = os.path.join(ROOT, "code")
os.makedirs(CODE_DIR, exist_ok=True)
''')

point_idx_v2 = json.load(open(os.path.join(HERE, "model_point_indices_v2.json")))
code("# 2. Save the new point list to Drive and check the code files\n"
     f"POINT_IDX_V2 = {json.dumps(point_idx_v2)}\n" + r'''
p = os.path.join(ROOT, "model_point_indices_v2.json")
if os.path.exists(p):
    assert json.load(open(p)) == POINT_IDX_V2, "model_point_indices_v2.json on Drive differs from this notebook's list"
else:
    json.dump(POINT_IDX_V2, open(p, "w"))
for name in ["bahusanket_layout.py", "bahusanket_viz.py"]:   # uploaded to Drive/BahuSanket/code by Claude
    assert os.path.exists(os.path.join(CODE_DIR, name)), f"code/{name} is missing from Drive: tell Claude"
# bahusanket_vq.py: let PoseVQVAE pick up the point count at creation time (a 2-line change)
vq_path = os.path.join(CODE_DIR, "bahusanket_vq.py")
src = open(vq_path).read()
if "in_dim = in_dim or IN_DIM" not in src:
    a = "def __init__(self, in_dim=IN_DIM, width=512,"
    b = "        enc = [nn.Conv1d(in_dim, width, 3, padding=1), nn.ReLU()]"
    assert src.count(a) == 1 and src.count(b) == 1, "bahusanket_vq.py on Drive looks different: tell Claude"
    src = src.replace(a, "def __init__(self, in_dim=None, width=512,").replace(
        b, "        in_dim = in_dim or IN_DIM         # read at call time, so a layout switch is picked up\n" + b)
    open(vq_path, "w").write(src)
    print("updated code/bahusanket_vq.py")
sys.path.insert(0, CODE_DIR)
import importlib, numpy as np
import bahusanket_layout as L; importlib.reload(L)
import bahusanket_viz as VZ; importlib.reload(VZ)
point_idx_v1 = json.load(open(os.path.join(ROOT, "model_point_indices.json")))
LAY = L.layout(POINT_IDX_V2)
V1_COLS = L.columns_of(point_idx_v1, POINT_IDX_V2)     # where the 103 old points sit among the 123
NEW_COLS = [k for k in range(len(POINT_IDX_V2)) if k not in set(V1_COLS)]
print(f"v1: {len(point_idx_v1)} points  ->  v2: {LAY['n_points']} points;",
      {k: (s.start, s.stop) for k, s in LAY["parts"].items()}, "| new columns:", len(NEW_COLS))
assert len(point_idx_v1) == 103 and LAY["n_points"] == 123 and len(NEW_COLS) == 20
''')

code(r'''
# 3. One pass over the raw batches: trim to 123 points and normalise, exactly as ISL.ipynb did
#    (x, y only; centre each frame on the shoulder midpoint; divide by the clip's median shoulder width).
MAX_FRAMES = 300
TR_OLD = os.path.join(ROOT, "training_ready")
TR_NEW = os.path.join(ROOT, "training_ready_v2"); os.makedirs(TR_NEW, exist_ok=True)
wanted = {}
for s in ["train", "val", "test"]:
    with np.load(os.path.join(TR_OLD, f"{s}.npz"), allow_pickle=True) as d:
        wanted[s] = [str(u) for u in d["uid"]]
need = {u for us in wanted.values() for u in us}
print("clips in training_ready:", {s: len(u) for s, u in wanted.items()})

LS, RS = POINT_IDX_V2.index(11), POINT_IDX_V2.index(12)
def normalize_and_trim(points):
    pts = points[:, 0, POINT_IDX_V2, :2].astype("float32")
    center = (pts[:, LS] + pts[:, RS]) / 2
    scale = np.median(np.linalg.norm(pts[:, LS] - pts[:, RS], axis=-1))
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    return ((pts - center[:, None, :]) / scale)[:MAX_FRAMES].astype("float16")

clips, t0 = {}, time.time()
batch_files = sorted(glob.glob(os.path.join(ROOT, "pose_data", "batch_*.npz")))
for i, bf in enumerate(batch_files):
    with np.load(bf, allow_pickle=True) as d:
        for u, p in zip(d["uids"], d["points"]):
            u = str(u)
            if u in need:
                clips[u] = normalize_and_trim(p)      # p as stored (may be a masked array), like ISL.ipynb
    print(f"  batch {i + 1}/{len(batch_files)}: {len(clips)} clips so far  [{time.time() - t0:.0f}s]")
missing = need - set(clips)
assert not missing, f"{len(missing)} clips not found in pose_data, e.g. {sorted(missing)[:3]}"
''')

code(r'''
# 4. Write training_ready_v2/{train,val,test}.npz (same clips, order, sentences, frame masks as training_ready/)
#    and check the 103 old points are identical to the old files.
report = {}
for s in ["train", "val", "test"]:
    with np.load(os.path.join(TR_OLD, f"{s}.npz"), allow_pickle=True) as d:
        old = {k: d[k] for k in d.files}
    N = len(wanted[s])
    pose = np.zeros((N, MAX_FRAMES, LAY["n_points"], 2), dtype="float16")
    for j, u in enumerate(wanted[s]):
        c = clips.pop(u)
        pose[j, :len(c)] = c
    assert (pose.any(axis=(2, 3)) <= old["mask"]).all(), f"{s}: pose data outside the frame mask"
    same, diff, found = True, 0.0, 0                     # compared clip by clip to keep memory low
    for j in range(N):
        a, b = pose[j][:, V1_COLS], old["pose"][j]
        same &= bool(np.array_equal(a, b, equal_nan=True))
        diff = max(diff, float(np.nanmax(np.abs(a.astype("float32") - b.astype("float32")))))
        found += int((old["mask"][j][:, None] & (np.abs(pose[j][:, NEW_COLS].astype("float32")).sum(-1) > 0)).sum())
    found = found / max(int(old["mask"].sum()) * len(NEW_COLS), 1)
    new = {k: v for k, v in old.items() if k != "pose"}
    new["pose"] = pose
    new["point_indices"] = np.array(POINT_IDX_V2)
    del old
    np.savez_compressed(os.path.join(TR_NEW, f"{s}.npz"), **new)
    mb = os.path.getsize(os.path.join(TR_NEW, f"{s}.npz")) / 1e6
    report[s] = dict(clips=N, shape=list(pose.shape), old_points_identical=bool(same), max_diff=diff,
                     new_points_found=round(found, 4), mb=round(mb))
    print(s, report[s])
    del pose, new
assert all(r["old_points_identical"] for r in report.values()), "the 103 old points changed: tell Claude"
json.dump({"point_indices": "model_point_indices_v2.json", "splits": report},
          open(os.path.join(TR_NEW, "build_report.json"), "w"), indent=2)
''')

code(r'''
# 5. Mouth opening check and a picture of one test clip with the new face points
with np.load(os.path.join(TR_NEW, "test.npz"), allow_pickle=True) as d:
    pose, mask, uids = d["pose"], d["mask"], [str(u) for u in d["uid"]]
up, lo = POINT_IDX_V2.index(33 + 13), POINT_IDX_V2.index(33 + 14)    # inner upper lip, inner lower lip
gap = np.linalg.norm(pose[:, :, up].astype("float32") - pose[:, :, lo].astype("float32"), axis=-1)[mask]
print(f"mouth opening over all test frames (shoulder widths): median {np.median(gap):.3f}, "
      f"90th pct {np.percentile(gap, 90):.3f}, max {gap.max():.3f}")
def lip_gap(i):
    t = int(mask[i].sum())
    return np.linalg.norm(pose[i, :t, up].astype("float32") - pose[i, :t, lo].astype("float32"), axis=-1)
k = int(np.argmax([lip_gap(i).std() for i in range(min(200, len(uids)))]))   # a clip with lots of mouthing
T_k = int(mask[k].sum())
os.makedirs(os.path.join(ROOT, "viz"), exist_ok=True)
out = os.path.join(ROOT, "viz", f"v2_test_{uids[k]}.gif")
VZ.save_gif([pose[k, :T_k].astype("float32")], out, POINT_IDX_V2, titles=["test clip, 123 points"], caption=uids[k])
print("saved", out)
''')

code(r'''
print("\n===== PASTE EVERYTHING BELOW THIS LINE BACK TO CLAUDE =====")
for s, r in report.items():
    print(f"{s}: {r['clips']} clips, shape {r['shape']}, old 103 points identical: {r['old_points_identical']} "
          f"(max diff {r['max_diff']:.2g}), new points found in {r['new_points_found']:.1%} of frames, {r['mb']} MB")
print(f"mouth opening median {np.median(gap):.3f}, 90th pct {np.percentile(gap, 90):.3f}, max {gap.max():.3f}")
print("files: model_point_indices_v2.json, training_ready_v2/*.npz, code/bahusanket_layout.py, code/bahusanket_viz.py, code/bahusanket_vq.py (2-line update)")
''')

nb["cells"] = C
nb["metadata"] = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, os.path.join(HERE, "07_face_points_v2.ipynb"))
print("ok", len(C), "cells")
