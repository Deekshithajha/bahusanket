import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 04 · Fair scoring (no training)

This notebook trains nothing. It re-scores the models we already have, fairly, so we can choose the next experiment from trustworthy numbers.

**Two fixes over notebook 03**
1. **motion_ratio is fixed.** It is now *total movement of all predictions ÷ total movement of all real clips* (1.0 = moves as much as real signers, 0 = frozen). Notebook 03 averaged a ratio per clip, and a few real clips that barely move made that average explode (values like 232).
2. **Scoring at the real length.** DTW-MJE gets better when a prediction simply runs longer, so models of different lengths can't be compared fairly. Here every prediction is given **exactly the real clip's length** (the "real-length" table). This is the fair comparison of *what* the models sign. The "own-length" table keeps the model's own stopping choice, as in notebook 03.

**Models scored:** mean-pose floor, Baseline 0 (retrieval), `pt_baseline_v2` (best), `pt_vel_v1` (best epoch and last epoch).

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. Each model's scores are saved to Drive as soon as they are done; if Colab disconnects, Run all again and finished models are skipped.
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
RESULTS_DIR = os.path.join(ROOT, "results"); os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, CODE_DIR)
import importlib, bahusanket_lib as B; importlib.reload(B)
import numpy as np, torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU: training will be very slow)")
''')

code(r'''
# 2. Reload the project config files from Drive and check them
point_idx = json.load(open(os.path.join(ROOT, "model_point_indices.json")))
excluded = set(json.load(open(os.path.join(ROOT, "excluded_uids.json"))))
splits = json.load(open(os.path.join(ROOT, "splits.json")))
assert len(point_idx) == 103, len(point_idx)
print("model_point_indices.json:", len(point_idx), "points")
print("excluded_uids.json:", len(excluded), "uids")
if isinstance(splits, dict):
    for k, v in splits.items():
        print(f"splits.json[{k!r}]: {type(v).__name__} len={len(v) if hasattr(v, '__len__') else v}",
              (list(v)[:3] if hasattr(v, '__iter__') and not isinstance(v, str) else ""))
else:
    print("splits.json:", type(splits).__name__, len(splits))
''')

code(r'''
# 3. Load train / val / test (keys are detected automatically and printed: please check them)
TR = os.path.join(ROOT, "training_ready")
for s in ["train", "val", "test"]:
    B.describe_npz(os.path.join(TR, f"{s}.npz"))
train = B.load_split(os.path.join(TR, "train.npz"))   # drops page/unit-marker clips
val = B.load_split(os.path.join(TR, "val.npz"))
test = B.load_split(os.path.join(TR, "test.npz"))
for name, d in [("train", train), ("val", val), ("test", test)]:
    print(name); B.missing_report(d)

# Leak check: no video may appear in two splits; excluded uids must be gone.
if train["uids"]:
    vid = lambda d: {u.rsplit("-", 1)[0] for u in d["uids"]}
    vt, vv, vs = vid(train), vid(val), vid(test)
    print(f"videos: train {len(vt)}, val {len(vv)}, test {len(vs)}; "
          f"overlap train-val {len(vt & vv)}, train-test {len(vt & vs)}, val-test {len(vv & vs)}")
    assert not (vt & vs) and not (vt & vv) and not (vv & vs), "video leakage between splits!"
    bad = excluded & set(train["uids"] + val["uids"] + test["uids"])
    print("excluded uids still present:", len(bad))
print("clips:", len(train["poses"]), len(val["poses"]), len(test["poses"]),
      "total", len(train["poses"]) + len(val["poses"]) + len(test["poses"]))
''')

md("""
## Part A · Floor and Baseline 0 (retrieval)

For the real-length table, each retrieved clip is stretched or squeezed in time to the real clip's length.
""")

code(r'''
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
emb, row2train = B.load_retrieval_index(os.path.join(ROOT, "retrieval_index.npz"), train)
chk = list(range(0, len(train["texts"]), max(1, len(train["texts"]) // 200)))[:200]
got, sim = B.retrieve(encoder, emb, row2train, [train["texts"][i] for i in chk])
self_hit = np.mean([train["texts"][g] == train["texts"][i] for g, i in zip(got, chk)])
print(f"self-retrieval: {self_hit:.1%} correct (should be close to 100%)")
assert self_hit > 0.9, "retrieval index does not match train.npz: stop and tell Claude"

OUT = os.path.join(RESULTS_DIR, "fair_scores_04.json")
SCORES = json.load(open(OUT)) if os.path.exists(OUT) else {}     # resume: finished models are kept
def save():
    json.dump(SCORES, open(OUT + ".tmp", "w"), indent=2); os.replace(OUT + ".tmp", OUT)

COLS = ["all", "body", "face", "hands", "len_ratio", "motion_ratio"]
def show(name, r):
    print(f"  {name:<28}" + " / ".join(f"{r.get(k, float('nan')):.4f}" for k in COLS)
          + (f"   [stop {r['stop_at']}]" if "stop_at" in r else ""))

test_idx = list(range(len(test["poses"])))
t0 = time.time()
if "floor" not in SCORES:
    mp = B.mean_pose(train)
    r, _ = B.score_predictions([np.repeat(mp[None], len(p), 0) for p in test["poses"]], test)
    SCORES["floor"] = {"real_length": r, "own_length": r}; save()
if "retrieval" not in SCORES:
    got, sim = B.retrieve(encoder, emb, row2train, test["texts"])
    ret = [train["poses"][g].astype(np.float32) for g in got]
    own, _ = B.score_predictions(ret, test)
    real, _ = B.score_predictions([B.resample_to_length(p, len(test["poses"][i])) for p, i in zip(ret, test_idx)], test)
    SCORES["retrieval"] = {"real_length": real, "own_length": own}; save()
print(f"[{time.time()-t0:.0f}s] TEST, real length:  all / body / face / hands / len_ratio / motion_ratio")
for k in ["floor", "retrieval"]: show(k, SCORES[k]["real_length"])
''')

md("""
## Part B · Re-score the trained models

For each checkpoint: generate 300 frames for 400 validation clips and every test clip, then score (1) at the real length and (2) at the model's own length, with the stopping cut-off chosen on the validation clips as in notebook 03. Roughly 10 minutes per model on a T4.
""")

code(r'''
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
D = train["poses"][0].shape[-1]
CAL_IDX = sorted(np.random.RandomState(0).choice(len(val["poses"]), min(400, len(val["poses"])), replace=False).tolist())
CK = os.path.join(ROOT, "checkpoints")
MODELS = {"pt_baseline_v2": os.path.join(CK, "pt_baseline_v2", "best.pt"),
          "pt_vel_v1 (best)": os.path.join(CK, "pt_vel_v1", "best.pt"),
          "pt_vel_v1 (last)": os.path.join(CK, "pt_vel_v1", "last.pt")}

for name, path in MODELS.items():
    if name in SCORES:
        print(f"{name}: already scored, skipping"); continue
    if not os.path.exists(path):
        print(f"{name}: no checkpoint at {path}, skipping"); continue
    t0 = time.time()
    model, ck = B.load_model(path, tok.vocab_size, B.N_POINTS * D, DEVICE)
    # stop_all_at=2: never stop early, so every clip has 300 frames to cut from
    vf, vc = B.generate_for_split(model, val, tok, CAL_IDX, DEVICE, batch_size=64, stop_all_at=2.0)
    thr, table = B.calibrate_stop(vf, vc, val, CAL_IDX)
    tf, tc = B.generate_for_split(model, test, tok, test_idx, DEVICE, batch_size=64, stop_all_at=2.0)
    real, _ = B.score_predictions(B.cut_to_real_length(tf, test, test_idx), test, test_idx)
    own, _ = B.score_predictions(B.apply_stop(tf, tc, thr), test, test_idx)
    own["stop_at"] = thr
    SCORES[name] = {"real_length": real, "own_length": own, "epoch": ck["epoch"] + 1}; save()
    del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    print(f"{name} (epoch {ck['epoch']+1}) scored in {time.time()-t0:.0f}s")
    show("  real length", real); show("  own length", own)
''')

md("""
## Part C · Comparison table
""")

code(r'''
print("===== PASTE EVERYTHING BELOW THIS LINE BACK TO CLAUDE =====")
print(f"test clips: {len(test_idx)}")
for which, title in [("real_length", "REAL LENGTH (fair comparison of what is signed)"),
                     ("own_length", "OWN LENGTH (model decides when to stop)")]:
    print(f"\n{title}\n  {'model':<28}all / body / face / hands / len_ratio / motion_ratio")
    for name, s in SCORES.items():
        show(name + (f" ep{s['epoch']}" if "epoch" in s else ""), s[which])
print("\nsaved", OUT)
''')

nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "04_fair_scoring.ipynb")
print("ok", len(C), "cells")
