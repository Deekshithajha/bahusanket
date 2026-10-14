import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 02 · Text-to-pose baseline (Progressive Transformer) + evaluation

**What this notebook does**
1. Reloads everything from Drive (no reliance on earlier sessions).
2. Scores **Baseline 0** (MiniLM retrieval) on the test set with **DTW-MJE**, so the new model has a number to beat.
3. Trains **Baseline 1**, a Progressive-Transformer-style text→pose model ([Saunders et al. 2020](https://arxiv.org/abs/2004.14874)), saving a checkpoint to Drive **after every epoch**. If Colab disconnects, just run all cells again: it resumes from the last saved epoch.
4. Scores Baseline 1 on the test set and prints a short summary to paste back to Claude.

**How to run:** Runtime → Change runtime type → **T4 GPU**. Then Runtime → **Run all**. Allow the Drive mount when asked.

**DTW-MJE in one paragraph.** *MJE* (Mean Joint Error) is the average distance between each predicted point (a fingertip, an eyebrow point) and the same point in the real video, for one frame. *DTW* (Dynamic Time Warping) first lines up the two sequences in time, so signing that is a bit faster or slower but otherwise right is not punished. DTW-MJE is the average MJE along that best alignment. Our poses are scaled so the signer's shoulder width = 1, so **DTW-MJE 0.10 means "off by 10% of a shoulder width on average"**. Lower is better.
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
train = B.load_split(os.path.join(TR, "train.npz"))
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
## Part A · Score Baseline 0 (retrieval) and a "do-nothing" floor

* **Mean pose (floor):** always output the average training pose, standing still, for as long as the real clip. Any real model must beat this, otherwise it has learned nothing about signing.
* **Baseline 0 (retrieval):** embed the test sentence with all-MiniLM-L6-v2, find the most similar training sentence, return that real recorded signing.

We first check the retrieval index is wired correctly: searching with a training sentence should return that same sentence.
""")

code(r'''
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
emb, row2train = B.load_retrieval_index(os.path.join(ROOT, "retrieval_index.npz"), train)

# Sanity check: 200 training sentences should retrieve themselves (or an identical sentence)
chk = list(range(0, len(train["texts"]), max(1, len(train["texts"]) // 200)))[:200]
got, sim = B.retrieve(encoder, emb, row2train, [train["texts"][i] for i in chk])
self_hit = np.mean([train["texts"][g] == train["texts"][i] for g, i in zip(got, chk)])
print(f"self-retrieval: {self_hit:.1%} correct (should be close to 100%), mean similarity {sim.mean():.3f}")
assert self_hit > 0.9, "retrieval index does not match train.npz: stop and tell Claude"
''')

code(r'''
t0 = time.time()
mp = B.mean_pose(train)
floor_preds = [np.repeat(mp[None], len(p), 0) for p in test["poses"]]
floor_res, _ = B.score_predictions(floor_preds, test)

got, sim = B.retrieve(encoder, emb, row2train, test["texts"])
ret_preds = [train["poses"][g].astype(np.float32) for g in got]
ret_res, ret_rows = B.score_predictions(ret_preds, test)
real_motion = float(np.mean([B.motion(p) for p in test["poses"]]))
ret_res["motion"] = float(np.mean([B.motion(p) for p in ret_preds]))
ret_res["mean_similarity"] = float(sim.mean())

fmt = lambda r: "  ".join(f"{k} {r[k]:.4f}" for k in ["all", "body", "face", "hands"])
print(f"TEST DTW-MJE (lower is better, units = shoulder widths)   [{time.time()-t0:.0f}s]")
print(f"  mean pose (floor) : {fmt(floor_res)}")
print(f"  Baseline 0 (retr.): {fmt(ret_res)}")
print(f"  motion: real {real_motion:.4f}, retrieval {ret_res['motion']:.4f}")
for k in range(3):   # a few examples, to read by eye
    print(f"  e.g. {test['texts'][k]!r}\n       -> retrieved {train['texts'][got[k]]!r} (sim {sim[k]:.2f}), DTW-MJE {ret_rows[k]['all']:.3f}")
json.dump({"split": "test", "metric": "DTW-MJE (x,y; shoulder widths)", "n_clips": ret_res["n_clips"],
           "mean_pose_floor": floor_res, "baseline0_retrieval": ret_res, "real_motion": real_motion},
          open(os.path.join(RESULTS_DIR, "baseline0_scores.json"), "w"), indent=2)
print("saved", os.path.join(RESULTS_DIR, "baseline0_scores.json"))
''')

md("""
## Part B · Train Baseline 1 (Progressive Transformer)

The model reads the sentence (text encoder), then draws the pose one frame at a time (decoder). Each frame also carries a **counter** from 0 to 1 that says how far through the sentence it is; when the counter reaches 1 the model stops. That is how it decides the length of the signing by itself.

**Settings** are in `CONFIG` below. Leave them as they are for this first run. The loss-weight and smoothness settings are there for the ablations we will run next (hands/face weighting, velocity/acceleration penalty).

The run saves `last.pt` (every epoch), `best.pt` (best validation score so far) and `history.json` to `Drive/BahuSanket/checkpoints/<RUN_NAME>/`. Re-running resumes automatically.
""")

code(r'''
CONFIG = dict(
    RUN_NAME="pt_baseline_v1",
    text_encoder="scratch",       # "scratch" = as in the PT paper; "distilbert" = frozen pretrained encoder (later ablation)
    d_model=256, n_heads=4, n_enc=2, n_dec=2, ff=1024, dropout=0.1,
    epochs=40, batch_size=32, lr=3e-4, weight_decay=1e-4, warmup_steps=500, grad_clip=1.0,
    noise_std=0.03,               # noise added to decoder inputs while training (PT paper's augmentation)
    w_body=1.0, w_face=1.0, w_hands=1.0,   # ablation 1: per-part loss weights
    w_counter=1.0,
    w_vel=0.0, w_acc=0.0,         # ablation 2: smoothness penalties (0 = off for the baseline)
    eval_every=2, eval_clips=200, # every 2 epochs: generate 200 val clips and compute DTW-MJE
    seed=42,
)
if os.environ.get("BAHU_QUICK"):   # (Claude's local smoke test only)
    CONFIG.update(epochs=int(os.environ.get("BAHU_QUICK_EPOCHS", 2)), eval_every=1, eval_clips=8,
                  d_model=64, ff=128, warmup_steps=10, batch_size=8)
CKPT_DIR = os.path.join(ROOT, "checkpoints", CONFIG["RUN_NAME"]); os.makedirs(CKPT_DIR, exist_ok=True)
print(json.dumps(CONFIG, indent=1))
''')

code(r'''
from transformers import AutoTokenizer
torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
D = train["poses"][0].shape[-1]
train_ds, val_ds = B.PoseTextDataset(train, tok), B.PoseTextDataset(val, tok)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                                       collate_fn=B.collate, num_workers=2, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
                                     collate_fn=B.collate, num_workers=2)

model = B.ProgressiveTransformer(vocab_size=tok.vocab_size, pose_dim=B.N_POINTS * D,
    d_model=CONFIG["d_model"], n_heads=CONFIG["n_heads"], n_enc=CONFIG["n_enc"],
    n_dec=CONFIG["n_dec"], ff=CONFIG["ff"], dropout=CONFIG["dropout"],
    text_encoder=CONFIG["text_encoder"]).to(DEVICE)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
total_steps = CONFIG["epochs"] * len(train_dl)
def lr_at(step):   # linear warm-up, then cosine decay
    if step < CONFIG["warmup_steps"]:
        return (step + 1) / CONFIG["warmup_steps"]
    p = (step - CONFIG["warmup_steps"]) / max(1, total_steps - CONFIG["warmup_steps"])
    return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
use_amp = DEVICE == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
pw = B.point_weights(CONFIG["w_body"], CONFIG["w_face"], CONFIG["w_hands"]).to(DEVICE)
print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f} M trainable; "
      f"{len(train_dl)} steps per epoch")

# Resume if a checkpoint exists
start_epoch, best_dtw, history = 0, float("inf"), []
last = os.path.join(CKPT_DIR, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
    sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    start_epoch, best_dtw, history = ck["epoch"] + 1, ck["best_dtw"], ck["history"]
    print(f"RESUMING from epoch {start_epoch} (best val DTW-MJE so far {best_dtw:.4f})")
else:
    print("starting a fresh run")
''')

code(r'''
def to_dev(b):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

def run_epoch(dl, train_mode):
    model.train(train_mode)
    sums, n = {}, 0
    for b in dl:
        b = to_dev(b)
        with torch.set_grad_enabled(train_mode), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred, cnt = model(b["ids"], b["attn"], B.teacher_inputs(b, CONFIG["noise_std"] if train_mode else 0.0))
        losses = B.pt_loss(pred.float(), cnt.float(), b, pw, CONFIG["w_counter"], CONFIG["w_vel"], CONFIG["w_acc"])
        if train_mode:
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}

rng = np.random.RandomState(0)
val_eval_idx = sorted(rng.choice(len(val["poses"]), min(CONFIG["eval_clips"], len(val["poses"])), replace=False).tolist())

for epoch in range(start_epoch, CONFIG["epochs"]):
    t0 = time.time()
    tr = run_epoch(train_dl, True)
    va = run_epoch(val_dl, False)
    row = {"epoch": epoch, "train_loss": tr["total"], "train_pose": tr["pose"], "train_counter": tr["counter"],
           "val_loss": va["total"], "val_pose": va["pose"], "lr": sched.get_last_lr()[0], "secs": time.time() - t0}
    if (epoch + 1) % CONFIG["eval_every"] == 0 or epoch + 1 == CONFIG["epochs"]:
        preds = B.generate_for_split(model, val, tok, val_eval_idx, DEVICE)
        r, _ = B.score_predictions(preds, val, val_eval_idx)
        row.update(val_dtw=r["all"], val_dtw_hands=r["hands"], val_dtw_face=r["face"],
                   val_len_ratio=r["len_ratio"], val_motion=float(np.mean([B.motion(p) for p in preds])))
        if r["all"] < best_dtw:
            best_dtw = r["all"]
            torch.save({"model": model.state_dict(), "config": CONFIG, "epoch": epoch, "val_dtw": best_dtw},
                       os.path.join(CKPT_DIR, "best.pt"))
            row["saved_best"] = True
    history.append(row)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch, "best_dtw": best_dtw, "history": history,
                "config": CONFIG}, last + ".tmp")
    os.replace(last + ".tmp", last)          # write-then-rename so a disconnect never leaves a broken file
    json.dump(history, open(os.path.join(CKPT_DIR, "history.json"), "w"), indent=1)
    msg = (f"epoch {epoch+1:>3}/{CONFIG['epochs']} | train {tr['total']:.4f} (pose {tr['pose']:.4f}, counter {tr['counter']:.4f})"
           f" | val {va['total']:.4f} | {row['secs']:.0f}s")
    if "val_dtw" in row:
        msg += (f" | val DTW-MJE {row['val_dtw']:.4f} (hands {row['val_dtw_hands']:.4f}) len x{row['val_len_ratio']:.2f}"
                f" motion {row['val_motion']:.4f}" + ("  *best*" if row.get("saved_best") else ""))
    print(msg, flush=True)
print("training finished; checkpoints in", CKPT_DIR)
''')

md("""
## Part C · Score Baseline 1 on the test set and compare
""")

code(r'''
ck = torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model"])
print(f"using best checkpoint from epoch {ck['epoch']+1} (val DTW-MJE {ck['val_dtw']:.4f})")
t0 = time.time()
test_idx = list(range(len(test["poses"])))
pt_preds = B.generate_for_split(model, test, tok, test_idx, DEVICE)
pt_res, _ = B.score_predictions(pt_preds, test, test_idx)
pt_res["motion"] = float(np.mean([B.motion(p) for p in pt_preds]))
b0 = json.load(open(os.path.join(RESULTS_DIR, "baseline0_scores.json")))
summary = {"run": CONFIG["RUN_NAME"], "best_epoch": ck["epoch"] + 1, "test_n": len(test_idx),
           "test_dtw_mje": {"mean_pose_floor": b0["mean_pose_floor"], "baseline0_retrieval": b0["baseline0_retrieval"],
                            "baseline1_pt": pt_res},
           "real_motion": b0["real_motion"], "config": CONFIG}
json.dump(summary, open(os.path.join(RESULTS_DIR, f"{CONFIG['RUN_NAME']}_test_scores.json"), "w"), indent=2)
print(f"[{time.time()-t0:.0f}s] saved results/{CONFIG['RUN_NAME']}_test_scores.json")

print("\n===== PASTE EVERYTHING BELOW THIS LINE BACK TO CLAUDE =====")
for h in history:
    print(f"ep{h['epoch']+1} tr {h['train_loss']:.4f} va {h['val_loss']:.4f}"
          + (f" vdtw {h['val_dtw']:.4f} len {h['val_len_ratio']:.2f} mot {h['val_motion']:.4f}" if "val_dtw" in h else ""))
print("TEST DTW-MJE  all / body / face / hands / len_ratio / motion")
for name, r in summary["test_dtw_mje"].items():
    print(f"  {name:<20} " + " / ".join(f"{r.get(k, float('nan')):.4f}" for k in ["all", "body", "face", "hands", "len_ratio", "motion"]))
print(f"  real data motion {b0['real_motion']:.4f}")
''')

nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "02_text_to_pose_baseline.ipynb")
print("ok", len(C), "cells")
