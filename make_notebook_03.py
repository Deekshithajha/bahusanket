import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 03 · Fair re-scoring + velocity-loss run

This replaces notebook 02 (02 no longer works with the updated code). It does four things:

1. **Fixes the missing-point bug.** In run v2, points the pose tracker never found (mostly hands, stored as 0,0) were wrongly counted as real points, in training and in scoring. They are now ignored again. It also **removes the 87 clips whose sentence is only "Page 54." / "Unit 3."**
2. **Re-scores Baseline 0 (retrieval) and the mean-pose floor** with the fixed scoring. (Part A)
3. **Re-scores the existing model `pt_baseline_v2`** (no retraining) with the fixed scoring and a better rule for *when to stop signing*. (Part B)
4. **Trains `pt_vel_v1`**: the same model with a **velocity loss**, which pushes predicted movement between frames to match the real movement, so the model is discouraged from standing still. Then it scores everything side by side. (Parts C and D)

**Two numbers per model, read together**
* **DTW-MJE** (lower is better): average distance to the real pose after lining the two up in time, in shoulder widths. It rewards standing still, which is why we need the second number.
* **motion_ratio** (closer to 1.0 is better): how much the prediction moves compared with the real clip. 1.0 = moves as much as the real signer; 0.0 = frozen statue.

**Stopping rule.** The model counts from 0 to 1 while it signs. Instead of stopping at a fixed 0.97, we now try several cut-off values on the **validation** set and pick the one whose clip lengths best match the real lengths. The test set is never used to choose it.

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. If Colab disconnects during training, Run all again: it resumes from the last saved epoch. Part B is quick to repeat.
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
## Part A · Re-score the mean-pose floor and Baseline 0 (retrieval)
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

t0 = time.time()
test_idx = list(range(len(test["poses"])))
mp = B.mean_pose(train)
floor_res, _ = B.score_predictions([np.repeat(mp[None], len(p), 0) for p in test["poses"]], test)
got, sim = B.retrieve(encoder, emb, row2train, test["texts"])
ret_res, _ = B.score_predictions([train["poses"][g].astype(np.float32) for g in got], test)
real_motion = float(np.mean([B.motion(test["poses"][i], B.joint_mask(test, i)) for i in test_idx]))
RESULTS = {"mean_pose_floor": floor_res, "baseline0_retrieval": ret_res}
COLS = ["all", "body", "face", "hands", "len_ratio", "motion_ratio"]
def show(name, r):
    print(f"  {name:<30}" + " / ".join(f"{r.get(k, float('nan')):.4f}" for k in COLS))
print(f"TEST  all / body / face / hands / len_ratio / motion_ratio   [{time.time()-t0:.0f}s]")
for k, r in RESULTS.items(): show(k, r)
json.dump({"split": "test", "n_clips": len(test_idx), "real_motion": real_motion, **RESULTS},
          open(os.path.join(RESULTS_DIR, "baseline0_scores_v2.json"), "w"), indent=2)
print("saved results/baseline0_scores_v2.json")
''')

md("""
## Part B · Re-score `pt_baseline_v2` (no retraining)

Loads `checkpoints/pt_baseline_v2/best.pt`, generates validation clips to choose the stopping cut-off, then scores the test set. The old fixed cut-off (0.97) is shown too, for comparison.
""")

code(r'''
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
D = train["poses"][0].shape[-1]
rng = np.random.RandomState(0)
CAL_IDX = sorted(rng.choice(len(val["poses"]), min(400, len(val["poses"])), replace=False).tolist())

def score_run(model, name, gen_bs=64):
    """Choose the stop cut-off on validation clips, then score the test set with it."""
    t0 = time.time()
    vf, vc = B.generate_for_split(model, val, tok, CAL_IDX, DEVICE, batch_size=gen_bs)
    thr, table = B.calibrate_stop(vf, vc, val, CAL_IDX)
    print(f"[{name}] stop cut-off chosen on {len(CAL_IDX)} val clips: {thr}  "
          f"(median generated/real length per cut-off: " + ", ".join(f"{k}:{v:.2f}" for k, v in table.items()) + ")")
    tf, tc = B.generate_for_split(model, test, tok, test_idx, DEVICE, batch_size=gen_bs)
    res, _ = B.score_predictions(B.apply_stop(tf, tc, thr), test, test_idx)
    res_old, _ = B.score_predictions(B.apply_stop(tf, tc, 0.97), test, test_idx)
    res["stop_at"], res_old["stop_at"] = thr, 0.97
    print(f"[{name}] scored {len(test_idx)} test clips in {time.time()-t0:.0f}s")
    return res, res_old, table

V2_CKPT = os.path.join(ROOT, "checkpoints", "pt_baseline_v2", "best.pt")
if os.path.exists(V2_CKPT):
    v2_model, v2_ck = B.load_model(V2_CKPT, tok.vocab_size, B.N_POINTS * D, DEVICE)
    print(f"pt_baseline_v2: best epoch {v2_ck['epoch']+1}")
    r, r_old, _ = score_run(v2_model, "pt_baseline_v2")
    RESULTS["pt_baseline_v2"], RESULTS["pt_baseline_v2 (stop 0.97)"] = r, r_old
    del v2_model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    for k in ["pt_baseline_v2", "pt_baseline_v2 (stop 0.97)"]: show(k, RESULTS[k])
else:
    print("no pt_baseline_v2 checkpoint found at", V2_CKPT, "- skipping")
''')

md("""
## Part C · Train `pt_vel_v1` (velocity loss)

Same settings as `pt_baseline_v2` except **`w_vel = 10`**: the loss now also compares how far each point moves from one frame to the next with the real movement. The run saves to `Drive/BahuSanket/checkpoints/pt_vel_v1/` every epoch and resumes after a disconnect.

Every 2 epochs it generates 200 validation clips and prints DTW-MJE, length ratio and motion_ratio. `best.pt` is the epoch with the lowest validation DTW-MJE.
""")

code(r'''
CONFIG = dict(
    RUN_NAME="pt_vel_v1",
    text_encoder="scratch",
    d_model=256, n_heads=4, n_enc=2, n_dec=2, ff=1024, dropout=0.1,
    epochs=40, batch_size=32, lr=3e-4, weight_decay=1e-4, warmup_steps=500, grad_clip=1.0,
    noise_std=0.03,
    w_body=1.0, w_face=1.0, w_hands=1.0,
    w_counter=1.0,
    w_vel=10.0, w_acc=0.0,        # the one change from pt_baseline_v2
    eval_every=2, eval_clips=200,
    amp=False,
    seed=42,
)
if os.environ.get("BAHU_QUICK"):   # (Claude's local smoke test only)
    CONFIG.update(epochs=int(os.environ.get("BAHU_QUICK_EPOCHS", 2)), eval_every=1, eval_clips=8,
                  d_model=64, ff=128, warmup_steps=10, batch_size=8)
CKPT_DIR = os.path.join(ROOT, "checkpoints", CONFIG["RUN_NAME"]); os.makedirs(CKPT_DIR, exist_ok=True)

torch.manual_seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
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
pw = B.point_weights(CONFIG["w_body"], CONFIG["w_face"], CONFIG["w_hands"]).to(DEVICE)

start_epoch, best_dtw, history = 0, float("inf"), []
last = os.path.join(CKPT_DIR, "last.pt")
if os.path.exists(last):
    ck = torch.load(last, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start_epoch, best_dtw, history = ck["epoch"] + 1, ck["best_dtw"], ck["history"]
    print(f"RESUMING from epoch {start_epoch} (best val DTW-MJE so far {best_dtw:.4f})")
else:
    print(f"starting a fresh run: {CONFIG['RUN_NAME']}, {len(train_dl)} steps per epoch")
''')

code(r'''
def to_dev(b):
    return {k: (v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}

def run_epoch(dl, train_mode):
    model.train(train_mode)
    sums, n, bad = {}, 0, 0
    for b in dl:
        b = to_dev(b)
        with torch.set_grad_enabled(train_mode):
            pred, cnt = model(b["ids"], b["attn"], B.teacher_inputs(b, CONFIG["noise_std"] if train_mode else 0.0))
        losses = B.pt_loss(pred, cnt, b, pw, CONFIG["w_counter"], CONFIG["w_vel"], CONFIG["w_acc"])
        if not torch.isfinite(losses["total"]):
            bad += 1                     # never let one broken batch wreck the weights
            continue
        if train_mode:
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step(); sched.step()
        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.detach())
        n += 1
    if n == 0:
        raise RuntimeError("every batch gave a non-finite loss: stop and tell Claude")
    out = {k: v / n for k, v in sums.items()}
    out["bad_batches"] = bad
    return out

val_eval_idx = sorted(np.random.RandomState(0).choice(len(val["poses"]), min(CONFIG["eval_clips"], len(val["poses"])), replace=False).tolist())

for epoch in range(start_epoch, CONFIG["epochs"]):
    t0 = time.time()
    tr = run_epoch(train_dl, True)
    va = run_epoch(val_dl, False)
    row = {"epoch": epoch, "train_loss": tr["total"], "train_pose": tr["pose"], "train_vel": tr.get("vel", 0.0),
           "val_loss": va["total"], "bad_batches": tr["bad_batches"] + va["bad_batches"],
           "lr": sched.get_last_lr()[0], "secs": time.time() - t0}
    if (epoch + 1) % CONFIG["eval_every"] == 0 or epoch + 1 == CONFIG["epochs"]:
        vf, vc = B.generate_for_split(model, val, tok, val_eval_idx, DEVICE, batch_size=64)
        thr, _ = B.calibrate_stop(vf, vc, val, val_eval_idx)
        r, _ = B.score_predictions(B.apply_stop(vf, vc, thr), val, val_eval_idx)
        row.update(val_dtw=r["all"], val_dtw_hands=r["hands"], val_len_ratio=r["len_ratio"],
                   val_motion_ratio=r["motion_ratio"], val_stop=thr)
        if r["all"] < best_dtw:
            best_dtw = r["all"]
            torch.save({"model": model.state_dict(), "config": CONFIG, "epoch": epoch, "val_dtw": best_dtw},
                       os.path.join(CKPT_DIR, "best.pt"))
            row["saved_best"] = True
    history.append(row)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "epoch": epoch, "best_dtw": best_dtw, "history": history, "config": CONFIG}, last + ".tmp")
    os.replace(last + ".tmp", last)          # write-then-rename so a disconnect never leaves a broken file
    json.dump(history, open(os.path.join(CKPT_DIR, "history.json"), "w"), indent=1)
    msg = (f"epoch {epoch+1:>3}/{CONFIG['epochs']} | train {tr['total']:.4f} (pose {tr['pose']:.4f}, vel {tr.get('vel', 0):.5f})"
           f" | val {va['total']:.4f} | {row['secs']:.0f}s" + (f" | skipped {row['bad_batches']} bad batches" if row["bad_batches"] else ""))
    if "val_dtw" in row:
        msg += (f" | val DTW-MJE {row['val_dtw']:.4f} len x{row['val_len_ratio']:.2f} motion x{row['val_motion_ratio']:.2f}"
                f" (stop {row['val_stop']})" + ("  *best*" if row.get("saved_best") else ""))
    print(msg, flush=True)
print("training finished; checkpoints in", CKPT_DIR)
''')

md("""
## Part D · Score `pt_vel_v1` on the test set and print the comparison
""")

code(r'''
best_ck = torch.load(os.path.join(CKPT_DIR, "best.pt"), map_location=DEVICE, weights_only=False)
model.load_state_dict(best_ck["model"])
print(f"pt_vel_v1: best epoch {best_ck['epoch']+1} (val DTW-MJE {best_ck['val_dtw']:.4f})")
RESULTS["pt_vel_v1"], _, _ = score_run(model, "pt_vel_v1")
final_epoch = history[-1]["epoch"]
if best_ck["epoch"] != final_epoch:          # also score the last epoch: the best-by-DTW one may be the stillest
    model.load_state_dict(torch.load(last, map_location=DEVICE, weights_only=False)["model"])
    RESULTS[f"pt_vel_v1 (last epoch {final_epoch+1})"], _, _ = score_run(model, "pt_vel_v1 last")
json.dump({"split": "test", "n_clips": len(test_idx), "real_motion": real_motion, "results": RESULTS,
           "config": CONFIG, "best_epoch": best_ck["epoch"] + 1},
          open(os.path.join(RESULTS_DIR, "compare_03_test_scores.json"), "w"), indent=2)
print("saved results/compare_03_test_scores.json")

print("\n===== PASTE EVERYTHING BELOW THIS LINE BACK TO CLAUDE =====")
print(f"clips: train {len(train['poses'])} / val {len(val['poses'])} / test {len(test['poses'])}")
for h in history:
    if "val_dtw" in h:
        print(f"ep{h['epoch']+1} tr {h['train_loss']:.4f} vel {h['train_vel']:.5f} va {h['val_loss']:.4f}"
              f" vdtw {h['val_dtw']:.4f} len {h['val_len_ratio']:.2f} mot {h['val_motion_ratio']:.2f} stop {h['val_stop']}"
              + (f" bad {h['bad_batches']}" if h.get("bad_batches") else ""))
print(f"TEST ({len(test_idx)} clips)  all / body / face / hands / len_ratio / motion_ratio   [stop cut-off]")
for k, r in RESULTS.items():
    print(f"  {k:<30}" + " / ".join(f"{r.get(c, float('nan')):.4f}" for c in COLS) + (f"   [{r['stop_at']}]" if "stop_at" in r else ""))
''')

nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "03_velocity_and_rescore.ipynb")
print("ok", len(C), "cells")
