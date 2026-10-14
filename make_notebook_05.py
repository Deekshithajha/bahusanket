import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 05 · Pose-code vocabulary (VQ-VAE)

**Why.** Our text-to-pose models predict coordinates directly, and when unsure they output the *average* pose, so they barely move (notebook 04: motion 10% of a real signer). Recent work instead first learns a **vocabulary of pose codes**: each code stands for a short piece of movement (here 4 frames = 0.16 s). A text model then *picks* codes one by one, like words, and picking a code can't blur two movements into a frozen average.
* T2M-GPT (Zhang et al., CVPR 2023, https://arxiv.org/abs/2301.06052): the recipe we copy (sizes, losses, training settings).
* T2S-GPT (Yin et al., ACL 2024, https://arxiv.org/abs/2406.07119): the same idea for sign language. On the German PHOENIX14T benchmark, the code-based models clearly beat the Progressive Transformer (back-translation BLEU-4 8.0 to 11.9 vs 4.4).

**What this notebook does (no text yet).** It trains only the vocabulary (a "VQ-VAE": encoder → nearest code → decoder) and checks that **real signing, turned into codes and back, still looks like the original**. If the rebuilt clips are close to the real ones *and* move like them, the vocabulary is good enough to build the text-to-codes model on next.

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. Training saves to Drive every 1,000 steps; if Colab disconnects, Run all again and it resumes.
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
import bahusanket_vq as V; importlib.reload(V)
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
## Part A · Train the pose-code vocabulary

Settings follow T2M-GPT: 512 codes, 1 code per 4 frames, 64-frame training windows, smooth-L1 loss on positions plus 0.5 × on velocities, codebook updated by moving average with reset of unused codes. Fewer steps than the paper (a T4 is slower than their V100); the speed and an estimate of the remaining time are printed after the first few hundred steps.
""")

code(r"""
VQ = dict(
    RUN_NAME="vq_v1",
    n_codes=512, code_dim=512, width=512,     # T2M-GPT: 512 x 512 codebook, 512-wide convolutions
    crop=64, batch_size=128, iters=30000, lr=2e-4, lr_low=1e-5, lr_drop_at=25000,
    alpha_vel=0.5, beta=0.02,                 # T2M-GPT loss weights
    amp=True,                                 # float16 for speed on the T4; the codebook itself stays float32
    log_every=500, save_every=1000, eval_every=5000, eval_clips=200, seed=0,
)
if os.environ.get("BAHU_QUICK"):   # (Claude's local smoke test only)
    VQ.update(n_codes=64, code_dim=64, width=64, batch_size=16, iters=int(os.environ.get("BAHU_QUICK_ITERS", 300)),
              lr_drop_at=200, log_every=50, save_every=100, eval_every=150, eval_clips=10)
VQ_DIR = os.path.join(ROOT, "checkpoints", VQ["RUN_NAME"]); os.makedirs(VQ_DIR, exist_ok=True)
torch.manual_seed(VQ["seed"]); np.random.seed(VQ["seed"])

vq = V.PoseVQVAE(width=VQ["width"], code_dim=VQ["code_dim"], n_codes=VQ["n_codes"]).to(DEVICE)
opt = torch.optim.AdamW(vq.parameters(), lr=VQ["lr"], betas=(0.9, 0.99))
use_amp = DEVICE == "cuda" and VQ["amp"]
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
step, history = 0, []
LAST = os.path.join(VQ_DIR, "last.pt")
if os.path.exists(LAST):
    ck = torch.load(LAST, map_location=DEVICE, weights_only=False)
    vq.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    step, history = ck["step"], ck["history"]
    print(f"RESUMING from step {step}")
else:
    print("starting a fresh run:", VQ["RUN_NAME"])
print(f"model parameters: {sum(p.numel() for p in vq.parameters())/1e6:.1f} M")
""")

code(r"""
def save_ckpt():
    torch.save({"model": vq.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "history": history, "config": VQ}, LAST + ".tmp")
    os.replace(LAST + ".tmp", LAST)        # write-then-rename: a disconnect never leaves a broken file
    json.dump(history, open(os.path.join(VQ_DIR, "history.json"), "w"), indent=1)

eval_idx = sorted(np.random.RandomState(1).choice(len(val["poses"]), min(VQ["eval_clips"], len(val["poses"])), replace=False).tolist())
def quick_eval():
    codes, rec = V.tokenize_and_rebuild(vq, val, eval_idx, DEVICE)
    used = len(np.unique(np.concatenate(codes))) / VQ["n_codes"]
    r, _ = B.score_predictions(rec, val, eval_idx)
    vq.train()
    return {"val_mje": V.aligned_mje(rec, val, eval_idx), "val_dtw": r["all"], "val_hands": r["hands"],
            "val_motion_ratio": r["motion_ratio"], "codes_used": used}

dl = torch.utils.data.DataLoader(V.CropDataset(train, VQ["crop"], samples_per_epoch=10**9), batch_size=VQ["batch_size"],
                                 num_workers=2, worker_init_fn=V.worker_init, drop_last=True)
vq.train()
t0, t_log, sums, n, bad = time.time(), time.time(), {}, 0, 0
for x, pm in dl:
    if step >= VQ["iters"]:
        break
    for g in opt.param_groups:
        g["lr"] = VQ["lr"] if step < VQ["lr_drop_at"] else VQ["lr_low"]
    x, pm = x.to(DEVICE, non_blocking=True), pm.to(DEVICE, non_blocking=True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        rec, commit, st, _ = vq(x)
    L = V.vq_loss(rec.float(), x, pm, commit, VQ["alpha_vel"], VQ["beta"])
    if not torch.isfinite(L["total"]):
        bad += 1; step += 1; continue                 # never let one broken batch wreck the weights
    opt.zero_grad(set_to_none=True)
    scaler.scale(L["total"]).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(vq.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    for k, v in {**L, **st}.items():
        sums[k] = sums.get(k, 0.0) + float(v.detach())
    n += 1; step += 1
    if step == 200:
        rate = 200 / (time.time() - t0)
        print(f"speed: {rate:.1f} steps/s -> about {(VQ['iters'] - step) / rate / 60:.0f} more minutes of training", flush=True)
    if step % VQ["log_every"] == 0:
        row = {"step": step, **{k: v / n for k, v in sums.items()}, "bad": bad, "secs": time.time() - t_log}
        if step % VQ["eval_every"] == 0 or step == VQ["iters"]:
            row.update(quick_eval())
        history.append(row); sums, n, bad, t_log = {}, 0, 0, time.time()
        print(f"step {step:>6} | loss {row['total']:.4f} (pos {row['pos']:.4f}, vel {row['vel']:.4f}, commit {row['commit']:.4f})"
              f" | codes in use {row['used']:.0%}, perplexity {row['perplexity']:.0f}"
              + (f" | val: error {row['val_mje']:.4f}, DTW-MJE {row['val_dtw']:.4f}, motion x{row['val_motion_ratio']:.2f}, {row['codes_used']:.0%} of codes" if "val_mje" in row else "")
              + (f" | skipped {row['bad']} bad batches" if row["bad"] else ""), flush=True)
    if step % VQ["save_every"] == 0 or step == VQ["iters"]:
        save_ckpt()
save_ckpt()
print("training finished at step", step)
""")

md("""
## Part B · Does real signing survive the round trip?

Every **test** clip is turned into codes and decoded back, then compared with the original:
* **error**: average distance per point, frame by frame (shoulder widths). Lower is better.
* **DTW-MJE**: the same score as notebooks 03 and 04, so it can sit next to our models: the mean-pose floor scored **0.2308**, our best text model **0.2152**. A good vocabulary should be far below both.
* **motion_ratio**: should be close to **1.0** (the rebuilt clips move like the real ones).
""")

code(r"""
t0 = time.time()
test_idx = list(range(len(test["poses"])))
codes_te, rec_te = V.tokenize_and_rebuild(vq, test, test_idx, DEVICE)
res, rows = B.score_predictions(rec_te, test, test_idx)
res["error"] = V.aligned_mje(rec_te, test, test_idx)
allc = np.concatenate(codes_te)
cnt = np.bincount(allc, minlength=VQ["n_codes"]); p = cnt / cnt.sum()
res["codes_used"] = float((cnt > 0).mean()); res["perplexity"] = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
json.dump({"run": VQ["RUN_NAME"], "step": step, "test": res, "config": VQ},
          open(os.path.join(RESULTS_DIR, f"{VQ['RUN_NAME']}_reconstruction.json"), "w"), indent=2)
print(f"[{time.time()-t0:.0f}s] TEST reconstruction ({len(test_idx)} clips)")
print(f"  error {res['error']:.4f} | DTW-MJE all {res['all']:.4f} body {res['body']:.4f} face {res['face']:.4f} hands {res['hands']:.4f}"
      f" | motion_ratio {res['motion_ratio']:.2f} | codes used {res['codes_used']:.0%}, perplexity {res['perplexity']:.0f}")
worst = np.argsort([-r["all"] for r in rows])[:3]
for k in worst:
    print(f"  worst: DTW-MJE {rows[k]['all']:.3f}  {test['texts'][k][:70]!r}")
""")

md("""
## Part C · Save the code sequences for the next step

Every clip in train / val / test becomes a short list of code numbers (one per 4 frames). The next notebook will train the text → codes model on these. Saved to `Drive/BahuSanket/tokens/`.
""")

code(r"""
TOK_DIR = os.path.join(ROOT, "tokens"); os.makedirs(TOK_DIR, exist_ok=True)
out = {}
for name, d in [("train", train), ("val", val), ("test", test)]:
    idx = list(range(len(d["poses"])))
    codes = codes_te if name == "test" else V.tokenize_and_rebuild(vq, d, idx, DEVICE)[0]
    out[f"{name}_uids"] = np.array(d["uids"]); out[f"{name}_texts"] = np.array(d["texts"])
    arr = np.empty(len(codes), dtype=object); arr[:] = codes; out[f"{name}_codes"] = arr; out[f"{name}_frames"] = np.array([len(p) for p in d["poses"]])
path = os.path.join(TOK_DIR, f"pose_codes_{VQ['RUN_NAME']}.npz")
np.savez_compressed(path, **out)
print("saved", path, f"({os.path.getsize(path)/1e6:.1f} MB)")

print("\n===== PASTE EVERYTHING BELOW THIS LINE BACK TO CLAUDE =====")
for h in history:
    if "val_mje" in h:
        print(f"step {h['step']} loss {h['total']:.4f} pos {h['pos']:.4f} vel {h['vel']:.4f} commit {h['commit']:.4f} perp {h['perplexity']:.0f}"
              f" | val err {h['val_mje']:.4f} dtw {h['val_dtw']:.4f} mot {h['val_motion_ratio']:.2f} codes {h['codes_used']:.0%}")
print(f"TEST reconstruction: error {res['error']:.4f} | DTW-MJE {res['all']:.4f} / {res['body']:.4f} / {res['face']:.4f} / {res['hands']:.4f}"
      f" | motion {res['motion_ratio']:.2f} | codes used {res['codes_used']:.0%} perplexity {res['perplexity']:.0f}")
print("reference: mean-pose floor DTW-MJE 0.2308, best text model 0.2152")
""")

nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "05_pose_vocabulary.ipynb")
print("ok", len(C), "cells")
