import nbformat as nbf
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 06 · Text → pose codes

**Where we are.** Notebook 05 built a vocabulary of 512 pose codes (1 code = 4 frames). Real signing turned into codes and back scored **DTW-MJE 0.079**, far below our best text model (0.215) and the mean-pose floor (0.231), and kept 91% of the movement.

**What this notebook does.** It trains a transformer that reads the English sentence and **writes pose codes one at a time**, then an *End* code. The notebook-05 decoder turns those codes into poses, which are scored exactly like our earlier models.
* Recipe: T2M-GPT (Zhang et al., CVPR 2023, https://arxiv.org/abs/2301.06052): next-code prediction, an End token, and *corruption*. During training a random share of the input codes is swapped for random ones, so the model learns to carry on sensibly after its own mistakes.
* Text side as in T2S-GPT (Yin et al., ACL 2024, https://arxiv.org/abs/2406.07119), which applies this to sign language: the model looks at **every word** of the sentence (here through a frozen DistilBERT), not one summary vector.

**Three checks at the end:**
1. **real length**: the result stretched to the real clip's length, so it compares fairly with notebook 04's table (best 0.2152).
2. **own length**: the model decided when to stop (End code).
3. **shuffled sentences**: the same model given *someone else's* sentence. If that scores as well as the right sentence, the model is ignoring the text.

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. It saves to Drive every epoch; if Colab disconnects, Run all again and it resumes.
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
import bahusanket_t2c as T; importlib.reload(T)
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoTokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU: training will be very slow)")
''')

code(r'''
# 2. Load val / test poses (for scoring) and the pose-code sequences saved by notebook 05
TR = os.path.join(ROOT, "training_ready")
val = B.load_split(os.path.join(TR, "val.npz"))
test = B.load_split(os.path.join(TR, "test.npz"))
VQ_RUN = "vq_v1"
tok = np.load(os.path.join(ROOT, "tokens", f"pose_codes_{VQ_RUN}.npz"), allow_pickle=True)
codes = {s: list(tok[f"{s}_codes"]) for s in ["train", "val", "test"]}
texts = {s: [str(t) for t in tok[f"{s}_texts"]] for s in ["train", "val", "test"]}
# The code file must line up clip-for-clip with the pose files (same clips, same order).
for name, d in [("val", val), ("test", test)]:
    assert list(tok[f"{name}_uids"]) == list(d["uids"]), f"{name}: code file and pose file disagree"
    assert all(len(c) == int(np.ceil(len(p) / V.DOWN)) for c, p in zip(codes[name], d["poses"])), name
vid = lambda s: {str(u).rsplit("-", 1)[0] for u in tok[f"{s}_uids"]}
assert not (vid("train") & vid("test")) and not (vid("train") & vid("val")), "video leakage between splits!"
for s in ["train", "val", "test"]:
    n = np.array([len(c) for c in codes[s]])
    print(f"{s}: {len(n)} clips, codes per clip median {np.median(n):.0f}, max {n.max()}")
''')

code(r'''
# 3. Load the pose-code vocabulary (notebook 05) and check it still rebuilds real clips
ck = torch.load(os.path.join(ROOT, "checkpoints", VQ_RUN, "last.pt"), map_location=DEVICE, weights_only=False)
vc = ck["config"]
vq = V.PoseVQVAE(width=vc["width"], code_dim=vc["code_dim"], n_codes=vc["n_codes"]).to(DEVICE)
vq.load_state_dict(ck["model"]); vq.eval()
N_CODES = vc["n_codes"]
chk = list(range(min(100, len(test["poses"]))))
r, _ = B.score_predictions(T.codes_to_poses(vq, [codes["test"][i] for i in chk], DEVICE), test, chk)
print(f"vocabulary check on {len(chk)} test clips: real codes -> poses DTW-MJE {r['all']:.4f} (notebook 05: 0.0791 on all clips)")
''')

md("""
## Part A · Train the text → codes model

6 decoder layers, 512 wide (~25 M trainable weights). The papers use 18 layers, but they had more data per model, and on our 15K sentences a big model would mostly memorise. Every 5 epochs it writes codes for 200 validation sentences and scores them (DTW-MJE, real length). The best of those is kept as `best.pt`.
""")

code(r"""
CFG = dict(
    RUN_NAME="t2c_v1", VQ_RUN=VQ_RUN, n_codes=N_CODES,
    d_model=512, n_heads=8, n_enc=2, n_dec=6, ff=2048, dropout=0.1, max_codes=80,
    text_encoder="distilbert",
    epochs=100, batch_size=64, lr=2e-4, lr_min=1e-5, warmup=1000, weight_decay=0.01,
    tau_max=1.0,                  # corruption share tau ~ U[0, 1] per sentence (T2M-GPT)
    amp=True, eval_every=5, eval_clips=200, seed=0,
)
if os.environ.get("BAHU_QUICK"):   # (Claude's local smoke test only)
    CFG.update(d_model=64, n_heads=4, n_enc=1, n_dec=2, ff=128, epochs=int(os.environ.get("BAHU_QUICK_EPOCHS", 4)),
               batch_size=8, warmup=10, eval_every=2, eval_clips=10)
RUN_DIR = os.path.join(ROOT, "checkpoints", CFG["RUN_NAME"]); os.makedirs(RUN_DIR, exist_ok=True)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data = {s: T.CodeTextData(texts[s], codes[s], tokenizer, CFG["max_codes"]) for s in ["train", "val", "test"]}
model = T.Text2Codes(n_codes=N_CODES, d_model=CFG["d_model"], n_heads=CFG["n_heads"], n_enc=CFG["n_enc"],
                     n_dec=CFG["n_dec"], ff=CFG["ff"], dropout=CFG["dropout"], max_codes=CFG["max_codes"],
                     text_encoder=CFG["text_encoder"]).to(DEVICE)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=CFG["lr"], betas=(0.9, 0.99), weight_decay=CFG["weight_decay"])
steps_per_epoch = int(np.ceil(len(data["train"]) / CFG["batch_size"]))
total_steps = CFG["epochs"] * steps_per_epoch
def lr_at(step):                  # short warm-up, then cosine down to lr_min
    if step < CFG["warmup"]:
        return CFG["lr"] * (step + 1) / CFG["warmup"]
    t = (step - CFG["warmup"]) / max(1, total_steps - CFG["warmup"])
    return CFG["lr_min"] + 0.5 * (CFG["lr"] - CFG["lr_min"]) * (1 + np.cos(np.pi * min(1.0, t)))
use_amp = DEVICE == "cuda" and CFG["amp"]
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
epoch, step, history, best = 0, 0, [], None
LAST, BEST = os.path.join(RUN_DIR, "last.pt"), os.path.join(RUN_DIR, "best.pt")
if os.path.exists(LAST):
    ck = torch.load(LAST, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    epoch, step, history, best = ck["epoch"], ck["step"], ck["history"], ck["best"]
    print(f"RESUMING after epoch {epoch}")
else:
    print("starting a fresh run:", CFG["RUN_NAME"])
print(f"trainable weights: {sum(p.numel() for p in params)/1e6:.1f} M | {steps_per_epoch} steps per epoch")
""")

code(r"""
def save(path, extra=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "step": step, "history": history, "best": best, "config": CFG, **(extra or {})},
               path + ".tmp")
    os.replace(path + ".tmp", path)        # write-then-rename: a disconnect never leaves a broken file

@torch.no_grad()
def val_loss():
    # next-code loss and accuracy on all val sentences, no corruption
    model.eval(); tot, right, n = 0.0, 0, 0
    for b in data["val"].batches(128, shuffle=False):
        ids, attn, inp, tgt = [x.to(DEVICE) for x in T.make_batch(data["val"].ids[b], data["val"].attn[b],
                                                                   [data["val"].codes[i] for i in b], model, train=False)]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(ids, attn, inp)
        m = tgt != -100
        tot += float(F.cross_entropy(logits.float()[m], tgt[m], reduction="sum"))
        right += int((logits.argmax(-1)[m] == tgt[m]).sum()); n += int(m.sum())
    return tot / n, right / n

eval_idx = sorted(np.random.RandomState(1).choice(len(val["poses"]), min(CFG["eval_clips"], len(val["poses"])), replace=False).tolist())
def gen_eval():
    gen = T.generate_codes(model, data["val"], eval_idx, DEVICE, mode="greedy")
    own, real = T.score_both(T.codes_to_poses(vq, gen, DEVICE), val, eval_idx)
    lr_ = np.median([len(g) / max(1, len(data["val"].codes[i])) for g, i in zip(gen, eval_idx)])
    return {"val_dtw_real": real["all"], "val_hands_real": real["hands"], "val_motion": real["motion_ratio"],
            "val_dtw_own": own["all"], "val_len_ratio": float(lr_)}

t_start = time.time()
while epoch < CFG["epochs"]:
    model.train(); t0, sums, n = time.time(), 0.0, 0
    for b in data["train"].batches(CFG["batch_size"]):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ids, attn, inp, tgt = [x.to(DEVICE) for x in T.make_batch(data["train"].ids[b], data["train"].attn[b],
                                                                   [data["train"].codes[i] for i in b], model, CFG["tau_max"])]
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(ids, attn, inp)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            step += 1; continue                       # never let one broken batch wreck the weights
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()
        sums += float(loss.detach()); n += 1; step += 1
    epoch += 1
    row = {"epoch": epoch, "train_loss": sums / max(n, 1), "secs": time.time() - t0}
    row["val_loss"], row["val_acc"] = val_loss()
    msg = f"epoch {epoch:>3} | train loss {row['train_loss']:.3f} | val loss {row['val_loss']:.3f}, next-code accuracy {row['val_acc']:.1%}"
    if epoch % CFG["eval_every"] == 0 or epoch == CFG["epochs"]:
        row.update(gen_eval())
        msg += (f" | val DTW-MJE {row['val_dtw_real']:.4f} (hands {row['val_hands_real']:.4f}), motion x{row['val_motion']:.2f},"
                f" length x{row['val_len_ratio']:.2f}")
        if best is None or row["val_dtw_real"] < best["val_dtw_real"]:
            best = dict(row); save(BEST); msg += "  <- best so far"
    history.append(row)
    save(LAST)
    json.dump(history, open(os.path.join(RUN_DIR, "history.json"), "w"), indent=1)
    print(msg, flush=True)
    if epoch == 1 or len(history) == 1:
        print(f"   ({row['secs']:.0f} s per epoch -> about {(CFG['epochs'] - epoch) * row['secs'] / 60:.0f} more minutes)", flush=True)
print("training finished after epoch", epoch, "| best:", {k: round(v, 4) for k, v in (best or {}).items()})
""")

md("""
## Part B · Score on the test sentences

The best checkpoint writes codes for all test sentences in two ways:
* **greedy**: always the most likely next code.
* **sample**: draws the next code at random according to the model's probabilities, as in T2M-GPT. This gives more varied movement.

Then the **shuffled-sentence** check is run with greedy decoding.
""")

code(r"""
ck = torch.load(BEST if os.path.exists(BEST) else LAST, map_location=DEVICE, weights_only=False)
model.load_state_dict(ck["model"]); model.eval()
print("using checkpoint from epoch", ck["epoch"])
test_idx = list(range(len(test["poses"])))
torch.manual_seed(0)
results, t0 = {}, time.time()
for name, kw in [("greedy", dict(mode="greedy")), ("sample", dict(mode="sample", temperature=1.0))]:
    gen = T.generate_codes(model, data["test"], test_idx, DEVICE, **kw)
    own, real = T.score_both(T.codes_to_poses(vq, gen, DEVICE), test, test_idx)
    results[name] = {"real_length": real, "own_length": own,
                     "len_ratio_median": float(np.median([len(g) / max(1, len(data["test"].codes[i])) for g, i in zip(gen, test_idx)])),
                     "codes_used": int(len(np.unique(np.concatenate([g for g in gen if len(g)]))))}
    if name == "greedy":
        examples = [(texts["test"][i], len(gen[i]), len(data["test"].codes[i])) for i in test_idx[:5]]
    print(f"[{time.time()-t0:.0f}s] {name} done", flush=True)

# Shuffled-sentence check: give each clip a different clip's sentence
perm = np.random.RandomState(0).permutation(len(test_idx))
shuf = T.CodeTextData([texts["test"][j] for j in perm], codes["test"], tokenizer, CFG["max_codes"])
gen = T.generate_codes(model, shuf, test_idx, DEVICE, mode="greedy")
own, real = T.score_both(T.codes_to_poses(vq, gen, DEVICE), test, test_idx)
results["shuffled_text_greedy"] = {"real_length": real, "own_length": own}
json.dump({"run": CFG["RUN_NAME"], "epoch": ck["epoch"], "best_val": ck["best"], "test": results, "config": CFG},
          open(os.path.join(RESULTS_DIR, f"{CFG['RUN_NAME']}_test_scores.json"), "w"), indent=2)
print(f"[{time.time()-t0:.0f}s] saved results/{CFG['RUN_NAME']}_test_scores.json")
""")

code(r"""
print("\n===== PASTE EVERYTHING BELOW THIS LINE BACK TO CLAUDE =====")
for h in history:
    if "val_dtw_real" in h:
        print(f"ep {h['epoch']} train {h['train_loss']:.3f} val {h['val_loss']:.3f} acc {h['val_acc']:.1%} | "
              f"dtw {h['val_dtw_real']:.4f} hands {h['val_hands_real']:.4f} mot {h['val_motion']:.2f} len {h['val_len_ratio']:.2f}")
print(f"best epoch {ck['epoch']}, {history[0]['secs']:.0f}s/epoch")
print(f"TEST ({len(test_idx)} clips)        DTW-MJE all / body / face / hands | motion | (own length: all, len) | codes used")
for name, r in results.items():
    a, o = r["real_length"], r["own_length"]
    print(f"  {name:<20} {a['all']:.4f} / {a['body']:.4f} / {a['face']:.4f} / {a['hands']:.4f} | {a['motion_ratio']:.2f} | "
          f"({o['all']:.4f}, x{r.get('len_ratio_median', float('nan')):.2f}) | {r.get('codes_used', '-')}")
print("reference (real length): floor 0.2308/hands 0.4410; best old model 0.2152/hands 0.4228, motion 0.10; codes round trip 0.0791")
for t, g, r in examples:
    print(f"  example: {g} codes (real {r}) {t[:60]!r}")
""")

nb["cells"] = C
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
nbf.write(nb, "06_text_to_codes.ipynb")
print("ok", len(C), "cells")
