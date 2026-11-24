"""Diagnostic: python check.py  (prints numbers to paste back to Claude)"""
import json, os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("BAHU_DEVICE", "cpu")
import numpy as np, torch, transformers
import pipeline as P
print("versions: torch", torch.__version__, "| transformers", transformers.__version__, "| numpy", np.__version__)
for f in sorted(os.listdir(P.CK)):
    ck = torch.load(os.path.join(P.CK, f), map_location="cpu", weights_only=False)
    m = ck.get("model", {}); print(f"  {f}: keys {sorted(ck)} | {len(m)} tensors | epoch {ck.get('epoch')} step {ck.get('step')} | config {ck.get('config')}")
pi = P.POINT_IDX; print("point_idx:", type(pi).__name__, len(pi), pi[:8], "...")
s = P.Signer()
bd = P.LAY["body"]; FACE = P.LAY["parts"]["face"]; HANDS = P.LAY["parts"]["hands"]
def stats(p, name):
    p = np.asarray(p, np.float32).reshape(len(p), -1, 2)
    sh = (p[:, bd["l_shoulder"]] + p[:, bd["r_shoulder"]]) / 2; sw = np.linalg.norm(p[:, bd["l_shoulder"]] - p[:, bd["r_shoulder"]], axis=1)
    face = p[:, FACE].mean(1); hands = p[:, HANDS].mean(1)
    print(f"  {name:<34} frames {len(p):4d} | shoulder mid ({sh[:,0].mean():+.2f},{sh[:,1].mean():+.2f}) width {sw.mean():.2f} | "
          f"face ({face[:,0].mean():+.2f},{face[:,1].mean():+.2f}) | hands ({hands[:,0].mean():+.2f},{hands[:,1].mean():+.2f}) | range {p.min():+.1f}..{p.max():+.1f}")
d = json.load(open(os.path.join(P.BUNDLE, "clips_model_18.json")))
idx = d["point_idx"]; assert list(idx) == list(pi), "point order differs from the avatar files!"
real = [c for c in d["clips"] if c.get("kind") == "real"][0]; colab = [c for c in d["clips"] if c.get("kind") == "model" and c["uid"] == real["uid"]][0]
arr = lambda c: np.nan_to_num(np.array([[np.nan if v is None else v for v in r] for r in c["pts"]], np.float32)).reshape(len(c["pts"]), -1, 2)
print("sentence:", real["text"])
R = arr(real); stats(R, "real signing (iSign)"); stats(arr(colab), "model output made on Colab")
with torch.no_grad():
    T_ = int(np.ceil(len(R) / 4) * 4); x = np.zeros((1, T_, R.shape[1] * 2), np.float32); x[0, :len(R)] = R.reshape(len(R), -1)
    codes = s.rvq.encode(torch.from_numpy(x)); rec = s.rvq.decode(codes).numpy().reshape(T_, -1, 2)[:len(R)]
stats(rec, "real -> vocabulary -> back"); print("  vocabulary round-trip error:", float(np.abs(rec - R)[np.abs(R).sum(-1) > 0].mean()))
print("  codes of the real clip (first 8):", codes[0, :8, 0].tolist())
g = s.default_pipeline([real["text"]])[0]; stats(g, "model output made on this laptop")
torch.manual_seed(0)
dd = P.T.CodeTextData([real["text"]], [np.zeros(2, np.int64)], s.tokenizer, s.cfg["max_codes"])
base = s.generate_len(dd, [0], 4.0, np.array([10]), np.array([20]))[0]; print("  laptop base codes (first 12):", base[:12].tolist())
dec = P.RVQ.codes_to_poses_rvq(s.rvq, [np.stack([base] + [np.zeros_like(base)] * 3, 1)], "cpu", n_res=0)[0]; stats(dec, "laptop base codes decoded (no extras)")

print("\n--- the same path as the web page ---")
def perframe(p, name):
    p = np.asarray(p, np.float32).reshape(len(p), -1, 2)
    sh = (p[:, bd["l_shoulder"]] + p[:, bd["r_shoulder"]]) / 2; sw = np.linalg.norm(p[:, bd["l_shoulder"]] - p[:, bd["r_shoulder"]], axis=1)
    fy = p[:, FACE, 1].mean(1) - sh[:, 1]
    print(f"  {name:<34} frames {len(p):4d} | face above shoulders in {np.mean(fy < -0.1):.0%} of frames | face-y min {fy.min():+.2f} max {fy.max():+.2f} | "
          f"shoulder width min {sw.min():.2f} max {sw.max():.2f}")
for txt in ["Hi I need help", "But she did not hunt every night."]:
    torch.manual_seed(19); g = s.default_pipeline([txt])[0]; perframe(g, f"pipeline: {txt[:22]}")
    r = s.make_paragraph(txt, "en"); c = r["clip"]
    perframe(np.array(c["pts"], np.float32), f"web reply: {txt[:22]}")
    z = np.array(c["z"], np.float32); print(f"  depth z range {z.min():+.2f}..{z.max():+.2f} | point_idx same as avatar files: {r['point_idx'] == list(idx)}")
perframe(R, "real signing"); perframe(arr(colab), "Colab model output")
