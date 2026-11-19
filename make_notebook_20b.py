"""Builds 20b_pack_for_laptop.ipynb: packs everything the laptop demo needs into BahuSanket/live_bundle/bahusanket_bundle.zip."""
import json, base64
import nbformat as nbf
emb = {n: base64.b64encode(open(f"code/{n}", "rb").read()).decode() for n in ("bahusanket_rvq.py", "bahusanket_rtrans.py")}
nb = nbf.v4.new_notebook(); C = []
C.append(nbf.v4.new_markdown_cell("""# BahuSanket · 20b · Pack the models for your laptop

Runs on **CPU** (no GPU needed, about 3–5 minutes). It copies the trained models, the code and the avatar files from your Drive into one
file, `BahuSanket/live_bundle/bahusanket_bundle.zip`. Download that file from Google Drive into your **Downloads** folder, then run
`bash setup.sh` in the `live_local` project (see its README).

The frozen DistilBERT part of two checkpoints is left out (setup.sh downloads the identical original), which saves about 500 MB.
**How to run:** Runtime → Run all."""))
C.append(nbf.v4.new_code_cell(r'''
# 1. Find the BahuSanket folder on Drive
import os, glob, json, zipfile, shutil, base64, hashlib, urllib.request, time
import torch
from google.colab import drive
drive.mount("/content/drive")
ROOT = "/content/drive/MyDrive/BahuSanket"
if not os.path.isdir(ROOT):
    hits = glob.glob("/content/drive/MyDrive/**/BahuSanket", recursive=True)
    assert hits, "Could not find the BahuSanket folder in your Drive"
    ROOT = hits[0]
print("Drive folder:", ROOT)
'''.strip()))
C.append(nbf.v4.new_code_cell(r'''
# 2. Pack
OUT = "/content/bundle"; shutil.rmtree(OUT, ignore_errors=True); os.makedirs(f"{OUT}/checkpoints"); os.makedirs(f"{OUT}/code")
CK = os.path.join(ROOT, "checkpoints")
KEEP = {"model", "config", "mu", "sd", "zmean", "zstd", "epoch", "step"}
for src, dst, drop_bert in [("vq_v2_123pts/last.pt", "vq_v2_123pts.pt", False), ("rvq_v2_123pts/last.pt", "rvq_v2_123pts.pt", False),
                            ("rtrans_123pts_b/best.pt", "rtrans_123pts_b.pt", True), ("t2c_more_123pts/best_loss.pt", "t2c_more_123pts.pt", True),
                            ("length_predictor_v1.pt", "length_predictor_v1.pt", False), ("depth_lifter_v1/best.pt", "depth_lifter_v1.pt", False)]:
    ck = torch.load(os.path.join(CK, src), map_location="cpu", weights_only=False)
    small = {k: v for k, v in ck.items() if k in KEEP}
    if drop_bert:
        small["model"] = {k: v for k, v in ck["model"].items() if not k.startswith("bert.")}
    torch.save(small, f"{OUT}/checkpoints/{dst}")
    print(f"  {src:<32} -> {dst:<24} {os.path.getsize(f'{OUT}/checkpoints/{dst}') / 1e6:6.1f} MB")
for n in ("bahusanket_lib.py", "bahusanket_vq.py", "bahusanket_t2c.py", "bahusanket_layout.py"):
    shutil.copy(os.path.join(ROOT, "code", n), f"{OUT}/code/{n}")
for n, b in EMBEDDED.items():                                   # notebook 13/14b module code (the same as cell 2 of notebooks 17-20)
    open(f"{OUT}/code/{n}", "wb").write(base64.b64decode(b))
shutil.copy(os.path.join(ROOT, "model_point_indices_v2.json"), OUT)
shutil.copy(os.path.join(ROOT, "results", "12_bone_limits.json"), OUT)
tc = os.path.join(ROOT, "translations19", "cache.json")
if os.path.exists(tc):
    shutil.copy(tc, f"{OUT}/translations_cache.json")
AV = os.path.join(ROOT, "avatar")
for n in ("clips_paragraph_19.json", "clips_model_18.json", "clips.json"):
    if os.path.exists(os.path.join(AV, n)):
        shutil.copy(os.path.join(AV, n), OUT)
    else:
        print("  (not on Drive, skipped:", n, ")")
if os.path.exists(os.path.join(AV, "avatar.b64.txt")):
    shutil.copy(os.path.join(AV, "avatar.b64.txt"), OUT)
else:
    url = "https://raw.githubusercontent.com/pixiv/three-vrm/release/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm"
    vrm = urllib.request.urlopen(url, timeout=120).read()
    assert hashlib.md5(vrm).hexdigest() == "b36a6fbb65316d5684de896dedee4f26", "unexpected avatar file"
    open(f"{OUT}/avatar.b64.txt", "w").write(base64.b64encode(vrm).decode())
ZIPF = os.path.join(ROOT, "live_bundle", "bahusanket_bundle.zip"); os.makedirs(os.path.dirname(ZIPF), exist_ok=True)
with zipfile.ZipFile("/content/bahusanket_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for dp, _, fs in os.walk(OUT):
        for f in fs:
            z.write(os.path.join(dp, f), os.path.relpath(os.path.join(dp, f), OUT))
shutil.copy("/content/bahusanket_bundle.zip", ZIPF)
print(f"\nsaved {ZIPF} ({os.path.getsize(ZIPF) / 1e6:.0f} MB). Download it from Google Drive into your Downloads folder.")
'''.strip().replace("EMBEDDED.items()", "EMBEDDED.items()").replace("# 2. Pack", "# 2. Pack\nEMBEDDED = " + json.dumps(emb))))
nb["cells"] = C
nb.metadata = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}}
nbf.write(nb, "20b_pack_for_laptop.ipynb"); print("ok")
