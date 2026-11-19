# BahuSanket live demo on your Mac (localhost)

Type a paragraph in English, Hindi or Kannada. The avatar signs it, running entirely on this laptop (no internet needed after setup).
Pipeline: IndicTrans2 (Hindi/Kannada → English) → BahuSanket text model (guidance s = 4, length control, extra codes, bone fix)
→ sentences blended (ease 8) → depth model → 3D avatar.

## One-time setup (about 30 minutes, mostly downloads)
1. In Colab, open **20b_pack_for_laptop.ipynb** and Runtime → Run all (CPU is fine). It saves
   `BahuSanket/live_bundle/bahusanket_bundle.zip` on your Drive. Download that file into your **Downloads** folder.
2. Unzip this project (`bahusanket_live_local.zip`) somewhere, e.g. Documents, and open the folder in **VS Code** (File → Open Folder).
3. In VS Code: Terminal → New Terminal, then type:
   ```
   bash setup.sh
   ```
   It finds the bundle in Downloads, installs Python 3.12 and the libraries into `.venv` (only for this folder), and downloads the
   Hugging Face models (about 4.5 GB). It asks once for your Hugging Face token (hidden while you type; it is not saved).

## Every time you present
```
bash run.sh
```
(or Terminal → Run Build Task in VS Code). After about a minute it prints `READY` and opens http://localhost:8000 in your browser.
Keep the terminal open; Ctrl+C stops it. The first sentence after starting is a little slower.

* http://localhost:8000 — the BahuSanket front end (for the panel)
* http://localhost:8000/viewer.html — the research viewer with all the switches (test sentences, real signing, depth, joins)

## If something goes wrong
* `Could not find bahusanket_bundle.zip`: put the zip in this folder or in Downloads.
* `Hugging Face refused ...`: log in at huggingface.co, click agree on https://huggingface.co/ai4bharat/indictrans2-indic-en-1B, run setup again.
* An error that mentions `mps`: start with `BAHU_DEVICE=cpu bash run.sh` (slower, but avoids the Apple GPU).
* Anything else: copy the last lines of the terminal and send them to Claude.

Avatar: VRM1 Constraint Twist Sample © pixiv Inc. (VRM Public License 1.0). Translation: IndicTrans2, AI4Bharat
(https://arxiv.org/abs/2305.16307). Data: iSign (CC BY-NC-SA 4.0).
