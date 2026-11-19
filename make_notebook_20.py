"""Builds 20_live_demo_v2.ipynb: a live 'type a paragraph -> avatar signs it' server on Colab, for the panel presentation.
Usage: python make_notebook_20.py <avatar index.html> <app.html>"""
import json, sys, gzip, base64
import nbformat as nbf
src = json.load(open("17_pck_backtranslation_v2.ipynb"))["cells"]
nb19 = json.load(open("19_paragraphs_v2.ipynb"))["cells"]
S = lambda i: "".join(src[i]["source"])
PAGE_B64 = base64.b64encode(gzip.compress(open(sys.argv[1], "rb").read(), 9)).decode()       # research viewer (/viewer.html)
APP_B64 = base64.b64encode(gzip.compress(open(sys.argv[2], "rb").read(), 9)).decode()        # presentation front end (/)
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 20 · Live demo: type a paragraph, the avatar signs it

This notebook is for **presenting**. It starts a small web server on this Colab GPU and gives you a link to the **BahuSanket front end**:
a text box, a language picker, the avatar and live captions. Type (or paste) English, Hindi or Kannada, press **Sign it**, and a few
seconds later the avatar signs it. The six prepared paragraphs from notebook 19 are already in its "Signed so far" list.
Behind the button is exactly the pipeline of notebook 19, nothing new is trained:

IndicTrans2 (Hindi/Kannada → English) → split into sentences → our text model (guidance s = 4, length control, extra codes, bone fix)
→ sentences blended into one sequence (ease 8, chosen in notebook 19) → depth model (notebook 18) → avatar.

**On presentation day**
1. About **15 minutes before**: Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all** (about 5–8 minutes).
2. The last cell prints **a link**. Open it (it opens in a new tab, in this same browser). Try one sentence once, so everything is warm.
3. **Keep this Colab tab open** during the presentation (Colab stops idle sessions after a while; a tab left open is usually enough).
4. Present from that tab (F11 makes it full screen). Ctrl+Enter also presses Sign it; everything you sign stays in "Signed so far".
   The research viewer with all the switches is at the same link + `viewer.html`.

**Backups, in case the Wi-Fi or Colab fails:** the published avatar page (https://claude.ai/artifact/6AKrgZDSC7SeuSdcXVWFgb) already has
the six demo paragraphs in English, Hindi and Kannada, and a short screen recording of this live demo is worth making the day before.

The Hugging Face token is read from Colab **Secrets** (`HF_TOKEN`), or a hidden box asks for it. It is never printed or saved.
""")

code(S(1).replace("# 1. Setup", "# 1. Setup (+ the HF token, needed for IndicTrans2)") + r'''
if not os.environ.get("HF_TOKEN") and not os.environ.get("BAHU_ROOT"):
    try:
        from google.colab import userdata
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    except Exception:
        # Secrets only work from the Colab web page (not e.g. VS Code). Ask instead: typed in a hidden box, never printed or saved.
        from getpass import getpass
        os.environ["HF_TOKEN"] = getpass("Colab Secrets not available. Paste your Hugging Face token (hidden) and press Enter: ").strip()
    assert os.environ.get("HF_TOKEN", "").startswith("hf_"), "That does not look like a Hugging Face token (it starts with hf_)"
    print("Hugging Face token: OK (not shown)")
''')
for i in range(2, 8):
    code(S(i))
code("".join(nb19[8]["source"]))      # 8. depth model + lift()

md("""
## Translation: IndicTrans2 stays loaded

Notebook 19 started IndicTrans2 for each batch. For a live demo it must answer in about a second, so here it runs as one small
background process that loads the Hindi/Kannada → English model **once** and then waits for sentences. Translations already made in
notebook 19 (the demo paragraphs) come from `translations19/cache.json` instantly, and new ones are added to it.
""")
code(r'''
# 9. IndicTrans2 as a background worker (model loaded once)
import re, hashlib, threading
IT2_LIB = "/content/it2lib" if not os.environ.get("BAHU_ROOT") else os.path.join(ROOT, "_it2lib")
IT2_WORKER = os.path.join(os.path.dirname(IT2_LIB), "it2_worker.py")
INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"
LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda"}
TR_DIR = os.path.join(ROOT, "translations19"); os.makedirs(TR_DIR, exist_ok=True)
TR_CACHE_F = os.path.join(TR_DIR, "cache.json")
TR_CACHE = json.load(open(TR_CACHE_F)) if os.path.exists(TR_CACHE_F) else {}
print("translations already saved:", len(TR_CACHE))

if not os.path.exists(os.path.join(IT2_LIB, "transformers", "__init__.py")):
    t0 = time.time()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", IT2_LIB, "transformers==4.46.3",
                    "tokenizers==0.20.3", "huggingface_hub==0.26.2", "IndicTransToolkit==1.1.1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacremoses", "indic-nlp-library-itt", "sentencepiece"], check=True)
    print(f"IndicTrans2 libraries installed [{time.time() - t0:.0f}s]")

open(IT2_WORKER, "w").write(r"""
import sys, json, time, types
import torch
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:                              # newer transformers break the toolkit's training helpers; we only need the processor
    for n in ("collator", "evaluator"):
        m = types.ModuleType("IndicTransToolkit." + n); m.IndicDataCollator = m.IndicEvaluator = None
        sys.modules["IndicTransToolkit." + n] = m
    from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
name = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
ip = IndicProcessor(inference=True)
def say(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n"); sys.stdout.flush()
try:
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True,
                                                torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
except Exception as e:
    say({"ready": False, "error": f"{type(e).__name__}: {e}"}); sys.exit(1)
say({"ready": True, "seconds": round(time.time() - t0, 1)})
for line in sys.stdin:
    try:
        job = json.loads(line)
        out = []
        for b0 in range(0, len(job["sents"]), 8):
            batch = ip.preprocess_batch(job["sents"][b0:b0 + 8], src_lang=job["src"], tgt_lang=job["tgt"])
            enc = tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(dev)
            with torch.no_grad():
                gen = mdl.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
            dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out += [d.strip() for d in ip.postprocess_batch(dec, lang=job["tgt"])]
        say({"id": job["id"], "out": out})
    except Exception as e:
        torch.cuda.empty_cache()
        say({"id": job.get("id") if isinstance(job, dict) else None, "error": f"{type(e).__name__}: {e}"})
""")

class IT2Worker:
    """One background IndicTrans2 process; translate() sends sentences and waits for the answer (one request at a time)."""
    def __init__(self):
        self.lock = threading.Lock(); self.n = 0; self.p = None
    def start(self):
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        env["PYTHONPATH"] = IT2_LIB + os.pathsep + env.get("PYTHONPATH", "")
        self.errf = open(os.path.join(os.path.dirname(IT2_LIB), "it2_worker_err.txt"), "w")
        self.p = subprocess.Popen([sys.executable, IT2_WORKER, INDIC_EN], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.errf, text=True, bufsize=1)
        first = json.loads(self.p.stdout.readline() or '{"ready": false, "error": "the worker stopped"}')
        if not first.get("ready"):
            err = first.get("error", "") + "\n" + open(self.errf.name).read()[-3000:]
            if any(k in err.lower() for k in ("gated", "restricted", "403", "401")):
                raise RuntimeError(f"Hugging Face refused the IndicTrans2 download: open https://huggingface.co/{INDIC_EN}, click agree, "
                                   "check HF_TOKEN, then run this cell again.\n" + err[-800:])
            raise RuntimeError("IndicTrans2 could not start; paste this to Claude:\n" + err[-3000:])
        print(f"IndicTrans2 loaded in {first['seconds']}s and waiting for sentences")
    def translate(self, sents, src, tgt="eng_Latn"):
        with self.lock:
            if self.p is None or self.p.poll() is not None:
                print("(re)starting IndicTrans2 ..."); self.start()
            self.n += 1
            self.p.stdin.write(json.dumps({"id": self.n, "src": src, "tgt": tgt, "sents": sents}, ensure_ascii=False) + "\n"); self.p.stdin.flush()
            ans = json.loads(self.p.stdout.readline() or '{"error": "IndicTrans2 stopped (out of memory?)"}')
        if "error" in ans:
            raise RuntimeError("translation failed: " + ans["error"])
        return ans["out"]

def _key(model, src, tgt, s):
    return hashlib.md5(f"{model}|{src}|{tgt}|{s}".encode("utf-8")).hexdigest()
_cache_lock = threading.Lock()
def translate(sents, lang):
    """Hindi/Kannada sentences -> English (cached on Drive)."""
    s_ = LANGS[lang]
    todo = [x for x in dict.fromkeys(sents) if _key(INDIC_EN, s_, "eng_Latn", x) not in TR_CACHE]
    if todo:
        for x, y in zip(todo, IT2.translate(todo, s_)):
            TR_CACHE[_key(INDIC_EN, s_, "eng_Latn", x)] = y
        with _cache_lock:
            json.dump(TR_CACHE, open(TR_CACHE_F + ".tmp", "w"), ensure_ascii=False); os.replace(TR_CACHE_F + ".tmp", TR_CACHE_F)
    return [TR_CACHE[_key(INDIC_EN, s_, "eng_Latn", x)] for x in sents]

IT2 = IT2Worker(); IT2.start()
t0 = time.time(); print("test:", IT2.translate(["आज मौसम अच्छा है।"], "hin_Deva"), f"[{time.time() - t0:.1f}s]")
''')

code(r'''
# 10. One paragraph -> one avatar clip (the notebook-19 steps: split, translate, cut long sentences, generate, join, depth)
JOIN = "ease 8"                                   # chosen in notebook 19 (smallest jump that keeps the meaning scores)
MAX_PIECES = 12                                   # a safety limit for the live demo (about 12 sentences)
def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+|(?<=[।॥])\s*", text)
    return [p.strip() for p in parts if p and re.search(r"\w", p)]
nw = np.array([len(t.split()) for t in texts["train"]])
MAX_WORDS = int(np.clip(np.percentile(nw, 90), 10, 22))
JOINERS = {"and", "but", "because", "so", "while", "which", "who", "where", "when", "after", "before", "or", "then", "although"}
def chunk_english(s, max_words=None):
    max_words = max_words or MAX_WORDS
    w = s.split()
    if len(w) <= max_words:
        return [s]
    cands = [i for i in range(3, len(w) - 2) if w[i - 1][-1] in ",;:" or w[i].lower() in JOINERS]
    cut = min(cands, key=lambda i: abs(i - len(w) / 2)) if cands else len(w) // 2
    return chunk_english(" ".join(w[:cut]).rstrip(",;:"), max_words) + chunk_english(" ".join(w[cut:]), max_words)

bd = LAY["body"]; ARM_HAND = [bd[k] for k in ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]] + \
     list(range(LAY["parts"]["hands"].start, LAY["parts"]["hands"].stop))
def smooth01(n):
    t = (np.arange(n) + 1) / (n + 1)
    return t * t * (3 - 2 * t)
def join(clips, method):
    clips = [np.asarray(c, np.float32)[..., :2] for c in clips]
    out, bounds = clips[0].copy(), [[0, len(clips[0])]]
    for c in clips[1:]:
        if method.startswith("ease"):
            m = min(int(method.split()[1]), max(1, len(c) // 2))
            c = c.copy(); c[:m] += (out[-1] - c[0])[None] * (1 - smooth01(m))[:, None, None]
        bounds.append([len(out), len(out) + len(c)]); out = np.concatenate([out, c])
    return out, bounds
def join_jumps(seq, bounds):
    x = np.asarray(seq, np.float32)[:, ARM_HAND]
    sp = np.linalg.norm(x[1:] - x[:-1], axis=-1).mean(1)
    typ = max(float(np.median(sp)), 1e-4)
    return [round(float(sp[max(0, b0 - 3):b0 + 2].max()) / typ, 2) for b0, _ in bounds[1:]]
def r3(a):
    return [[round(float(v), 3) for v in row] for row in np.asarray(a, np.float32).reshape(len(a), -1)]

def detect_lang(text):
    kn = len(re.findall(r"[ಀ-೿]", text)); hi = len(re.findall(r"[ऀ-ॿ]", text))
    return "kn" if kn > max(hi, 2) else "hi" if hi > 2 else "en"

GPU_LOCK = threading.Lock(); SEEN = {}; N_LIVE = [0]
def make_paragraph(text, lang="auto"):
    text = re.sub(r"\s+", " ", (text or "").strip())[:1500]
    if not text:
        raise ValueError("empty text")
    lang = detect_lang(text) if lang == "auto" else lang
    if lang not in LANGS:
        raise ValueError("language must be en, hi or kn")
    if (lang, text) in SEEN:
        return {**SEEN[(lang, text)], "timing": {"cached": True}}
    t0 = time.time()
    sents = split_sentences(text) or [text]
    en = sents if lang == "en" else translate(sents, lang)
    segs = [{"src": s, "en": piece} for s, e in zip(sents, en) for piece in chunk_english(e)][:MAX_PIECES]
    t1 = time.time()
    seg_en = [g["en"] for g in segs]
    with GPU_LOCK:
        d_ = T.CodeTextData(seg_en, [np.zeros(2, np.int64)] * len(seg_en), tokenizer, CFG["max_codes"])
        poses = default_pipeline(d_, list(range(len(seg_en))), seg_en, seed=19)
        t2 = time.time()
        variants = {}
        for mth in (JOIN, "cut"):
            seq, bnd = join(poses, mth)
            z = lift([seq], [np.ones(seq.shape[:2], bool)])[0]
            variants[mth] = {"n": len(seq), "pts": r3(seq), "z": r3(z), "jumps": join_jumps(seq, bnd),
                             "segments": [{**g, "start": int(a), "end": int(b)} for g, (a, b) in zip(segs, bnd)]}
    t3 = time.time()
    N_LIVE[0] += 1
    clip = {"uid": f"live{N_LIVE[0]}_{lang}", "kind": "paragraph", "lang": lang, "source": text,
            "text": " ".join(seg_en), "join": JOIN, **variants[JOIN], "cut": variants["cut"]}
    res = {"ok": True, "clip": clip, "point_idx": [int(v) for v in POINT_IDX], "fps": 25,
           "timing": {"translate": round(t1 - t0, 1), "generate": round(t2 - t1, 1), "depth": round(t3 - t2, 1)}}
    SEEN[(lang, text)] = res
    print(f"[{time.strftime('%H:%M:%S')}] {lang}: {text[:60]!r} -> {len(segs)} pieces, {clip['n'] / 25:.1f}s of signing, "
          f"took {t3 - t0:.1f}s (translate {t1 - t0:.1f}, generate {t2 - t1:.1f}, depth {t3 - t2:.1f})", flush=True)
    return res

# warm-up: one English and one Hindi sentence (the first run of each step is the slowest)
for txt in ["Thank you for coming today.", "आप सभी का स्वागत है।"]:
    r_ = make_paragraph(txt)
print("live pipeline ready")
''')

md("""
## Start the server and open the avatar page

The link opens the BahuSanket front end. It works in this browser while this Colab tab is open. If it does not open, use the optional
cell after it.
""")
code(r'''
# 11. Web server on port 8000: the avatar page + /api/generate
import gzip, base64, urllib.request, http.server, socketserver
LIVE_DIR = "/content/live20" if not os.environ.get("BAHU_ROOT") else os.path.join(ROOT, "_live20")
os.makedirs(LIVE_DIR, exist_ok=True)
open(os.path.join(LIVE_DIR, "viewer.html"), "wb").write(gzip.decompress(base64.b64decode("''' + PAGE_B64 + r'''")))
open(os.path.join(LIVE_DIR, "app.html"), "wb").write(gzip.decompress(base64.b64decode("''' + APP_B64 + r'''")))
AV_DIR = os.path.join(ROOT, "avatar"); os.makedirs(AV_DIR, exist_ok=True)

# the avatar (pixiv's VRM1 Constraint Twist Sample, VRM Public License 1.0), kept on Drive after the first download
AV_B64 = os.path.join(AV_DIR, "avatar.b64.txt")
if not os.path.exists(AV_B64):
    url = "https://raw.githubusercontent.com/pixiv/three-vrm/release/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm"
    vrm_bytes = urllib.request.urlopen(url, timeout=120).read()
    assert hashlib.md5(vrm_bytes).hexdigest() == "b36a6fbb65316d5684de896dedee4f26", "unexpected avatar file"
    open(AV_B64, "w").write(base64.b64encode(vrm_bytes).decode())
    print("avatar downloaded")

def first_file(*names):
    for n in names:
        f = os.path.join(AV_DIR, n)
        if os.path.exists(f):
            return f
    return None
EMPTY = os.path.join(LIVE_DIR, "empty.json"); open(EMPTY, "w").write("[]")
FILES = {"/": os.path.join(LIVE_DIR, "app.html"), "/app.html": os.path.join(LIVE_DIR, "app.html"),     # the BahuSanket front end
         "/viewer.html": os.path.join(LIVE_DIR, "viewer.html"), "/avatar.b64.txt": AV_B64,          # the research viewer, same data
         "/clips_paragraph.json": first_file("clips_paragraph_19.json"),                      # notebook 19's six demo paragraphs
         "/clips_model.json": first_file("clips_model_18.json", "clips_model.json"),          # test sentences, model + real
         "/clips.json": first_file("clips.json") or EMPTY}                                    # real iSign clips (optional)
for k, v in FILES.items():
    print(f"  {k:<22} {v if v else '(not on Drive: skipped)'}")
TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json", ".txt": "text/plain"}

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass
    def _send(self, code_, body, ctype):
        self.send_response(code_); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)
    def _json(self, obj, code_=200):
        self._send(code_, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), "application/json")
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/health":
            return self._json({"ok": True, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", "langs": list(LANGS)})
        f = FILES.get(p)
        if not f:
            return self._send(404, b"not found", "text/plain")
        self.send_response(200); self.send_header("Content-Type", TYPES.get(os.path.splitext(f)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(os.path.getsize(f))); self.end_headers()
        with open(f, "rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                self.wfile.write(b)
    def do_POST(self):
        if self.path.split("?")[0] != "/api/generate":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._json(make_paragraph(req.get("text", ""), req.get("lang", "auto")))
        except Exception as e:
            print("request failed:", repr(e), flush=True)
            torch.cuda.empty_cache()
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"})

class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True; allow_reuse_address = True
PORT = int(os.environ.get("BAHU_PORT", 8000))
try:
    SERVER.shutdown(); SERVER.server_close()          # re-running this cell restarts the server
except NameError:
    pass
SERVER = Server(("0.0.0.0", PORT), Handler)
threading.Thread(target=SERVER.serve_forever, daemon=True).start()
print(f"server running on port {PORT}")
if not os.environ.get("BAHU_ROOT"):
    from google.colab.output import eval_js, serve_kernel_port_as_window
    LINK = eval_js(f"google.colab.kernel.proxyPort({PORT})")
    print("\nOPEN THIS LINK (keep this Colab tab open while presenting):\n  " + LINK +
          "\n(the research viewer with all the switches is at the same link + viewer.html)")
    serve_kernel_port_as_window(PORT, anchor_text="Open BahuSanket (live)")
''')

md("""
### Optional: a public link (only if the link above does not open)

Some browsers block the Colab link (for example strict cookie settings). This cell makes a temporary public `trycloudflare.com` link
instead, with Cloudflare's free quick tunnel (no account needed). Anyone with that link can use the demo while this notebook runs,
so share it with no one and stop it (Runtime → Interrupt) after the presentation. Set `PUBLIC_LINK = True` to use it.
""")
code(r'''
# 12. (optional) public link through a Cloudflare quick tunnel
PUBLIC_LINK = False
if PUBLIC_LINK:
    if not os.path.exists("/content/cloudflared"):
        subprocess.run(["wget", "-q", "-O", "/content/cloudflared",
                        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
        os.chmod("/content/cloudflared", 0o755)
    TUNNEL = subprocess.Popen(["/content/cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://localhost:{PORT}"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in TUNNEL.stdout:
        m_ = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m_:
            print("PUBLIC LINK (temporary):", m_.group(0)); break
    threading.Thread(target=lambda: [None for _ in TUNNEL.stdout], daemon=True).start()   # keep reading its log so it never blocks
else:
    print("not used (set PUBLIC_LINK = True only if the Colab link does not open)")
''')

nb["cells"] = C
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}}
nbf.write(nb, "20_live_demo_v2.ipynb")
print("ok", len(C), "cells")
