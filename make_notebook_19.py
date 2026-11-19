import json
import nbformat as nbf
src = json.load(open("17_pck_backtranslation_v2.ipynb"))["cells"]
S = lambda i: "".join(src[i]["source"])
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# BahuSanket · 19 · Hindi and Kannada paragraphs on the avatar

Until now the model only took **one English sentence** at a time. This notebook adds the two missing pieces of the 30 November goal:

1. **Hindi and Kannada input.** Each Hindi or Kannada sentence is first translated into English with **IndicTrans2**, a free translation
   model made by AI4Bharat for all 22 scheduled Indian languages (Gala et al. 2023, https://arxiv.org/abs/2305.16307). Our text-to-pose
   model then runs on the English exactly as before, so nothing we trained changes. (Translating first is called a *pivot*: English is the
   bridge language. Later, the same idea lets any of the 22 languages drive the avatar.)
2. **Paragraphs.** A paragraph is split into sentences, every sentence is generated with today's default pipeline, and the clips are
   **joined**: the end of one sentence is blended into the start of the next, so the avatar signs one continuous sequence instead of
   jumping between clips. Very long sentences are first cut at a comma or "and/but" into pieces the model is used to.

**Parts.**
* **A** splits the paragraphs and translates everything this notebook needs (saved on Drive, so a re-run skips it).
* **B** picks the joining method fairly: 100 short "paragraphs" made of 3 *consecutive* test sentences from the same video (so they really
  follow each other), each joined 5 ways. Scored: how big the jump at each join is, compared with normal signing speed, and whether each
  sentence still matches its meaning after joining (the notebook-10 matcher, top-1/top-3).
* **C** measures what Hindi and Kannada cost: 512 test sentences go English → Hindi/Kannada → English (IndicTrans2 both ways) and through
  the full pipeline. This is a *pessimistic* check (two translations instead of one), but it needs no human translator.
* **D** makes the avatar file: 2 demo paragraphs written in English, Hindi and Kannada, plus anything you add in cell 10,
  with real depth from notebook 18 → `avatar/clips_paragraph_19.json`.

**Before you run (one time, about 2 minutes).**
* IndicTrans2 asks you to accept its terms first. Log in to huggingface.co, open both pages and click the agree button:
  https://huggingface.co/ai4bharat/indictrans2-indic-en-1B and https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
* Your Hugging Face token is read from Colab's **Secrets** (`HF_TOKEN`, as in notebook 08). If Secrets can't be reached, cell 1 shows a
  hidden box: paste the token there. It is never printed or saved.

**How to run:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all** (about 15–25 minutes; no training).
Paste the last cell's output back to Claude and attach `avatar/clips_paragraph_19.json` from your Drive.
""")

code(S(1).replace("# 1. Setup", "# 1. Setup (+ the HF token from Colab Secrets, needed for IndicTrans2)") + r'''
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
code(S(2))
code(S(3))
code(S(4))
code(S(5))
code(S(6))
code(S(7))
code(r'''
# 8. Notebook 18's depth model (depth_lifter_v1, best pass) and lift(): x, y -> depth of all 123 points
MAX_FRAMES = 300
class DepthNet(nn.Module):
    def __init__(self, n=B.N_POINTS, d=256, layers=4, heads=4, maxT=MAX_FRAMES, dropout=0.1):
        super().__init__()
        for k in ("xm", "xs", "vs"):                                 # input scaling; the real values come from the checkpoint
            self.register_buffer(k, torch.ones(n, 2))
        self.inp = nn.Linear(n * 5, d); self.pos = nn.Embedding(maxT, d)
        self.conv = nn.Conv1d(d, d, 5, padding=2)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True), layers,
                                         enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n))
    def forward(self, xy, pres, pad):
        pm = pres.unsqueeze(-1).float()
        v = torch.zeros_like(xy); ok = (pres[:, 1:] & pres[:, :-1]).unsqueeze(-1)
        v[:, 1:] = (xy[:, 1:] - xy[:, :-1]) * ok / self.vs
        x = torch.cat([(((xy - self.xm) / self.xs) * pm).flatten(2), v.flatten(2).clamp(-10, 10), pres.float()], -1)
        h = self.inp(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        h = h + self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.out(self.enc(h, src_key_padding_mask=pad))

dck = torch.load(os.path.join(ROOT, "checkpoints", "depth_lifter_v1", "best.pt"), map_location=DEVICE, weights_only=False)
dnet = DepthNet().to(DEVICE); dnet.load_state_dict(dck["model"]); dnet.eval()
ZMEAN_T, ZSTD_T = (torch.as_tensor(np.asarray(a, np.float32), device=DEVICE) for a in (dck["zmean"], dck["zstd"]))
print("depth model: pass", dck["epoch"] + 1)

@torch.no_grad()
def lift(xys, press, bs=32, win=MAX_FRAMES, step=200):
    """lists of (T, 123, 2) x, y and (T, 123) tracked flags -> list of (T, 123) depth in shoulder widths.
    Long sequences (a whole paragraph) are read in overlapping 300-frame windows and blended (as in notebook 18)."""
    pieces = []
    for i, x in enumerate(xys):
        T_ = len(x)
        starts = [0] if T_ <= win else list(range(0, T_ - win, step)) + [T_ - win]
        pieces += [(i, a, min(T_, a + win)) for a in starts]
    acc = [np.zeros((len(x), B.N_POINTS), np.float64) for x in xys]
    wsum = [np.zeros((len(x), 1), np.float64) for x in xys]
    for b0 in range(0, len(pieces), bs):
        ch = pieces[b0:b0 + bs]; Tm = max(e - a for _, a, e in ch)
        xy = np.zeros((len(ch), Tm, B.N_POINTS, 2), np.float32); pr = np.zeros((len(ch), Tm, B.N_POINTS), bool); pad = np.ones((len(ch), Tm), bool)
        for k, (i, a, e) in enumerate(ch):
            xy[k, :e - a] = np.asarray(xys[i], np.float32)[a:e, :, :2]; pr[k, :e - a] = np.asarray(press[i])[a:e]; pad[k, :e - a] = False
        xy[~pr] = 0
        t = lambda v: torch.as_tensor(v, device=DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            pred = dnet(t(xy), t(pr), t(pad)).float()
        z = (pred * ZSTD_T + ZMEAN_T).cpu().numpy()
        for k, (i, a, e) in enumerate(ch):
            n_ = e - a; tt = np.arange(n_)
            w = np.minimum(np.minimum(tt + 1, n_ - tt), 50).astype(np.float64)[:, None]
            acc[i][a:e] += z[k, :n_] * w; wsum[i][a:e] += w
    return [(a / np.maximum(w, 1e-9)).astype(np.float32) for a, w in zip(acc, wsum)]
''')

md("""
## Part A · Translation with IndicTrans2 (saved on Drive, so a re-run skips it)

IndicTrans2 is run in a separate small Python process with the library versions it was written for (transformers 4.46), so it cannot
clash with the rest of this notebook. Every translation is stored in `translations19/cache.json`; only new sentences are translated.
Language codes: `hin_Deva` = Hindi, `kan_Knda` = Kannada, `eng_Latn` = English.
""")
code(r'''
# 9. IndicTrans2 setup and translate()
import re, hashlib
IT2_LIB = "/content/it2lib" if not os.environ.get("BAHU_ROOT") else os.path.join(ROOT, "_it2lib")
IT2_SCRIPT = os.path.join(os.path.dirname(IT2_LIB), "it2_translate.py")
INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"            # Hindi/Kannada -> English (the 1B model, best quality)
EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"     # English -> Hindi/Kannada (only for Part C's round trip)
LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda"}
TR_DIR = os.path.join(ROOT, "translations19"); os.makedirs(TR_DIR, exist_ok=True)
TR_CACHE_F = os.path.join(TR_DIR, "cache.json")
TR_CACHE = json.load(open(TR_CACHE_F)) if os.path.exists(TR_CACHE_F) else {}
print("translations already saved:", len(TR_CACHE))

if not os.path.exists(os.path.join(IT2_LIB, "transformers", "__init__.py")):
    t0 = time.time()
    # transformers 4.46 (what IndicTrans2's model code was written for) goes in its own folder; this notebook keeps Colab's own version
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", IT2_LIB, "transformers==4.46.3",
                    "tokenizers==0.20.3", "huggingface_hub==0.26.2", "IndicTransToolkit==1.1.1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacremoses", "indic-nlp-library-itt", "sentencepiece"], check=True)
    print(f"IndicTrans2 libraries installed [{time.time() - t0:.0f}s]")

open(IT2_SCRIPT, "w").write(r"""
import sys, json, time, types
import torch
try:
    from IndicTransToolkit.processor import IndicProcessor
except ImportError:                              # newer transformers break the toolkit's training helpers; we only need the processor
    for n in ("collator", "evaluator"):
        m = types.ModuleType("IndicTransToolkit." + n); m.IndicDataCollator = m.IndicEvaluator = None
        sys.modules["IndicTransToolkit." + n] = m
    from IndicTransToolkit.processor import IndicProcessor
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
jobs = json.load(open(sys.argv[1]))
dev = "cuda" if torch.cuda.is_available() else "cpu"
ip = IndicProcessor(inference=True)
SCRIPT = {"eng_Latn": (0x41, 0x7A), "hin_Deva": (0x900, 0x97F), "kan_Knda": (0xC80, 0xCFF)}

def looks_right(texts, tgt):
    lo, hi = SCRIPT[tgt]; ok = tot = 0
    for t in texts:
        for ch in t:
            if ch.isalpha():
                tot += 1; ok += lo <= ord(ch) <= hi
    return tot == 0 or ok / tot > 0.8

out, models = [], {}
for job in jobs:
    name, src, tgt, sents = job["model"], job["src"], job["tgt"], job["sents"]
    if name not in models:
        models.clear(); torch.cuda.empty_cache()
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True,
                                                    torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
        models[name] = (tok, mdl)
        print(f"  loaded {name} (transformers {transformers.__version__}) [{time.time() - t0:.0f}s]", flush=True)
    tok, mdl = models[name]
    res = [None] * len(sents)
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    for b0 in range(0, len(order), 32):
        ix = order[b0:b0 + 32]
        batch = ip.preprocess_batch([sents[i] for i in ix], src_lang=src, tgt_lang=tgt)
        enc = tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(dev)
        kw = dict(min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
        with torch.no_grad():
            try:
                gen = mdl.generate(**enc, use_cache=True, **kw)
            except Exception:
                gen = mdl.generate(**enc, use_cache=False, **kw)
        dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if not looks_right(dec, tgt) and hasattr(tok, "as_target_tokenizer"):    # older tokenizer code decodes with the target vocabulary only here
            with tok.as_target_tokenizer():
                dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        dec = ip.postprocess_batch(dec, lang=tgt)
        for i, d in zip(ix, dec):
            res[i] = d.strip()
    if not looks_right(res, tgt):
        print("WARNING: output is not in the expected script:", res[:3], flush=True)
    out.append(res)
    print(f"  {src} -> {tgt}: {len(sents)} sentences done", flush=True)
json.dump(out, open(sys.argv[2], "w"), ensure_ascii=False)
""")

def _key(model, src, tgt, s):
    return hashlib.md5(f"{model}|{src}|{tgt}|{s}".encode("utf-8")).hexdigest()

def translate_many(requests_):
    """requests_: list of (model, src, tgt, [sentences]). Translates what is not cached, saves the cache, returns lists of translations."""
    jobs = []
    for model_, s_, t_, sents in requests_:
        todo = sorted({x for x in sents if _key(model_, s_, t_, x) not in TR_CACHE})
        if todo:
            jobs.append({"model": model_, "src": s_, "tgt": t_, "sents": todo})
    jobs.sort(key=lambda j: j["model"])                      # each model is loaded once
    if jobs:
        print(f"translating {sum(len(j['sents']) for j in jobs)} new sentences ...", flush=True)
        jf, of = os.path.join(TR_DIR, "_jobs.json"), os.path.join(TR_DIR, "_out.json")
        json.dump(jobs, open(jf, "w"), ensure_ascii=False)
        ef = os.path.join(os.path.dirname(IT2_LIB), "it2_err.txt")
        def run_it2(pinned):
            env = dict(os.environ)
            if pinned:                                       # transformers 4.46 from IT2_LIB; else Colab's own transformers
                env["PYTHONPATH"] = IT2_LIB + os.pathsep + env.get("PYTHONPATH", "")
            with open(ef, "w") as errf:
                pr_ = subprocess.Popen([sys.executable, IT2_SCRIPT, jf, of], env=env, stdout=subprocess.PIPE, stderr=errf, text=True)
                for line in pr_.stdout:
                    print(line, end="", flush=True)
                rc = pr_.wait()
            return rc, open(ef).read()[-4000:]
        rc, err = run_it2(True)
        gated = lambda e: any(k in e.lower() for k in ("gated", "restricted", "403", "401", "awaiting a review"))
        if rc != 0 and not gated(err):
            print("IndicTrans2 with transformers 4.46 failed; trying Colab's own transformers ...")
            rc, err2 = run_it2(False)
            err = err if rc == 0 else err + "\n\n----- second try -----\n" + err2
        if rc != 0:
            if gated(err):
                raise RuntimeError("Hugging Face refused the IndicTrans2 download. Log in to huggingface.co, open "
                                   f"https://huggingface.co/{INDIC_EN} and https://huggingface.co/{EN_INDIC}, click agree on both, "
                                   "check HF_TOKEN in Colab Secrets, then run this cell again.\n\n" + err[-1200:])
            raise RuntimeError("IndicTrans2 failed; paste this to Claude:\n" + err[-6000:])
        for j, res in zip(jobs, json.load(open(of))):
            for x, y in zip(j["sents"], res):
                TR_CACHE[_key(j["model"], j["src"], j["tgt"], x)] = y
        json.dump(TR_CACHE, open(TR_CACHE_F + ".tmp", "w"), ensure_ascii=False); os.replace(TR_CACHE_F + ".tmp", TR_CACHE_F)
    return [[TR_CACHE[_key(m_, s_, t_, x)] for x in sents] for m_, s_, t_, sents in requests_]

print("translate() ready")
''')
code(r'''
# 10. Paragraphs: split into sentences, translate (+ Part C's round trip), cut very long sentences
# ---- ADD YOUR OWN PARAGRAPHS HERE (language "en", "hi" or "kn"); they go into the avatar file too ----
YOUR_PARAGRAPHS = [
    # ("hi", "यहाँ अपना हिंदी अनुच्छेद लिखें। दूसरा वाक्य।"),
]
DEMO = [
    ("rain", "en", "Heavy rain is expected in Bengaluru tomorrow. Schools will remain closed. "
                   "People should stay at home and drink clean water."),
    ("rain", "hi", "कल बेंगलुरु में भारी बारिश की संभावना है। स्कूल बंद रहेंगे। लोगों को घर पर रहना चाहिए और साफ पानी पीना चाहिए।"),
    ("rain", "kn", "ನಾಳೆ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಭಾರಿ ಮಳೆಯಾಗುವ ಸಾಧ್ಯತೆ ಇದೆ. ಶಾಲೆಗಳು ಮುಚ್ಚಿರುತ್ತವೆ. ಜನರು ಮನೆಯಲ್ಲೇ ಇರಬೇಕು ಮತ್ತು ಶುದ್ಧ ನೀರನ್ನು ಕುಡಿಯಬೇಕು."),
    ("scheme", "en", "The government has started a new scheme for deaf students. Every student will get free books and a laptop. "
                     "The scheme will begin next month."),
    ("scheme", "hi", "सरकार ने बधिर छात्रों के लिए एक नई योजना शुरू की है। हर छात्र को मुफ्त किताबें और एक लैपटॉप मिलेगा। "
                     "यह योजना अगले महीने से शुरू होगी।"),
    ("scheme", "kn", "ಸರ್ಕಾರವು ಕಿವುಡ ವಿದ್ಯಾರ್ಥಿಗಳಿಗಾಗಿ ಹೊಸ ಯೋಜನೆಯನ್ನು ಆರಂಭಿಸಿದೆ. ಪ್ರತಿಯೊಬ್ಬ ವಿದ್ಯಾರ್ಥಿಗೂ ಉಚಿತ ಪುಸ್ತಕಗಳು ಮತ್ತು ಒಂದು ಲ್ಯಾಪ್‌ಟಾಪ್ ಸಿಗಲಿದೆ. "
                     "ಈ ಯೋಜನೆ ಮುಂದಿನ ತಿಂಗಳಿನಿಂದ ಆರಂಭವಾಗಲಿದೆ."),
]
PARAS = [{"id": f"{t}_{l}", "lang": l, "source": s.strip()} for t, l, s in DEMO]
PARAS += [{"id": f"yours{k + 1}_{l}", "lang": l, "source": s.strip()} for k, (l, s) in enumerate(YOUR_PARAGRAPHS)]
assert all(p["lang"] in LANGS for p in PARAS), "language must be 'en', 'hi' or 'kn'"

def split_sentences(text):
    """Split after . ! ? and the Hindi danda । (and ॥), when a space or the end follows (so 3.5 or U.S.A. stay whole)."""
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+|(?<=[।॥])\s*", text)
    return [p.strip() for p in parts if p and re.search(r"\w", p)]

# how long are the sentences our model was trained on? (cut longer English sentences into pieces of about that size)
nw = np.array([len(t.split()) for t in texts["train"]])
MAX_WORDS = int(np.clip(np.percentile(nw, 90), 10, 22))
print(f"training sentences: median {np.median(nw):.0f} words, 90% have <= {np.percentile(nw, 90):.0f} -> pieces of at most {MAX_WORDS} words")
JOINERS = {"and", "but", "because", "so", "while", "which", "who", "where", "when", "after", "before", "or", "then", "although"}
def chunk_english(s, max_words=None):
    max_words = max_words or MAX_WORDS
    w = s.split()
    if len(w) <= max_words:
        return [s]
    cands = [i for i in range(3, len(w) - 2) if w[i - 1][-1] in ",;:" or w[i].lower() in JOINERS]
    cut = min(cands, key=lambda i: abs(i - len(w) / 2)) if cands else len(w) // 2
    return chunk_english(" ".join(w[:cut]).rstrip(",;:"), max_words) + chunk_english(" ".join(w[cut:]), max_words)

for p in PARAS:
    p["src_sents"] = split_sentences(p["source"])
by_lang = {}
for p in PARAS:
    by_lang.setdefault(p["lang"], []).extend(p["src_sents"])

# Part C's 512 test sentences (English -> Hindi/Kannada first), so that each translation model is loaded only once
RT_N = 512 if not QUICK else 64
cand = [i for i in range(len(texts["test"])) if len(texts["test"][i].split()) >= 3]
ridx = sorted(np.random.RandomState(12).choice(cand, min(RT_N, len(cand)), replace=False).tolist())
orig = [texts["test"][i].strip() for i in ridx]
t0 = time.time()
to_hi, to_kn = translate_many([(EN_INDIC, "eng_Latn", "hin_Deva", orig), (EN_INDIC, "eng_Latn", "kan_Knda", orig)])
ind = [l for l in by_lang if l != "en"]
res = translate_many([(INDIC_EN, LANGS[l], "eng_Latn", by_lang[l]) for l in ind] +
                     [(INDIC_EN, "hin_Deva", "eng_Latn", to_hi), (INDIC_EN, "kan_Knda", "eng_Latn", to_kn)])
back_hi, back_kn = res[-2], res[-1]
EN_OF = {("en", s): s for s in by_lang.get("en", [])}
for l, tr in zip(ind, res[:len(ind)]):
    EN_OF.update({(l, s): e for s, e in zip(by_lang[l], tr)})
print(f"all translations ready [{time.time() - t0:.0f}s]")
assert all(re.search("[A-Za-z]", e) for e in EN_OF.values()), "IndicTrans2 did not return English: tell Claude"
for p in PARAS:
    p["segments"] = []
    for s in p["src_sents"]:
        en = EN_OF[(p["lang"], s)]
        for piece in chunk_english(en):
            p["segments"].append({"src": s, "en": piece})
    print(f"\n[{p['id']}] {len(p['src_sents'])} sentences -> {len(p['segments'])} pieces")
    for g in p["segments"]:
        print(f"   {g['src'][:60]:<60} -> {g['en']}")
''')

md("""
## Part B · Joining sentences, chosen on real test paragraphs

The five ways to join two clips (25 frames = 1 second):
* **cut**: play one clip after the other (today's behaviour on the avatar, a jump at every join);
* **crossfade 6 / 10**: the last 6 (or 10) frames of one sentence and the first 6 (or 10) of the next are mixed, sliding smoothly from one
  to the other (an S-shaped "smoothstep" curve), so the sentence boundary becomes a short shared transition;
* **ease 8 / 16**: nothing is mixed; instead the next clip starts exactly where the last one ended, and that offset fades out over 8 (or 16)
  frames. This is a simple form of *inertialization* from game animation (Bollo 2016, GDC talk "Inertialization: High-Performance
  Animation Transitions in Gears of War", https://www.gdcvault.com/play/1025165/).

These are hand-made joins. Learning real co-articulation between signs (e.g. Saunders et al. 2022, https://arxiv.org/abs/2203.15354)
is on the December plan; today's joins are the baseline it will be compared against.

**Scores.** *Join jump*: at each join, the fastest arm-and-hand movement in the 5 frames around it, divided by that paragraph's typical
movement per frame (1.0 = no visible jump; real signing's fast moments reach about 2–3). *Visible jumps*: share of joins above 3.
*Top-1 / top-3*: each sentence is cut back out of the joined paragraph and the matcher checks it still means that sentence.
**Rule:** the join with the smallest jump whose top-3 stays within 2 points of the separate clips.
""")
code(r'''
# 11. Join methods and the test paragraphs
bd = LAY["body"]; ARM_HAND = [bd[k] for k in ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]] + \
     list(range(LAY["parts"]["hands"].start, LAY["parts"]["hands"].stop))

def smooth01(n):
    t = (np.arange(n) + 1) / (n + 1)
    return t * t * (3 - 2 * t)

def join(clips, method):
    """clips: list of (T, 123, 2). Returns the joined (T, 123, 2) and each sentence's [start, end) frames in it."""
    clips = [np.asarray(c, np.float32)[..., :2] for c in clips]
    out, bounds = clips[0].copy(), [[0, len(clips[0])]]
    for c in clips[1:]:
        if method == "cut":
            bounds.append([len(out), len(out) + len(c)]); out = np.concatenate([out, c])
        elif method.startswith("crossfade"):
            k = min(int(method.split()[1]), len(out) // 3, len(c) // 3)
            if k < 1:
                bounds.append([len(out), len(out) + len(c)]); out = np.concatenate([out, c]); continue
            w = smooth01(k)[:, None, None]
            mid = (1 - w) * out[-k:] + w * c[:k]
            s0 = len(out) - k
            bounds[-1][1] = s0 + k // 2
            out = np.concatenate([out[:-k], mid, c[k:]])
            bounds.append([s0 + k // 2, len(out)])
        elif method.startswith("ease"):
            m = min(int(method.split()[1]), max(1, len(c) // 2))
            c2 = c.copy()
            c2[:m] += (out[-1] - c[0])[None] * (1 - smooth01(m))[:, None, None]
            bounds.append([len(out), len(out) + len(c2)]); out = np.concatenate([out, c2])
        else:
            raise ValueError(method)
    return out, bounds

def join_jumps(seq, bounds, pres=None):
    """For each join: the fastest arm/hand movement within 2 frames of it, divided by the median movement per frame of the whole paragraph."""
    x = np.asarray(seq, np.float32)[:, ARM_HAND]
    ok = np.ones(x.shape[:2], bool) if pres is None else np.asarray(pres)[:, ARM_HAND]
    okv = ok[1:] & ok[:-1]
    d = np.linalg.norm(x[1:] - x[:-1], axis=-1)
    sp = (d * okv).sum(1) / np.maximum(okv.sum(1), 1)                     # speed from frame t to t+1
    typ = max(float(np.median(sp)), 1e-4)
    return [float(sp[max(0, b0 - 3):b0 + 2].max()) / typ for b0, _ in bounds[1:]]

JOINS = ["cut", "crossfade 6", "crossfade 10", "ease 8", "ease 16"]
# 3 consecutive test clips from the same video = a real little paragraph
vid_of = lambda u: str(u).rsplit("-", 1)
pos = {}
for i, u in enumerate(test["uids"]):
    v, k = vid_of(u)
    if k.isdigit():
        pos[(v, int(k))] = i
runs, used = [], set()
for (v, k) in sorted(pos):
    trio = [(v, k), (v, k + 1), (v, k + 2)]
    if all(t in pos for t in trio) and not any(t in used for t in trio):
        runs.append([pos[t] for t in trio]); used.update(trio)
runs = [runs[j] for j in sorted(np.random.RandomState(19).permutation(len(runs))[:100 if not QUICK else 12])]
flat_idx = [i for r in runs for i in r]
print(f"{len(runs)} test paragraphs of 3 consecutive sentences ({len(flat_idx)} sentences)")

t0 = time.time()
gen_flat = default_pipeline(data_test, flat_idx, [texts["test"][i] for i in flat_idx], seed=19)
GEN = {i: g for i, g in zip(flat_idx, gen_flat)}
print(f"generated [{time.time() - t0:.0f}s]")
TE_F = embed_sentences([texts["test"][i] for i in flat_idx])
REAL_F = embed_motion([test["poses"][i].astype(np.float32)[..., :2] for i in flat_idx])

JROWS = {}
# reference: the REAL clips glued end to end (they come from one continuous video, with small gaps between subtitle lines)
jr = []
for r in runs:
    seq = np.concatenate([test["poses"][i].astype(np.float32)[..., :2] for i in r]); pr = np.concatenate([test["masks"][i] for i in r])
    L_ = np.cumsum([0] + [len(test["poses"][i]) for i in r]); jr += join_jumps(seq, [[a, b] for a, b in zip(L_[:-1], L_[1:])], pr)
JROWS["real signing, clips glued"] = {"jump": float(np.median(jr)), "visible": float(np.mean(np.array(jr) > 3))}
JROWS["our model, separate clips"] = scores([GEN[i] for i in flat_idx], TE_F, REAL_F)
for mth in JOINS:
    jj, segs = [], []
    for r in runs:
        seq, bnd = join([GEN[i] for i in r], mth)
        jj += join_jumps(seq, bnd); segs += [seq[a:b] for a, b in bnd]
    JROWS[mth] = {**scores(segs, TE_F, REAL_F), "jump": float(np.median(jj)), "visible": float(np.mean(np.array(jj) > 3))}
    r_ = JROWS[mth]
    print(f"  {mth:<14} join jump x{r_['jump']:.2f} | visible jumps {r_['visible']:.0%} | top1/top3 {r_['top1']:.1%}/{r_['top3']:.1%} | FID {r_['fid']:.3f}")
floor3 = JROWS["our model, separate clips"]["top3"] - 0.02
ok_ = [m for m in JOINS[1:] if JROWS[m]["top3"] >= floor3]
JOIN = min(ok_, key=lambda m: JROWS[m]["jump"]) if ok_ else "cut"
print("CHOSEN join:", JOIN)
''')

md("""
## Part C · What do Hindi and Kannada cost? (512 test sentences, English → Hindi/Kannada → English)

Scores: **chrF** (letter-level overlap with the original English, 0–100, https://aclanthology.org/W15-3049/), **meaning similarity**
(MiniLM cosine, 1 = same meaning), and the full pipeline's **top-1 / top-3** where the matcher judges the signing against the
*original* English sentence. "wrong sentence" is the floor.
""")
code(r'''
# 12. Round trip through Hindi and Kannada
import sacrebleu    # (the translations were made in cell 10)

def sim(a, b):
    ea = st.encode(a, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    eb = st.encode(b, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return float((ea * eb).sum(1).mean())

TE_R = embed_sentences(orig); REAL_R = embed_motion([test["poses"][i].astype(np.float32)[..., :2] for i in ridx])
RROWS = {}
wrong = [orig[j] for j in np.random.RandomState(0).permutation(len(orig))]
for name, tx in [("English (original)", orig), ("via Hindi", back_hi), ("via Kannada", back_kn), ("wrong sentence", wrong)]:
    d_ = T.CodeTextData(tx, [np.zeros(2, np.int64)] * len(tx), tokenizer, CFG["max_codes"])
    poses = default_pipeline(d_, list(range(len(tx))), tx, seed=12)
    RROWS[name] = {**scores(poses, TE_R, REAL_R), "chrf": float(sacrebleu.corpus_chrf(tx, [orig]).score), "sim": sim(tx, orig),
                   "same": float(np.mean([a.lower().strip(" .") == b.lower().strip(" .") for a, b in zip(tx, orig)]))}
    r_ = RROWS[name]
    print(f"  {name:<20} chrF {r_['chrf']:5.1f} | meaning sim {r_['sim']:.3f} | identical {r_['same']:.0%} | top1/top3 {r_['top1']:.1%}/{r_['top3']:.1%} | FID {r_['fid']:.3f}")
RT_EXAMPLES = [{"en": orig[k], "hi": to_hi[k], "hi_back": back_hi[k], "kn": to_kn[k], "kn_back": back_kn[k]} for k in range(8)]
''')

md("""
## Part D · The avatar file

Every paragraph is generated sentence by sentence with the default pipeline, joined with the chosen method (and, for comparison, with
a plain cut), given depth by notebook 18's depth model *over the whole paragraph at once*, and saved to `avatar/clips_paragraph_19.json`.
""")
code(r'''
# 13. Generate, join, add depth, save avatar/clips_paragraph_19.json
seg_en = [g["en"] for p in PARAS for g in p["segments"]]
d_ = T.CodeTextData(seg_en, [np.zeros(2, np.int64)] * len(seg_en), tokenizer, CFG["max_codes"])
t0 = time.time()
seg_poses = default_pipeline(d_, list(range(len(seg_en))), seg_en, seed=19)
print(f"{len(seg_en)} pieces generated [{time.time() - t0:.0f}s]")

def frames(p):
    return [[round(float(v), 3) for v in row] for row in np.asarray(p, np.float32).reshape(len(p), -1)]
def zframes(z):
    return [[round(float(v), 3) for v in row] for row in np.asarray(z, np.float32)]

k0, clips_out, PSTAT = 0, [], []
for p in PARAS:
    parts = seg_poses[k0:k0 + len(p["segments"])]; k0 += len(p["segments"])
    variants = {}
    for mth in ([JOIN, "cut"] if JOIN != "cut" else ["cut"]):
        seq, bnd = join(parts, mth)
        z = lift([seq], [np.ones(seq.shape[:2], bool)])[0]
        variants[mth] = {"n": len(seq), "pts": frames(seq), "z": zframes(z),
                         "segments": [{**g, "start": int(a), "end": int(b)} for g, (a, b) in zip(p["segments"], bnd)],
                         "jumps": [round(v, 2) for v in join_jumps(seq, bnd)]}
    main = variants[JOIN]
    clips_out.append({"uid": p["id"], "kind": "paragraph", "lang": p["lang"], "source": p["source"],
                      "text": " ".join(g["en"] for g in p["segments"]), "join": JOIN, **main,
                      "cut": variants["cut"] if JOIN != "cut" else None})
    PSTAT.append((p["id"], len(p["segments"]), main["n"] / 25, variants["cut"]["jumps"], main["jumps"]))
out = {"format": "bahusanket-v2-123-z-paragraph",
       "note": ("x, y in shoulder widths from the shoulder midpoint, y down; z in shoulder widths, negative = toward the camera "
                "(depth model, notebook 18). Each clip is one paragraph; segments give each sentence's frames [start, end). "
                "'cut' holds the same paragraph with plain cuts, for comparison."),
       "point_idx": [int(v) for v in POINT_IDX], "fps": 25,
       "pipeline": f"IndicTrans2 ({INDIC_EN}) -> t2c_more best_loss s=4 + window 0.8-1.25 + rtrans_123pts_b random T=1 + bone fix -> join {JOIN} -> depth_lifter_v1",
       "clips": clips_out}
AV_DIR = os.path.join(ROOT, "avatar"); os.makedirs(AV_DIR, exist_ok=True)
AV_OUT = os.path.join(AV_DIR, "clips_paragraph_19.json")
json.dump(out, open(AV_OUT + ".tmp", "w"), separators=(",", ":"), ensure_ascii=False); os.replace(AV_OUT + ".tmp", AV_OUT)
print(f"saved {AV_OUT} ({os.path.getsize(AV_OUT) / 1e6:.1f} MB, {len(clips_out)} paragraphs)")
''')

code(r'''
# 14. Save the results and print the paste block
RES = {"max_words": MAX_WORDS, "join_rows": JROWS, "join_chosen": JOIN, "n_test_paragraphs": len(runs), "roundtrip_rows": RROWS,
       "roundtrip_examples": RT_EXAMPLES, "paragraphs": [{"id": p["id"], "lang": p["lang"], "segments": p["segments"]} for p in PARAS],
       "models": {"indic_en": INDIC_EN, "en_indic": EN_INDIC}}
json.dump(RES, open(os.path.join(RESULTS_DIR, "19_paragraphs.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("\n===== PASTE EVERYTHING BELOW THIS LINE BACK TO CLAUDE =====")
print(f"SENTENCE PIECES: at most {MAX_WORDS} words")
print(f"JOINS ({len(runs)} test paragraphs x 3 sentences)   join jump | visible | top1 / top3 | FID")
for k, r in JROWS.items():
    j = f"x{r['jump']:.2f}" if "jump" in r else "  -  "; v = f"{r['visible']:.0%}" if "visible" in r else " - "
    t = f"{r['top1']:.1%} / {r['top3']:.1%} | {r['fid']:.3f}" if "top1" in r else "-"
    print(f"  {k:<28} {j:>6} | {v:>4} | {t}")
print("chosen:", JOIN)
print(f"ROUND TRIP ({len(ridx)} test sentences)   chrF | meaning sim | identical | top1 / top3 | FID")
for k, r in RROWS.items():
    print(f"  {k:<20} {r['chrf']:5.1f} | {r['sim']:.3f} | {r['same']:>4.0%} | {r['top1']:.1%} / {r['top3']:.1%} | {r['fid']:.3f}")
print("ROUND TRIP EXAMPLES:")
for e in RT_EXAMPLES[:4]:
    print(f"  EN {e['en']}\n     HI {e['hi']} -> {e['hi_back']}\n     KN {e['kn']} -> {e['kn_back']}")
print("AVATAR PARAGRAPHS (id | pieces | seconds | jumps with cut -> with join):")
for pid, n, s, jc, jn in PSTAT:
    print(f"  {pid:<12} {n} | {s:.1f}s | {jc} -> {jn}")
for p in PARAS:
    print(f"  [{p['id']}] " + " / ".join(g["en"] for g in p["segments"]))
print("files: translations19/cache.json, results/19_paragraphs.json, avatar/clips_paragraph_19.json (please attach it)")
''')

nb["cells"] = C
nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}}
nbf.write(nb, "19_paragraphs_v2.ipynb")
print("ok", len(C), "cells")
