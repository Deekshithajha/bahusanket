"""BahuSanket live pipeline on a laptop: text (English / Hindi / Kannada) -> avatar clip.
The same steps and models as notebooks 16-19 (default pipeline, ease-8 joins, depth model), loaded from the bundle folder."""
import hashlib, json, os, re, sys, threading, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.environ.get("BAHU_BUNDLE", os.path.join(HERE, "bundle"))
MODELS = os.environ.get("BAHU_MODELS", os.path.join(HERE, "models"))           # Hugging Face models, downloaded by setup.sh
sys.path.insert(0, os.path.join(BUNDLE, "code"))

import bahusanket_lib as B
import bahusanket_vq as V
import bahusanket_t2c as T
import bahusanket_layout as L
import bahusanket_rvq as RVQ
import bahusanket_rtrans as RT

POINT_IDX = json.load(open(os.path.join(BUNDLE, "model_point_indices_v2.json")))
LAY = L.apply(POINT_IDX, B, V, T, RVQ, RT)


def pick_device():
    if os.environ.get("BAHU_DEVICE"):
        return os.environ["BAHU_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"                                        # Apple M-series graphics chip
    return "cpu"


DEVICE = pick_device()
HF = {"distilbert": os.path.join(MODELS, "distilbert-base-uncased"),
      "minilm": os.path.join(MODELS, "all-MiniLM-L6-v2"),
      "indic_en": os.path.join(MODELS, "indictrans2-indic-en-1B")}
CK = os.path.join(BUNDLE, "checkpoints")


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_without_bert(module, state):
    """Checkpoints were packed without the frozen DistilBERT weights (identical to the downloaded model)."""
    missing, unexpected = module.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith("bert.")] + list(unexpected)
    assert not bad, f"checkpoint does not match the model: {bad[:5]}"


class Signer:
    def __init__(self, log=print):
        t0 = time.time()
        self.log = log
        from transformers import AutoTokenizer, AutoModel
        import transformers.models.auto.modeling_auto  # noqa: F401
        # the model code asks for "distilbert-base-uncased" by name: point that name at the local copy
        _orig = AutoModel.from_pretrained.__func__
        AutoModel.from_pretrained = classmethod(lambda cls, name, *a, **k: _orig(cls, HF["distilbert"] if name == "distilbert-base-uncased" else name, *a, **k))
        self.tokenizer = AutoTokenizer.from_pretrained(HF["distilbert"])

        ck_vq = _load(os.path.join(CK, "vq_v2_123pts.pt")); vc = ck_vq["config"]
        vq = V.PoseVQVAE(in_dim=B.N_POINTS * 2, width=vc["width"], code_dim=vc["code_dim"], n_codes=vc["n_codes"])
        vq.load_state_dict(ck_vq["model"]); vq.eval()
        ck_rv = _load(os.path.join(CK, "rvq_v2_123pts.pt")); rvc = ck_rv["config"]
        self.rvq = RVQ.ResidualPoseVQ(vq, n_res=rvc["n_res"], n_codes=rvc["n_codes"])
        self.rvq.load_state_dict(ck_rv["model"]); self.rvq.to(DEVICE).eval()

        ckp = _load(os.path.join(CK, "rtrans_123pts_b.pt")); pr = ckp["config"]
        self.pred = RT.ResidualCodePredictor(n_layers=1 + rvc["n_res"], n_codes=rvc["n_codes"], d_model=pr["d_model"], n_heads=pr["n_heads"],
                                             n_blocks=pr["n_blocks"], ff=pr["ff"], dropout=pr["dropout"], max_len=pr["max_len"])
        _load_without_bert(self.pred, ckp["model"]); self.pred.to(DEVICE).eval()

        ck = _load(os.path.join(CK, "t2c_more_123pts.pt")); self.cfg = ck["config"]; c = self.cfg
        self.model = T.Text2Codes(n_codes=c["n_codes"], d_model=c["d_model"], n_heads=c["n_heads"], n_enc=c["n_enc"], n_dec=c["n_dec"],
                                  ff=c["ff"], dropout=c["dropout"], max_codes=c["max_codes"], text_encoder=c["text_encoder"])
        _load_without_bert(self.model, ck["model"]); self.model.to(DEVICE).eval()
        e = self.tokenizer([""], padding="max_length", truncation=True, max_length=B.MAX_TOKENS, return_tensors="np")
        self.EMPTY_IDS, self.EMPTY_ATTN = e["input_ids"][0].astype(np.int64), e["attention_mask"][0].astype(np.int64)
        self.S_GUIDE = 4.0

        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer(HF["minilm"], device=DEVICE)
        lck = _load(os.path.join(CK, "length_predictor_v1.pt"))
        self.mu_x, self.sd_x = lck["mu"], lck["sd"]
        self.lenp = nn.Sequential(nn.Linear(len(self.mu_x), 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.lenp.load_state_dict(lck["model"]); self.lenp.to(DEVICE).eval()

        lim = json.load(open(os.path.join(BUNDLE, "12_bone_limits.json")))
        self.EDGES = [(b["from"], b["to"], b["name"]) for b in lim["bones"]]
        self.LIMIT = np.array([b["max"] for b in lim["bones"]], np.float32)
        self.SH_LIMIT = lim["shoulder_width_max"]
        bd = L.layout(POINT_IDX)["body"]; self.SH = (bd["l_shoulder"], bd["r_shoulder"])

        dck = _load(os.path.join(CK, "depth_lifter_v1.pt"))
        self.dnet = DepthNet(); self.dnet.load_state_dict(dck["model"]); self.dnet.to(DEVICE).eval()
        self.ZM = torch.as_tensor(np.asarray(dck["zmean"], np.float32), device=DEVICE)
        self.ZS = torch.as_tensor(np.asarray(dck["zstd"], np.float32), device=DEVICE)

        bd = LAY["body"]
        self.ARM_HAND = [bd[k] for k in ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]] + \
            list(range(LAY["parts"]["hands"].start, LAY["parts"]["hands"].stop))
        self.translator = None
        self.lock = threading.Lock()
        self.seen = {}
        self.n = 0
        log(f"BahuSanket models loaded on {DEVICE} [{time.time() - t0:.0f}s]")

    # ---------- notebook 16: length predictor + text model with guidance ----------
    @torch.no_grad()
    def predict_frames(self, tx):
        e = self.st.encode(list(tx), convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        w = np.array([[len(t.split()), len(t)] for t in tx], np.float32)
        x = torch.as_tensor((np.concatenate([e, np.log1p(w)], 1) - self.mu_x) / self.sd_x, dtype=torch.float32, device=DEVICE)
        return np.clip(np.exp(self.lenp(x)[:, 0].cpu().numpy()), 8, 300)

    @torch.no_grad()
    def generate_len(self, d, idx, s, lo, hi, temperature=1.0):
        m = self.model; n = len(idx); ch = np.asarray(idx)
        lo_t = torch.as_tensor(lo, device=DEVICE); hi_t = torch.as_tensor(np.minimum(hi, m.max_codes - 1), device=DEVICE)
        ids = torch.as_tensor(np.concatenate([d.ids[ch], np.repeat(self.EMPTY_IDS[None], n, 0)]), device=DEVICE)
        attn = torch.as_tensor(np.concatenate([d.attn[ch], np.repeat(self.EMPTY_ATTN[None], n, 0)]), device=DEVICE)
        mem, mem_pad = m.encode_text(ids, attn)
        seq = torch.full((2 * n, 1), m.BOS, dtype=torch.long, device=DEVICE)
        done = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        lengths = torch.full((n,), m.max_codes, dtype=torch.long, device=DEVICE)
        for t in range(m.max_codes):
            lg = m.decode_step(seq, mem, mem_pad)[:, -1].float()
            lg = lg[n:] + s * (lg[:n] - lg[n:])
            lg[t < lo_t, m.END] = float("-inf")
            force = t >= hi_t
            lg[force] = float("-inf"); lg[force, m.END] = 0.0
            nxt = torch.multinomial(F.softmax(lg / temperature, -1), 1)[:, 0]
            ended = (nxt == m.END) & ~done
            lengths[ended] = t; done |= ended
            nxt = nxt.masked_fill(done, m.PAD)
            seq = torch.cat([seq, torch.cat([nxt, nxt])[:, None]], 1)
            if bool(done.all()):
                break
        sq = seq[:n, 1:].cpu().numpy()
        return [sq[k, :max(1, int(lengths[k]))].astype(np.int64) for k in range(n)]

    def fix_bones(self, p):
        p = np.asarray(p, np.float32)[..., :2]
        q = p.copy(); valid = np.abs(p).sum(-1) > 0
        shift = np.zeros_like(p)
        a, b = self.SH
        ok = valid[:, a] & valid[:, b]
        v = p[:, b] - p[:, a]; d = np.linalg.norm(v, axis=1)
        sc = np.where(ok & (d > self.SH_LIMIT), self.SH_LIMIT / np.maximum(d, 1e-6), 1.0)[:, None]
        mid = (p[:, a] + p[:, b]) / 2
        q[:, a] = np.where(ok[:, None], mid - v / 2 * sc, p[:, a]); q[:, b] = np.where(ok[:, None], mid + v / 2 * sc, p[:, b])
        shift[:, a], shift[:, b] = q[:, a] - p[:, a], q[:, b] - p[:, b]
        for (a, b, _), lim_ in zip(self.EDGES, self.LIMIT):
            v = p[:, b] - p[:, a]; d = np.linalg.norm(v, axis=1)
            sc = np.minimum(1.0, lim_ / np.maximum(d, 1e-6))[:, None]
            new = np.where(valid[:, [a]], q[:, a] + v * sc, p[:, b] + shift[:, a])
            q[:, b] = np.where(valid[:, [b]], new, p[:, b])
            shift[:, b] = q[:, b] - p[:, b]
        return q

    @torch.no_grad()
    def default_pipeline(self, sentences, seed=19):
        torch.manual_seed(seed)
        d = T.CodeTextData(sentences, [np.zeros(2, np.int64)] * len(sentences), self.tokenizer, self.cfg["max_codes"])
        idx = list(range(len(sentences)))
        Pc = np.maximum(2, np.round(self.predict_frames(sentences) / 4)).astype(int)
        base = self.generate_len(d, idx, self.S_GUIDE, np.floor(Pc * 0.8).astype(int), np.ceil(Pc * 1.25).astype(int))
        full = self.pred.fill(torch.as_tensor(d.ids, device=DEVICE), torch.as_tensor(d.attn, device=DEVICE), base, 1.0)
        return [self.fix_bones(p) for p in RVQ.codes_to_poses_rvq(self.rvq, full, DEVICE)]

    # ---------- notebook 18: depth ----------
    @torch.no_grad()
    def lift(self, x, win=300, step=200):
        T_ = len(x)
        starts = [0] if T_ <= win else list(range(0, T_ - win, step)) + [T_ - win]
        acc = np.zeros((T_, B.N_POINTS)); ws = np.zeros((T_, 1))
        for a in starts:
            e = min(T_, a + win)
            xy = torch.as_tensor(np.asarray(x, np.float32)[None, a:e, :, :2], device=DEVICE)
            pr = torch.ones(xy.shape[:3], dtype=torch.bool, device=DEVICE); pad = torch.zeros(xy.shape[:2], dtype=torch.bool, device=DEVICE)
            z = (self.dnet(xy, pr, pad).float() * self.ZS + self.ZM)[0].cpu().numpy()
            n_ = e - a; tt = np.arange(n_)
            w = np.minimum(np.minimum(tt + 1, n_ - tt), 50).astype(np.float64)[:, None]
            acc[a:e] += z * w; ws[a:e] += w
        return (acc / np.maximum(ws, 1e-9)).astype(np.float32)

    # ---------- notebook 19: joins ----------
    def join(self, clips, method):
        clips = [np.asarray(c, np.float32)[..., :2] for c in clips]
        out, bounds = clips[0].copy(), [[0, len(clips[0])]]
        for c in clips[1:]:
            if method.startswith("ease"):
                m = min(int(method.split()[1]), max(1, len(c) // 2))
                c = c.copy(); c[:m] += (out[-1] - c[0])[None] * (1 - smooth01(m))[:, None, None]
            bounds.append([len(out), len(out) + len(c)]); out = np.concatenate([out, c])
        return out, bounds

    def join_jumps(self, seq, bounds):
        x = np.asarray(seq, np.float32)[:, self.ARM_HAND]
        sp = np.linalg.norm(x[1:] - x[:-1], axis=-1).mean(1)
        typ = max(float(np.median(sp)), 1e-4)
        return [round(float(sp[max(0, b0 - 3):b0 + 2].max()) / typ, 2) for b0, _ in bounds[1:]]

    # ---------- the whole thing ----------
    def make_paragraph(self, text, lang="auto"):
        text = re.sub(r"\s+", " ", (text or "").strip())[:1500]
        if not text:
            raise ValueError("empty text")
        lang = detect_lang(text) if lang == "auto" else lang
        if lang not in LANGS:
            raise ValueError("language must be en, hi or kn")
        if (lang, text) in self.seen:
            return {**self.seen[(lang, text)], "timing": {"cached": True}}
        with self.lock:
            t0 = time.time()
            sents = split_sentences(text) or [text]
            en = sents if lang == "en" else self.get_translator().translate(sents, lang)
            segs = [{"src": s, "en": piece} for s, e in zip(sents, en) for piece in chunk_english(e)][:MAX_PIECES]
            t1 = time.time()
            poses = self.default_pipeline([g["en"] for g in segs])
            t2 = time.time()
            variants = {}
            for mth in (JOIN, "cut"):
                seq, bnd = self.join(poses, mth)
                variants[mth] = {"n": len(seq), "pts": r3(seq), "z": r3(self.lift(seq)), "jumps": self.join_jumps(seq, bnd),
                                 "segments": [{**g, "start": int(a), "end": int(b)} for g, (a, b) in zip(segs, bnd)]}
            t3 = time.time()
            self.n += 1
        clip = {"uid": f"live{self.n}_{lang}", "kind": "paragraph", "lang": lang, "source": text,
                "text": " ".join(g["en"] for g in segs), "join": JOIN, **variants[JOIN], "cut": variants["cut"]}
        res = {"ok": True, "clip": clip, "point_idx": [int(v) for v in POINT_IDX], "fps": 25,
               "timing": {"translate": round(t1 - t0, 1), "generate": round(t2 - t1, 1), "depth": round(t3 - t2, 1)}}
        self.seen[(lang, text)] = res
        self.log(f"[{time.strftime('%H:%M:%S')}] {lang}: {text[:60]!r} -> {len(segs)} pieces, {clip['n'] / 25:.1f}s of signing, "
                 f"took {t3 - t0:.1f}s (translate {t1 - t0:.1f}, generate {t2 - t1:.1f}, depth {t3 - t2:.1f})")
        return res

    def get_translator(self):
        if self.translator is None:
            self.translator = Translator(self.log)
        return self.translator


class DepthNet(nn.Module):
    """Notebook 18's depth model (x, y -> depth of all 123 points)."""
    def __init__(self, n=None, d=256, layers=4, heads=4, maxT=300, dropout=0.1):
        super().__init__()
        n = n or B.N_POINTS
        for k in ("xm", "xs", "vs"):
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


class Translator:
    """IndicTrans2 (Hindi/Kannada -> English), loaded on first use; translations are kept in bundle/translations_cache.json."""
    def __init__(self, log=print):
        t0 = time.time()
        import types
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            for n in ("collator", "evaluator"):
                m = types.ModuleType("IndicTransToolkit." + n); m.IndicDataCollator = m.IndicEvaluator = None
                sys.modules["IndicTransToolkit." + n] = m
            from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.ip = IndicProcessor(inference=True)
        self.tok = AutoTokenizer.from_pretrained(HF["indic_en"], trust_remote_code=True)
        # Apple GPU: full precision is the safe choice for this model; NVIDIA: half precision
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        self.mdl = AutoModelForSeq2SeqLM.from_pretrained(HF["indic_en"], trust_remote_code=True, torch_dtype=dtype).to(DEVICE).eval()
        self.cache_f = os.path.join(BUNDLE, "translations_cache.json")
        self.cache = json.load(open(self.cache_f)) if os.path.exists(self.cache_f) else {}
        log(f"IndicTrans2 loaded on {DEVICE} [{time.time() - t0:.0f}s]")

    @staticmethod
    def key(src, s):
        return hashlib.md5(f"ai4bharat/indictrans2-indic-en-1B|{src}|eng_Latn|{s}".encode("utf-8")).hexdigest()   # same keys as notebook 19

    @torch.no_grad()
    def translate(self, sents, lang):
        src = LANGS[lang]
        todo = [x for x in dict.fromkeys(sents) if self.key(src, x) not in self.cache]
        for b0 in range(0, len(todo), 8):
            part = todo[b0:b0 + 8]
            batch = self.ip.preprocess_batch(part, src_lang=src, tgt_lang="eng_Latn")
            enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE)
            gen = self.mdl.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for x, y in zip(part, self.ip.postprocess_batch(dec, lang="eng_Latn")):
                self.cache[self.key(src, x)] = y.strip()
        if todo:
            json.dump(self.cache, open(self.cache_f + ".tmp", "w"), ensure_ascii=False); os.replace(self.cache_f + ".tmp", self.cache_f)
        return [self.cache[self.key(src, x)] for x in sents]


LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "kn": "kan_Knda"}
JOIN = "ease 8"            # chosen in notebook 19
MAX_PIECES = 12
MAX_WORDS = 18             # notebook 19: 90th percentile of training sentence length
JOINERS = {"and", "but", "because", "so", "while", "which", "who", "where", "when", "after", "before", "or", "then", "although"}


def smooth01(n):
    t = (np.arange(n) + 1) / (n + 1)
    return t * t * (3 - 2 * t)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+|(?<=[।॥])\s*", text)
    return [p.strip() for p in parts if p and re.search(r"\w", p)]


def chunk_english(s, max_words=MAX_WORDS):
    w = s.split()
    if len(w) <= max_words:
        return [s]
    cands = [i for i in range(3, len(w) - 2) if w[i - 1][-1] in ",;:" or w[i].lower() in JOINERS]
    cut = min(cands, key=lambda i: abs(i - len(w) / 2)) if cands else len(w) // 2
    return chunk_english(" ".join(w[:cut]).rstrip(",;:"), max_words) + chunk_english(" ".join(w[cut:]), max_words)


def detect_lang(text):
    kn = len(re.findall(r"[ಀ-೿]", text)); hi = len(re.findall(r"[ऀ-ॿ]", text))
    return "kn" if kn > max(hi, 2) else "hi" if hi > 2 else "en"


def r3(a):
    return [[round(float(v), 3) for v in row] for row in np.asarray(a, np.float32).reshape(len(a), -1)]
