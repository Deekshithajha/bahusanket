"""BahuSanket shared code: data loading, the Progressive-Transformer-style model, and the
DTW-MJE pose-distance metric. Imported by the Colab notebooks (copied to Drive/BahuSanket/code).

Conventions (from project memory / earlier notebook):
  * 103 points per frame, in model_point_indices.json order: 7 body, 54 face, 42 hands
    (21 left then 21 right).
  * Coordinates already normalised: centred on the shoulder midpoint, divided by shoulder width.
    So every distance below is in "shoulder widths" (0.10 = 10% of the signer's shoulder width).
  * Text: distilbert-base-uncased tokenizer, 28 tokens. Poses: at most 300 frames (25 fps).
"""
import json, math, os, time
import numpy as np

N_POINTS = 103
PART_SLICES = {"body": slice(0, 7), "face": slice(7, 61), "hands": slice(61, 103)}
MAX_FRAMES = 300
MAX_TOKENS = 28
OUTLIER = 10.0     # |coordinate| above this many shoulder widths cannot be a real body point
JUNK_RE = r"^\s*(page|unit|chapter|lesson)\s*(no\.?|number)?\s*\d+\s*[.!:,]*\s*$"


# --------------------------------------------------------------------------------------
# Data loading. The training_ready/*.npz files were written by ISL.ipynb; we detect their
# keys instead of hard-coding them, and print what we found so a human can check it.
# --------------------------------------------------------------------------------------
def describe_npz(path, n_show=2):
    z = np.load(path, allow_pickle=True)
    print(f"{os.path.basename(path)}: {len(z.files)} arrays")
    for k in z.files:
        a = z[k]
        line = f"  {k:<20} shape={a.shape} dtype={a.dtype}"
        if a.dtype == object and a.size:
            e = a.flat[0]
            line += f"  first item: {type(e).__name__} {getattr(e, 'shape', '')}"
        elif a.dtype.kind in "USO" and a.size:
            line += f"  e.g. {list(a.flat[:n_show])}"
        print(line)
    return z


def _find(z, words, kinds=None):
    for k in z.files:
        if any(w in k.lower() for w in words):
            if kinds is None or z[k].dtype.kind in kinds:
                return k
    return None


def _find_poses(z):
    """The pose array: numeric with a 103-sized axis, or an object array of such clips."""
    best = None
    for k in z.files:
        a = z[k]
        if a.dtype.kind == "f" and a.ndim >= 3 and (N_POINTS in a.shape or a.shape[-1] % N_POINTS == 0):
            if best is None or a.size > z[best].size:
                best = k
        elif a.dtype == object and a.size and isinstance(a.flat[0], np.ndarray) \
                and a.flat[0].dtype.kind == "f" and a.flat[0].ndim >= 2:
            if any(w in k.lower() for w in ("pose", "keypoint", "landmark", "x")) or best is None:
                best = k
    return best


def _to_frames(a):
    """One clip -> float array (T, 103, D)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4:                       # (T, people, 103, D): take person 0
        a = a[:, 0]
    if a.ndim == 2:                       # (T, 103*D)
        a = a.reshape(a.shape[0], N_POINTS, -1)
    assert a.ndim == 3 and a.shape[1] == N_POINTS, f"unexpected clip shape {a.shape}"
    return a


def load_split(path, keymap=None, verbose=True, drop_junk=True):
    """Return dict(poses=[(T,103,D) float16], masks=[(T,103) bool], texts, uids, input_ids).

    keymap can force key names, e.g. {"poses": "X", "texts": "sentences"}.
    drop_junk removes clips whose sentence is only a page/unit marker ("Page 54.").
    """
    keymap = keymap or {}
    z = np.load(path, allow_pickle=True)
    kp = keymap.get("poses") or _find_poses(z)
    kt = keymap.get("texts") or _find(z, ["text", "sentence", "sent"], "USO")
    ku = keymap.get("uids") or _find(z, ["uid", "id"], "USO")
    kl = keymap.get("lengths") or _find(z, ["len", "n_frames", "num_frames"], "iu")
    km = keymap.get("masks") or _find(z, ["mask", "conf", "valid", "vis"], "fbiuO")
    ki = keymap.get("input_ids") or _find(z, ["input_ids", "token"], "iu")
    if ku == kt:
        ku = None
    if verbose:
        print(f"[{os.path.basename(path)}] poses={kp} texts={kt} uids={ku} lengths={kl} "
              f"masks={km} input_ids={ki}")
    assert kp is not None, "could not find the pose array; pass keymap={'poses': ...}"
    raw = z[kp]
    n = len(raw)
    lengths = z[kl].astype(int) if kl else None
    masks_raw = z[km] if km and km != kp else None
    if masks_raw is not None and masks_raw.dtype != object and masks_raw.ndim == 2 \
            and masks_raw.shape[1] != N_POINTS:
        # (clips, frames) mask = which frames are real (padding marker), not which points.
        # Use it for clip lengths; missing points are then found as exact zeros.
        fm = masks_raw.astype(bool)
        lengths = np.where(fm.any(1), fm.shape[1] - np.argmax(fm[:, ::-1], axis=1), 0)
        masks_raw = None
        if verbose:
            print(f"  '{km}' is a per-frame padding mask: used for clip lengths")

    poses, masks = [], []
    n_points, n_bad, clips_bad, absmax = 0, 0, 0, []
    for i in range(n):
        a = _to_frames(raw[i])
        L = int(lengths[i]) if lengths is not None else _infer_length(a)
        L = max(1, min(L, MAX_FRAMES, len(a)))
        a = a[:L]
        if masks_raw is not None:
            m = np.asarray(masks_raw[i])
            if m.ndim == 3:                # (T, people, 103) or (T,103,1)
                m = m[:, 0] if m.shape[1] != N_POINTS else m[..., 0]
            m = m[:L] > 0
        else:
            # no per-point mask stored: a point at exactly (0, 0) is one the tracker lost
            m = np.abs(np.nan_to_num(a)).sum(-1) > 0
        n_points += a.shape[0] * a.shape[1]
        finite = np.isfinite(a).all(-1)
        if finite.any():
            absmax.append(np.abs(a[finite]).max(-1))
        bad = ~finite | (np.abs(np.nan_to_num(a, posinf=1e9, neginf=1e9)).max(-1) > OUTLIER)
        if bad.any():
            # broken values (inf/nan, or e.g. divided by a near-zero shoulder width): treat as missing
            n_bad += int(bad.sum()); clips_bad += 1
            a = np.where(bad[..., None], 0.0, a)
            m = m & ~bad
        poses.append(a.astype(np.float16))
        masks.append(m)

    texts = [str(t) for t in z[kt]] if kt else None
    uids = [str(u) for u in z[ku]] if ku else None
    if texts is None and uids is not None:
        texts = _texts_from_csv(path, uids, verbose)
    input_ids = z[ki] if ki else None
    n_junk = 0
    if drop_junk and texts:
        import re
        keep = [i for i, t in enumerate(texts) if not re.match(JUNK_RE, t, re.I)]
        n_junk = len(texts) - len(keep)
        poses, masks, texts = [poses[i] for i in keep], [masks[i] for i in keep], [texts[i] for i in keep]
        uids = [uids[i] for i in keep] if uids else uids
        input_ids = input_ids[keep] if input_ids is not None else None
    out = dict(poses=poses, masks=masks, texts=texts, uids=uids, input_ids=input_ids)
    if verbose:
        Ls = np.array([len(p) for p in poses])
        print(f"  {n} clips, D={poses[0].shape[-1]}, frames: mean {Ls.mean():.0f}, "
              f"max {Ls.max()}, ={MAX_FRAMES}: {(Ls == MAX_FRAMES).mean():.1%}")
        av = np.concatenate([x.ravel() for x in absmax]) if absmax else np.zeros(1)
        q = np.percentile(av, [50, 99, 99.9])
        print(f"  |coordinate| per point: median {q[0]:.2f}, 99% {q[1]:.2f}, 99.9% {q[2]:.2f}, "
              f"max {av.max():.3g} (shoulder widths)")
        print(f"  broken points (inf/nan or > {OUTLIER:g} shoulder widths) set to missing: "
              f"{n_bad} of {n_points} ({n_bad / max(n_points, 1):.3%}) in {clips_bad} clips")
        if texts:
            print(f"  e.g. text: {texts[0]!r}")
        if drop_junk:
            print(f"  dropped {n_junk} page/unit-marker clips (e.g. 'Page 54.'); {len(poses)} clips kept")
    return out


def _texts_from_csv(npz_path, uids, verbose=True):
    """The npz files hold token ids but no sentences; take the sentences from
    subset_uids.csv (uid,text,...) in the BahuSanket folder, one level above training_ready/."""
    import csv
    root = os.path.dirname(os.path.dirname(os.path.abspath(npz_path)))
    p = os.path.join(root, "subset_uids.csv")
    if not os.path.exists(p):
        if verbose:
            print(f"  no texts in the npz and no {p}; retrieval baseline will not run")
        return None
    with open(p, newline="", encoding="utf-8") as f:
        lookup = {r["uid"]: r["text"] for r in csv.DictReader(f)}
    missing = [u for u in uids if u not in lookup]
    assert not missing, f"{len(missing)} uids not in subset_uids.csv, e.g. {missing[:3]}"
    if verbose:
        print(f"  texts taken from subset_uids.csv for all {len(uids)} clips")
    return [lookup[u] for u in uids]


def _infer_length(a):
    """Padded clips: last frame that is not all-zero."""
    nz = np.where(np.abs(a).reshape(len(a), -1).sum(1) > 0)[0]
    return int(nz[-1]) + 1 if len(nz) else len(a)


def missing_report(split):
    """How often each body part is missing (hands drop out when MediaPipe loses them)."""
    rep = {}
    for part, sl in PART_SLICES.items():
        rep[part] = float(np.mean([1 - m[:, sl].mean() for m in split["masks"]]))
    print("  missing points: " + ", ".join(f"{k} {v:.1%}" for k, v in rep.items()))
    return rep


def joint_mask(split, i):
    """(T,103) bool: which points are real (not missing) in clip i."""
    return split["masks"][i]


# --------------------------------------------------------------------------------------
# DTW-MJE: Dynamic Time Warping over Mean Joint Error.
# MJE between two frames = average Euclidean distance between matching points.
# DTW lets a slower or faster (but otherwise right) signing still line up with the
# reference, so we don't punish timing differences, only wrong shapes/positions.
# Lower is better. Units: shoulder widths.
# --------------------------------------------------------------------------------------
try:
    from numba import njit
except ImportError:                       # pure-python fallback (slow)
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True)
def _dtw_path(C):
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            b = D[i - 1, j - 1]
            if D[i - 1, j] < b:
                b = D[i - 1, j]
            if D[i, j - 1] < b:
                b = D[i, j - 1]
            D[i, j] = C[i - 1, j - 1] + b
    # backtrack
    pi = np.empty(n + m, np.int64)
    pj = np.empty(n + m, np.int64)
    i, j, k = n, m, 0
    while i > 0 and j > 0:
        pi[k] = i - 1
        pj[k] = j - 1
        k += 1
        a, b, c = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        if a <= b and a <= c:
            i -= 1; j -= 1
        elif b <= c:
            i -= 1
        else:
            j -= 1
    return pi[:k][::-1], pj[:k][::-1]


def dtw_mje(pred, gt, gt_mask=None, dims=2):
    """pred (Tp,103,D), gt (Tg,103,D), gt_mask (Tg,103) bool. Returns dict of errors.

    dims=2 scores x,y only (MediaPipe's z/depth is unreliable from a single camera).
    'all' is the DTW-MJE over all 103 points; body/face/hands are errors of those points
    along the same alignment.
    """
    p = np.asarray(pred, np.float32)[..., :dims]
    g = np.asarray(gt, np.float32)[..., :dims]
    m = np.ones(g.shape[:2], bool) if gt_mask is None else np.asarray(gt_mask, bool)
    # per-point distance for every (pred frame, gt frame) pair: (Tp, Tg, 103)
    d = np.linalg.norm(p[:, None] - g[None], axis=-1)
    w = m[None].astype(np.float32)
    C = (d * w).sum(-1) / np.maximum(w.sum(-1), 1)
    pi, pj = _dtw_path(C.astype(np.float64))
    out = {"all": float(C[pi, pj].mean())}
    for part, sl in PART_SLICES.items():
        dd, ww = d[pi, pj, sl], w[0, pj, sl]
        out[part] = float((dd * ww).sum() / ww.sum()) if ww.sum() > 0 else float("nan")
    out["len_ratio"] = len(p) / len(g)
    return out


def score_predictions(preds, split, idx=None, dims=2):
    """Average DTW-MJE over clips. preds[k] is the prediction for clip idx[k]."""
    idx = range(len(split["poses"])) if idx is None else idx
    rows = []
    for p, i in zip(preds, idx):
        r = dtw_mje(p, split["poses"][i], joint_mask(split, i), dims)
        r["motion_pred"] = motion(p)
        r["motion_real"] = motion(split["poses"][i], joint_mask(split, i))
        rows.append(r)
    keys = rows[0].keys()
    res = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    # motion_ratio: total movement of the predictions / total movement of the real clips
    # (1.0 = moves as much as the real signers, 0 = frozen). A ratio of totals, not an average
    # of per-clip ratios: a few real clips barely move, and dividing by them blew the average up.
    res["motion_ratio"] = res["motion_pred"] / max(res["motion_real"], 1e-6)
    res["n_clips"] = len(rows)
    return res, rows


def mean_pose(split, max_clips=3000):
    """The average pose over the training set (per point). Used as a 'do nothing' floor."""
    acc, cnt = np.zeros((N_POINTS, split["poses"][0].shape[-1])), np.zeros((N_POINTS, 1))
    for i in range(min(max_clips, len(split["poses"]))):
        p, m = split["poses"][i].astype(np.float64), joint_mask(split, i)
        acc += (p * m[..., None]).sum(0)
        cnt += m.sum(0)[:, None]
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


# --------------------------------------------------------------------------------------
# Model: Progressive Transformer (Saunders et al. 2020, https://arxiv.org/abs/2004.14874).
# Text encoder -> autoregressive pose decoder. Each frame carries a "counter" in [0,1]
# saying how far through the sentence we are; generation stops when it reaches 1.
# --------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class PosEnc(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class ProgressiveTransformer(nn.Module):
    def __init__(self, vocab_size=30522, pose_dim=N_POINTS * 3, d_model=256, n_heads=4,
                 n_enc=2, n_dec=2, ff=1024, dropout=0.1, text_encoder="scratch"):
        super().__init__()
        self.pose_dim, self.text_encoder = pose_dim, text_encoder
        if text_encoder == "distilbert":
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
            for p in self.bert.parameters():
                p.requires_grad = False
            self.txt_proj = nn.Linear(self.bert.config.dim, d_model)
        else:
            self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe = PosEnc(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, batch_first=True), n_enc)
        self.frame_in = nn.Linear(pose_dim + 1, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads, ff, dropout, batch_first=True), n_dec)
        self.frame_out = nn.Linear(d_model, pose_dim + 1)
        self.d_model = d_model

    def encode(self, ids, attn):
        if self.text_encoder == "distilbert":
            with torch.no_grad():
                h = self.bert(input_ids=ids, attention_mask=attn).last_hidden_state
            x = self.txt_proj(h)
        else:
            x = self.tok(ids) * math.sqrt(self.d_model)
        return self.encoder(self.pe(x), src_key_padding_mask=~attn.bool()), ~attn.bool()

    def decode(self, mem, mem_pad, frames_in):
        """frames_in (B,T,pose_dim+1) -> pose (B,T,pose_dim), counter (B,T)."""
        T = frames_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=frames_in.device), 1)
        h = self.decoder(self.pe(self.frame_in(frames_in)), mem, tgt_mask=causal,
                         memory_key_padding_mask=mem_pad)
        o = self.frame_out(h)
        return o[..., :-1], o[..., -1]   # linear counter (a sigmoid struggles to reach 1)

    def forward(self, ids, attn, frames_in):
        mem, pad = self.encode(ids, attn)
        return self.decode(mem, pad, frames_in)

    @torch.no_grad()
    def generate(self, ids, attn, max_frames=MAX_FRAMES, stop_all_at=0.995):
        """Greedy autoregressive generation of up to max_frames frames.
        Returns (poses (B,T,pose_dim), counters (B,T)); cut each clip with apply_stop().
        Generation ends early once every clip's counter has passed stop_all_at."""
        self.eval()
        B = ids.size(0)
        mem, pad = self.encode(ids, attn)
        seq = torch.zeros(B, 1, self.pose_dim + 1, device=ids.device)  # start frame
        cmax = torch.zeros(B, device=ids.device)
        for t in range(max_frames):
            pose, cnt = self.decode(mem, pad, seq)
            nxt = torch.cat([pose[:, -1], cnt[:, -1:].clamp(0, 1)], -1)
            seq = torch.cat([seq, nxt.unsqueeze(1)], 1)
            cmax = torch.maximum(cmax, cnt[:, -1])
            if bool((cmax >= stop_all_at).all()):
                break
        return seq[:, 1:, :-1], seq[:, 1:, -1]


# --------------------------------------------------------------------------------------
# Batching and loss
# --------------------------------------------------------------------------------------
class PoseTextDataset(torch.utils.data.Dataset):
    def __init__(self, split, tokenizer=None):
        self.s = split
        if split["texts"] is not None and tokenizer is not None:
            enc = tokenizer(split["texts"], padding="max_length", truncation=True,
                            max_length=MAX_TOKENS, return_tensors="np")
            self.ids, self.attn = enc["input_ids"], enc["attention_mask"]
        else:
            self.ids = np.asarray(split["input_ids"])[:, :MAX_TOKENS]
            self.attn = (self.ids != 0).astype(np.int64)

    def __len__(self):
        return len(self.s["poses"])

    def __getitem__(self, i):
        return (self.ids[i], self.attn[i], self.s["poses"][i], joint_mask(self.s, i), i)


def collate(batch):
    ids = torch.as_tensor(np.stack([b[0] for b in batch]), dtype=torch.long)
    attn = torch.as_tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
    T = max(len(b[2]) for b in batch)
    J, D = batch[0][2].shape[1:]
    poses = torch.zeros(len(batch), T, J, D)
    jmask = torch.zeros(len(batch), T, J, dtype=torch.bool)
    fmask = torch.zeros(len(batch), T, dtype=torch.bool)
    counter = torch.zeros(len(batch), T)
    for k, b in enumerate(batch):
        L = len(b[2])
        poses[k, :L] = torch.from_numpy(b[2].astype(np.float32))
        jmask[k, :L] = torch.from_numpy(b[3])
        fmask[k, :L] = True
        counter[k, :L] = torch.linspace(1.0 / L, 1.0, L)  # reaches exactly 1 on the last frame
    return dict(ids=ids, attn=attn, poses=poses, jmask=jmask, fmask=fmask, counter=counter,
                idx=[b[4] for b in batch])


def teacher_inputs(batch, noise_std=0.0):
    """Decoder input = ground truth shifted right by one frame (plus optional noise, which
    the PT paper uses so the model learns to recover from its own small mistakes)."""
    B, T, J, D = batch["poses"].shape
    gt = torch.cat([batch["poses"].reshape(B, T, J * D), batch["counter"].unsqueeze(-1)], -1)
    if noise_std > 0:
        gt = gt + torch.randn_like(gt) * noise_std * torch.cat(
            [torch.ones(J * D), torch.zeros(1)]).to(gt.device)
    start = torch.zeros(B, 1, J * D + 1, device=gt.device)
    return torch.cat([start, gt[:, :-1]], 1)


def point_weights(w_body=1.0, w_face=1.0, w_hands=1.0):
    w = torch.ones(N_POINTS)
    w[PART_SLICES["body"]], w[PART_SLICES["face"]], w[PART_SLICES["hands"]] = w_body, w_face, w_hands
    return w


def pt_loss(pred_pose, pred_cnt, batch, pw, w_counter=1.0, w_vel=0.0, w_acc=0.0):
    """Masked, per-part-weighted MSE + counter loss + optional smoothness terms.

    w_vel: match predicted frame-to-frame motion (velocity) to the real motion.
    w_acc: penalise jerky predicted motion (acceleration = change in velocity).
    """
    B, T, J, D = batch["poses"].shape
    p = pred_pose.reshape(B, T, J, D)
    g = batch["poses"]
    m = batch["jmask"].float() * pw.to(p.device)                # (B,T,J)
    se = ((p - g) ** 2).sum(-1)                                 # (B,T,J)
    l_pose = (se * m).sum() / batch["jmask"].float().sum().clamp(min=1)
    fm = batch["fmask"].float()
    l_cnt = (((pred_cnt - batch["counter"]) ** 2) * fm).sum() / fm.sum()
    out = {"pose": l_pose, "counter": l_cnt}
    total = l_pose + w_counter * l_cnt
    if w_vel > 0 or w_acc > 0:
        vm = batch["jmask"][:, 1:] & batch["jmask"][:, :-1]
        vp, vg = p[:, 1:] - p[:, :-1], g[:, 1:] - g[:, :-1]
        if w_vel > 0:
            l_vel = (((vp - vg) ** 2).sum(-1) * vm).sum() / vm.sum().clamp(min=1)
            out["vel"] = l_vel
            total = total + w_vel * l_vel
        if w_acc > 0:
            am = vm[:, 1:] & vm[:, :-1]
            l_acc = (((vp[:, 1:] - vp[:, :-1]) ** 2).sum(-1) * am).sum() / am.sum().clamp(min=1)
            out["acc"] = l_acc
            total = total + w_acc * l_acc
    out["total"] = total
    return out


def generate_for_split(model, split, tokenizer, idx, device, batch_size=32, max_frames=MAX_FRAMES,
                       stop_all_at=0.995):
    """Returns (full-length poses, counters) per clip; pick lengths with apply_stop().
    stop_all_at > 1 always generates max_frames frames (needed by cut_to_real_length)."""
    ds = PoseTextDataset(split, tokenizer)
    D = split["poses"][0].shape[-1]
    full, counters = [], []
    for s in range(0, len(idx), batch_size):
        chunk = idx[s:s + batch_size]
        ids = torch.as_tensor(ds.ids[chunk], dtype=torch.long, device=device)
        attn = torch.as_tensor(ds.attn[chunk], dtype=torch.long, device=device)
        poses, cnt = model.generate(ids, attn, max_frames=max_frames, stop_all_at=stop_all_at)
        poses = poses.float().cpu().numpy().reshape(len(chunk), -1, N_POINTS, D)
        cnt = cnt.float().cpu().numpy()
        full += list(poses); counters += list(cnt)
    return full, counters


def apply_stop(full, counters, stop_at, min_frames=10):
    """Cut each generated clip at the first frame whose counter reaches stop_at."""
    out = []
    for p, c in zip(full, counters):
        hit = np.where(c[min_frames - 1:] >= stop_at)[0]
        L = min_frames + int(hit[0]) if len(hit) else len(p)
        out.append(p[:L])
    return out


def cut_to_real_length(full, split, idx):
    """'Oracle length': give every prediction exactly the real clip's length, so a model can't
    look better (or worse) on DTW-MJE just by signing for longer. full must hold >= that many frames."""
    out = []
    for p, i in zip(full, idx):
        L = len(split["poses"][i])
        assert len(p) >= L, f"prediction has {len(p)} frames, real clip {L}: generate with stop_all_at=2"
        out.append(p[:L])
    return out


def resample_to_length(p, L):
    """Stretch/squeeze a pose sequence to L frames by picking the nearest frame (no blending,
    so missing (0,0) points stay missing). Used to give retrieved clips the real length."""
    return p[np.round(np.linspace(0, len(p) - 1, L)).astype(int)]


STOP_CANDIDATES = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99)


def calibrate_stop(full, counters, split, idx, candidates=STOP_CANDIDATES):
    """Pick the counter threshold whose generated lengths best match the real lengths
    (on validation clips only; the test set never influences this choice)."""
    real = np.array([len(split["poses"][i]) for i in idx], float)
    best, table = None, {}
    for thr in candidates:
        L = np.array([len(p) for p in apply_stop(full, counters, thr)], float)
        gap = abs(float(np.median(np.log(L / real))))
        table[thr] = float(np.median(L / real))
        if best is None or gap < best[1]:
            best = (thr, gap)
    return best[0], table


def load_model(ckpt_path, vocab_size, pose_dim, device):
    """Rebuild a model from a saved checkpoint, using the settings stored inside it."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    C = ck["config"]
    m = ProgressiveTransformer(vocab_size=vocab_size, pose_dim=pose_dim, d_model=C["d_model"],
                               n_heads=C["n_heads"], n_enc=C["n_enc"], n_dec=C["n_dec"], ff=C["ff"],
                               dropout=C["dropout"], text_encoder=C["text_encoder"]).to(device)
    m.load_state_dict(ck["model"])
    return m, ck


# --------------------------------------------------------------------------------------
# Baseline 0: sentence-embedding retrieval (all-MiniLM-L6-v2). For a new sentence, find the
# most similar training sentence and return its real pose sequence.
# --------------------------------------------------------------------------------------
def load_retrieval_index(path, train, verbose=True):
    """Return (embeddings (N,d) unit-length, train_clip_index (N,)) for retrieval_index.npz."""
    z = np.load(path, allow_pickle=True)
    ke = max((k for k in z.files if z[k].dtype.kind == "f" and z[k].ndim == 2),
             key=lambda k: z[k].size, default=None)
    assert ke is not None, f"no 2-D embedding array in {z.files}"
    emb = z[ke].astype(np.float32)
    ku = _find(z, ["uid"], "USO")
    kt = _find(z, ["text", "sentence"], "USO")
    ki = _find(z, ["idx", "index", "row"], "iu")
    if ku and train["uids"]:
        pos = {u: i for i, u in enumerate(train["uids"])}
        row2train = np.array([pos.get(str(u), -1) for u in z[ku]])
        how = f"uid key '{ku}'"
    elif ki:
        row2train = z[ki].astype(int)
        how = f"index key '{ki}'"
    elif kt and train["texts"]:
        pos = {}
        for i, t in enumerate(train["texts"]):
            pos.setdefault(t, i)
        row2train = np.array([pos.get(str(t), -1) for t in z[kt]])
        how = f"text key '{kt}'"
    else:
        assert len(emb) == len(train["poses"]), "index rows don't line up with train.npz"
        row2train = np.arange(len(emb))
        how = "same row order as train.npz"
    keep = row2train >= 0
    emb, row2train = emb[keep], row2train[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    if verbose:
        print(f"retrieval index: {z.files}; embeddings '{ke}' {emb.shape}; matched to train by "
              f"{how}; {keep.mean():.1%} of rows matched")
    return emb, row2train


def retrieve(encoder, emb, row2train, texts, batch_size=256):
    q = encoder.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False).astype(np.float32)
    sims = q @ emb.T
    best = sims.argmax(1)
    return row2train[best], sims[np.arange(len(q)), best]


def motion(p, mask=None):
    """Average amount a point moves over the clip (std over time, x,y), using only frames where
    the point is real. A model that has collapsed to an 'average pose' scores far below real data."""
    p = np.asarray(p, np.float32)[..., :2]
    if len(p) < 2:
        return 0.0
    if mask is None:                  # points at exactly (0, 0) are missing (e.g. in retrieved real clips)
        mask = np.abs(p).sum(-1) > 0
    m = np.asarray(mask, bool)[..., None].astype(np.float32)          # (T,103,1)
    n = m.sum(0)                                                      # (103,1)
    mu = (p * m).sum(0) / np.maximum(n, 1)
    sd = np.sqrt(((p - mu) ** 2 * m).sum(0) / np.maximum(n, 1))       # (103,2)
    ok = n[:, 0] >= 2
    return float(sd[ok].mean()) if ok.any() else 0.0
