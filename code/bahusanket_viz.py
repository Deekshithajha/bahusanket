"""BahuSanket pose viewer: draw our 103-point poses as a stick-figure signer.

Use it to look at real iSign clips and at what a model generates, side by side.
  * pose_from_bytes(raw, point_idx)  -> (T,103,3) normalised poses + (T,103) mask, from a .pose file
  * save_gif([seq, ...], path, titles, caption)  -> animated GIF (one panel per sequence)
  * save_strip(seq, path, caption, n=6)          -> PNG of a few still frames
Input sequences are (T,103,2|3) in model_point_indices.json order (7 body, 54 face, 42 hands),
in shoulder-width units, y pointing down (image convention). Points at exactly (0,0) or NaN
are treated as missing and not drawn, matching bahusanket_lib.

What our 54 face points cover (checked against the MediaPipe face-mesh numbering):
jaw outline (19), both eyebrows (7 each), lower lip outer (11) and inner (10) edges.
There is no upper lip, eye or nose-bridge point, so mouth opening and eye gaze are not visible.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Face-mesh ids of our 54 face points, in the stored order (point_idx[7:61] minus 33).
FACE_MESH_IDS = [14, 17, 55, 58, 61, 63, 65, 66, 70, 78, 84, 87, 88, 91, 95, 105, 107, 132, 136,
                 146, 148, 149, 150, 152, 172, 176, 178, 181, 234, 285, 288, 291, 293, 295, 296,
                 300, 314, 317, 318, 321, 324, 334, 336, 361, 365, 375, 377, 378, 379, 397, 400,
                 402, 405, 454]
_FACE_LINES = {   # drawing order along each contour, as face-mesh ids
    "jaw": [234, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 454],
    "brow_r": [70, 63, 105, 66, 107, 55, 65],
    "brow_l": [300, 293, 334, 296, 336, 285, 295],
    "lip_outer": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lip_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
}
FACE_LINES = {k: [7 + FACE_MESH_IDS.index(m) for m in v] for k, v in _FACE_LINES.items()}

# Body: 0 nose, 1/2 left/right shoulder, 3/4 elbows, 5/6 wrists (MediaPipe pose 0, 11-16).
BODY_EDGES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6)]
_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
LHAND, RHAND = 61, 82
COLORS = {"body": "#5b6170", "face": "#c0504d", "lhand": "#2f7ed8", "rhand": "#e08a1e"}


def pose_from_bytes(raw, point_idx):
    """Read a .pose file (bytes), keep our 103 points, centre on the shoulder midpoint and divide
    by shoulder width, per frame. Missing points (confidence 0) become (0,0,0) with mask False."""
    from pose_format import Pose
    p = Pose.read(raw)
    data = np.asarray(p.body.data.filled(0) if hasattr(p.body.data, "filled") else p.body.data,
                      np.float32)[:, 0][:, point_idx]                  # (T,103,3) pixels
    mask = np.asarray(p.body.confidence, np.float32)[:, 0][:, point_idx] > 0
    ls, rs = data[:, 1, :2], data[:, 2, :2]
    ok = mask[:, 1] & mask[:, 2]
    mid = (ls + rs) / 2
    width = np.linalg.norm(ls - rs, axis=-1)
    ok &= width > 1e-3
    out = np.zeros_like(data)
    out[ok] = (data[ok] - np.concatenate([mid[ok], np.zeros((ok.sum(), 1))], -1)[:, None]) / width[ok, None, None]
    mask = mask & ok[:, None]
    out[~mask] = 0
    return out, mask, float(p.body.fps)


def _xy(seq):
    a = np.asarray(seq, np.float32)[..., :2].copy()
    a[np.abs(a).sum(-1) == 0] = np.nan
    return a


def _draw(ax, f, lw=2.0):
    """Draw one frame f (103,2) onto ax."""
    def line(i, j, c, w):
        if not (np.isnan(f[i]).any() or np.isnan(f[j]).any()):
            ax.plot([f[i, 0], f[j, 0]], [f[i, 1], f[j, 1]], color=c, lw=w, solid_capstyle="round")
    for i, j in BODY_EDGES:
        line(i, j, COLORS["body"], lw * 2.2)
    chin = FACE_LINES["jaw"][9]
    if not (np.isnan(f[[1, 2, chin]]).any()):                          # neck: shoulder midpoint to chin
        m = (f[1] + f[2]) / 2
        ax.plot([m[0], f[chin, 0]], [m[1], f[chin, 1]], color=COLORS["body"], lw=lw * 2.2)
    for base, wrist, c in ((LHAND, 5, COLORS["lhand"]), (RHAND, 6, COLORS["rhand"])):
        line(wrist, base, c, lw * 1.2)                                  # forearm to hand
        for i, j in _HAND:
            line(base + i, base + j, c, lw)
        ax.scatter(f[base:base + 21, 0], f[base:base + 21, 1], s=6, color=c, zorder=3)
    for idx in FACE_LINES.values():
        ax.plot(f[idx, 0], f[idx, 1], color=COLORS["face"], lw=lw * 0.8)
    ax.scatter(f[0, 0], f[0, 1], s=18, color=COLORS["face"], zorder=3)  # nose


def _limits(seqs):
    allp = np.concatenate([s.reshape(-1, 2) for s in seqs])
    allp = allp[~np.isnan(allp).any(-1)]
    lo, hi = allp.min(0), allp.max(0)
    c, r = (lo + hi) / 2, max(hi - lo) / 2 * 1.05
    return (c[0] - r, c[0] + r), (c[1] + r, c[1] - r)                   # y flipped: down is down


def _panel(ax, f, title, xl, yl):
    ax.clear()
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")
    _draw(ax, f)
    if title:
        ax.set_title(title, fontsize=11)


def _fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor="white")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(seqs, path, titles=None, caption="", fps=25, size=3.6, max_frames=None):
    """Animated GIF with one panel per sequence (e.g. [real, generated]). Shorter sequences hold
    their last frame. Frame counter is shown bottom-right."""
    seqs = [_xy(s) for s in (seqs if isinstance(seqs, (list, tuple)) else [seqs])]
    titles = titles or [""] * len(seqs)
    T = max(len(s) for s in seqs) if max_frames is None else min(max_frames, max(len(s) for s in seqs))
    xl, yl = _limits(seqs)
    fig, axes = plt.subplots(1, len(seqs), figsize=(size * len(seqs), size + 0.6), dpi=110, squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12)
    frames = []
    for t in range(T):
        for ax, s, ti in zip(axes[0], seqs, titles):
            _panel(ax, s[min(t, len(s) - 1)], ti, xl, yl)
        for txt in list(fig.texts):
            txt.remove()
        if caption:
            fig.text(0.5, 0.04, caption, ha="center", fontsize=10, wrap=True)
        fig.text(0.98, 0.94, f"{t / fps:4.1f}s", ha="right", fontsize=8, color="#888")
        frames.append(_fig_to_image(fig).quantize(colors=64))
    plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0,
                   optimize=True)
    return path


def save_strip(seq, path, caption="", n=6, fps=25, size=2.6):
    """PNG of n evenly spaced still frames from one sequence, with timestamps."""
    s = _xy(seq)
    xl, yl = _limits([s])
    ts = np.linspace(0, len(s) - 1, n).round().astype(int)
    fig, axes = plt.subplots(1, n, figsize=(size * n, size + 0.9), dpi=110)
    for ax, t in zip(np.atleast_1d(axes), ts):
        _panel(ax, s[t], f"{t / fps:.1f}s", xl, yl)
    if caption:
        fig.suptitle(caption, fontsize=13, y=0.97)
    fig.text(0.5, 0.02, "grey: arms   blue: left hand   orange: right hand   red: face (jaw, eyebrows, lower lip)",
             ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.08, wspace=0.02)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
