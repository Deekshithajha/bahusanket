"""BahuSanket pose viewer: draw our poses as a stick-figure signer.

Use it to look at real iSign clips and at what a model generates, side by side.
  * pose_from_bytes(raw, point_idx)  -> (T,N,2) normalised poses + (T,N) mask, from a .pose file
  * save_gif([seq, ...], path, point_idx, titles, caption)  -> animated GIF, one panel per sequence
  * save_strip(seq, path, point_idx, caption, n=6)          -> PNG of a few still frames
Sequences are (T,N,2|3) in the order of point_idx (the list in model_point_indices*.json; works
for both the 103-point v1 and the 123-point v2 layout), in shoulder-width units, y pointing down
(image convention). Points at exactly (0,0) or NaN are treated as missing and not drawn.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import bahusanket_layout as L

_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
COLORS = {"body": "#5b6170", "face": "#c0504d", "lhand": "#2f7ed8", "rhand": "#e08a1e"}


def pose_from_bytes(raw, point_idx):
    """Read a .pose file (bytes) and prepare it exactly like ISL.ipynb built training_ready/:
    keep our points (x, y), centre each frame on the shoulder midpoint, divide by the clip's
    median shoulder width. Points MediaPipe did not find (confidence 0) become (0,0), mask False."""
    from pose_format import Pose
    p = Pose.read(raw)
    data = p.body.data
    data = np.asarray(data.filled(0) if hasattr(data, "filled") else data, np.float32)[:, 0][:, point_idx, :2]
    mask = np.asarray(p.body.confidence, np.float32)[:, 0][:, point_idx] > 0
    ls, rs = list(point_idx).index(11), list(point_idx).index(12)
    centre = (data[:, ls] + data[:, rs]) / 2
    width = np.linalg.norm(data[:, ls] - data[:, rs], axis=-1)
    scale = np.median(width[width > 1e-3]) if (width > 1e-3).any() else 1.0
    out = (data - centre[:, None]) / scale
    out[~mask] = 0
    return out, mask, float(p.body.fps)


def _xy(seq):
    a = np.asarray(seq, np.float32)[..., :2].copy()
    a[np.abs(a).sum(-1) == 0] = np.nan
    return a


def _draw(ax, f, lay, lw=2.0):
    """Draw one frame f (N,2) onto ax."""
    def ok(*ks):
        return all(k is not None and not np.isnan(f[k]).any() for k in ks)

    def line(i, j, c, w):
        if ok(i, j):
            ax.plot([f[i, 0], f[j, 0]], [f[i, 1], f[j, 1]], color=c, lw=w, solid_capstyle="round")
    b = lay["body"]
    for i, j in (("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
                 ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist")):
        line(b.get(i), b.get(j), COLORS["body"], lw * 2.2)
    if ok(b.get("l_shoulder"), b.get("r_shoulder"), lay["chin"]):      # neck: shoulder midpoint to chin
        m = (f[b["l_shoulder"]] + f[b["r_shoulder"]]) / 2
        ax.plot([m[0], f[lay["chin"], 0]], [m[1], f[lay["chin"], 1]], color=COLORS["body"], lw=lw * 2.2)
    for base, wrist, c in ((lay["lhand"], b.get("l_wrist"), COLORS["lhand"]),
                           (lay["rhand"], b.get("r_wrist"), COLORS["rhand"])):
        if base is None:
            continue
        line(wrist, base, c, lw * 1.2)                                  # forearm to hand
        for i, j in _HAND:
            line(base + i, base + j, c, lw)
        ax.scatter(f[base:base + 21, 0], f[base:base + 21, 1], s=6, color=c, zorder=3)
    for name, ks in lay["face_lines"].items():
        ax.plot(f[ks, 0], f[ks, 1], color=COLORS["face"], lw=lw * (0.6 if name.startswith("eye") else 0.8))
    if "nose" in b:
        ax.scatter(f[b["nose"], 0], f[b["nose"], 1], s=18, color=COLORS["face"], zorder=3)


def _limits(seqs):
    allp = np.concatenate([s.reshape(-1, 2) for s in seqs])
    allp = allp[~np.isnan(allp).any(-1)]
    lo, hi = allp.min(0), allp.max(0)
    c, r = (lo + hi) / 2, max(hi - lo) / 2 * 1.05
    return (c[0] - r, c[0] + r), (c[1] + r, c[1] - r)                   # y flipped: down is down


def _panel(ax, f, lay, title, xl, yl):
    ax.clear()
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.axis("off")
    _draw(ax, f, lay)
    if title:
        ax.set_title(title, fontsize=11)


def _fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor="white")
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(seqs, path, point_idx, titles=None, caption="", fps=25, size=3.6, max_frames=None):
    """Animated GIF with one panel per sequence (e.g. [real, generated]). Shorter sequences hold
    their last frame. Time is shown top-right."""
    lay = L.layout(point_idx)
    seqs = [_xy(s) for s in (seqs if isinstance(seqs, (list, tuple)) else [seqs])]
    titles = titles or [""] * len(seqs)
    T = max(len(s) for s in seqs) if max_frames is None else min(max_frames, max(len(s) for s in seqs))
    xl, yl = _limits(seqs)
    fig, axes = plt.subplots(1, len(seqs), figsize=(size * len(seqs), size + 0.6), dpi=110, squeeze=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.12)
    frames = []
    for t in range(T):
        for ax, s, ti in zip(axes[0], seqs, titles):
            _panel(ax, s[min(t, len(s) - 1)], lay, ti, xl, yl)
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


def save_strip(seq, path, point_idx, caption="", n=6, fps=25, size=2.6, frames=None):
    """PNG of n evenly spaced still frames (or the given frame numbers) from one sequence."""
    lay = L.layout(point_idx)
    s = _xy(seq)
    xl, yl = _limits([s])
    ts = np.linspace(0, len(s) - 1, n).round().astype(int) if frames is None else np.asarray(frames)
    fig, axes = plt.subplots(1, len(ts), figsize=(size * len(ts), size + 0.9), dpi=110)
    for ax, t in zip(np.atleast_1d(axes), ts):
        _panel(ax, s[t], lay, f"{t / fps:.1f}s", xl, yl)
    if caption:
        fig.suptitle(caption, fontsize=13, y=0.97)
    fig.text(0.5, 0.02, "grey: arms   blue: left hand   orange: right hand   red: face",
             ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.08, wspace=0.02)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
