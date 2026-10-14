"""BahuSanket pose tokenizer: a VQ-VAE that turns pose sequences into discrete "pose codes".

Recipe from T2M-GPT (Zhang et al., CVPR 2023, https://arxiv.org/abs/2301.06052), Sec. 3.1,
Table 9 and implementation details:
  * 1-D convolutional encoder/decoder with dilated residual blocks (dilation 9, 3, 1),
    two stride-2 downsamplings, so one code covers 4 frames (0.16 s at 25 fps).
  * Codebook of 512 codes x 512 dims, updated by exponential moving average (EMA, 0.99) with
    "code reset" (unused codes are re-initialised from current encoder outputs).
  * Loss = smooth-L1 reconstruction + 0.5 x smooth-L1 on velocity + 0.02 x commitment.
  * Trained on 64-frame crops, AdamW (0.9, 0.99), lr 2e-4 then 1e-5.
T2S-GPT (Yin et al., ACL 2024, https://arxiv.org/abs/2406.07119) uses the same idea for sign
language; its Table 3 shows this fixed-rate VQ-VAE already roughly doubles back-translation
BLEU-4 over Progressive Transformers on PHOENIX14T (8.39 vs 4.38).

Our version masks the loss so missing points (stored as 0,0) are never treated as real.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bahusanket_lib as B

IN_DIM = B.N_POINTS * 2       # 103 points x (x, y)
DOWN = 4                      # frames per code


class ResConv1DBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


def resnet1d(width, dilations=(9, 3, 1)):
    return nn.Sequential(*[ResConv1DBlock(width, d) for d in dilations])


class PoseVQVAE(nn.Module):
    def __init__(self, in_dim=IN_DIM, width=512, code_dim=512, n_codes=512, n_down=2, mu=0.99):
        super().__init__()
        enc = [nn.Conv1d(in_dim, width, 3, padding=1), nn.ReLU()]
        for _ in range(n_down):
            enc += [nn.Conv1d(width, width, 4, stride=2, padding=1), resnet1d(width)]
        enc += [nn.Conv1d(width, code_dim, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)
        dec = [nn.Conv1d(code_dim, width, 3, padding=1), nn.ReLU()]
        for _ in range(n_down):
            dec += [resnet1d(width), nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv1d(width, width, 3, padding=1)]
        dec += [nn.Conv1d(width, width, 3, padding=1), nn.ReLU(), nn.Conv1d(width, in_dim, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)
        self.n_codes, self.code_dim, self.mu = n_codes, code_dim, mu
        self.register_buffer("codebook", torch.zeros(n_codes, code_dim))
        self.register_buffer("code_sum", torch.zeros(n_codes, code_dim))
        self.register_buffer("code_count", torch.ones(n_codes))
        self.register_buffer("initted", torch.tensor(False))

    # ---- quantizer (EMA + code reset) ----
    def _random_rows(self, z, n):
        if len(z) < n:   # repeat with a little noise if the batch has fewer vectors than codes
            z = z.repeat((n + len(z) - 1) // len(z), 1)
            z = z + torch.randn_like(z) * 0.01 / np.sqrt(z.shape[1])
        return z[torch.randperm(len(z), device=z.device)[:n]]

    def quantize(self, z):
        """z (N, code_dim) -> (codes (N,), quantised vectors (N, code_dim), stats)."""
        if self.training and not bool(self.initted):
            init = self._random_rows(z.detach(), self.n_codes)
            self.codebook.copy_(init); self.code_sum.copy_(init); self.code_count.fill_(1.0)
            self.initted.fill_(True)
        d = (z.pow(2).sum(1, keepdim=True) - 2 * z @ self.codebook.t()
             + self.codebook.pow(2).sum(1)[None])
        idx = d.argmin(1)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(z.dtype)          # (N, K)
                cnt, s = onehot.sum(0), onehot.t() @ z.detach()
                self.code_sum.mul_(self.mu).add_(s, alpha=1 - self.mu)
                self.code_count.mul_(self.mu).add_(cnt, alpha=1 - self.mu)
                used = (self.code_count >= 1.0).float()[:, None]
                fresh = self._random_rows(z.detach(), self.n_codes)
                self.codebook.copy_(used * self.code_sum / self.code_count[:, None].clamp(min=1e-5)
                                    + (1 - used) * fresh)                     # code reset
                self.code_sum.copy_(used * self.code_sum + (1 - used) * fresh)
                self.code_count.copy_(used[:, 0] * self.code_count + (1 - used[:, 0]))
        zq = self.codebook[idx]
        prob = torch.bincount(idx, minlength=self.n_codes).float() / len(idx)
        perplexity = torch.exp(-(prob * torch.log(prob + 1e-10)).sum())
        return idx, zq, {"perplexity": perplexity, "used": (prob > 0).float().mean()}

    # ---- full model ----
    def encode(self, x):
        """x (B, T, IN_DIM) with T a multiple of 4 -> codes (B, T/4)."""
        z = self.encoder(x.transpose(1, 2))                      # (B, C, T/4)
        Bn, C, Tq = z.shape
        idx, _, _ = self.quantize(z.transpose(1, 2).reshape(-1, C).float())
        return idx.view(Bn, Tq)

    def decode(self, codes):
        """codes (B, T/4) -> poses (B, T, IN_DIM)."""
        zq = self.codebook[codes].transpose(1, 2)                # (B, C, T/4)
        return self.decoder(zq).transpose(1, 2)

    def forward(self, x):
        z = self.encoder(x.transpose(1, 2))
        Bn, C, Tq = z.shape
        zf = z.transpose(1, 2).reshape(-1, C).float()
        with torch.autocast(device_type=zf.device.type, enabled=False):   # quantize in full precision
            idx, zq, stats = self.quantize(zf)
            commit = F.mse_loss(zf, zq.detach())
        zq = zf + (zq - zf).detach()                              # straight-through gradient
        out = self.decoder(zq.view(Bn, Tq, C).transpose(1, 2)).transpose(1, 2)
        return out, commit, stats, idx.view(Bn, Tq)


def vq_loss(recon, x, pmask, commit, alpha_vel=0.5, beta=0.02):
    """Masked smooth-L1 on positions and velocities. pmask (B,T,103) bool: real points only."""
    Bn, T, _ = x.shape
    r, g = recon.view(Bn, T, B.N_POINTS, 2), x.view(Bn, T, B.N_POINTS, 2)
    m = pmask.float()[..., None]
    l_pos = (F.smooth_l1_loss(r, g, reduction="none") * m).sum() / (m.sum() * 2).clamp(min=1)
    vm = (pmask[:, 1:] & pmask[:, :-1]).float()[..., None]
    l_vel = (F.smooth_l1_loss(r[:, 1:] - r[:, :-1], g[:, 1:] - g[:, :-1], reduction="none") * vm).sum() \
        / (vm.sum() * 2).clamp(min=1)
    total = l_pos + alpha_vel * l_vel + beta * commit
    return {"total": total, "pos": l_pos, "vel": l_vel, "commit": commit}


class CropDataset(torch.utils.data.Dataset):
    """Random 64-frame windows from the training clips (shorter clips are padded and masked)."""
    def __init__(self, split, crop=64, samples_per_epoch=None):
        self.s, self.crop = split, crop
        self.n = samples_per_epoch or len(split["poses"])

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        i = np.random.randint(len(self.s["poses"]))
        p, m = self.s["poses"][i], self.s["masks"][i]
        L = len(p)
        st = np.random.randint(0, max(1, L - self.crop + 1))
        x = np.zeros((self.crop, B.N_POINTS, 2), np.float32)
        pm = np.zeros((self.crop, B.N_POINTS), bool)
        seg = slice(st, min(L, st + self.crop))
        n = seg.stop - seg.start
        x[:n], pm[:n] = p[seg, :, :2], m[seg]
        x[~pm] = 0.0
        return torch.from_numpy(x.reshape(self.crop, -1)), torch.from_numpy(pm)


def worker_init(_):
    """Give each DataLoader worker its own random crops (otherwise they repeat each other)."""
    np.random.seed(torch.initial_seed() % 2 ** 32)


@torch.no_grad()
def tokenize_and_rebuild(model, split, idx, device, batch_size=64):
    """Encode whole clips to codes and decode them again. Returns (codes list, rebuilt poses list)."""
    model.eval()
    codes_out, rec_out = [], []
    for s in range(0, len(idx), batch_size):
        chunk = idx[s:s + batch_size]
        Ls = [len(split["poses"][i]) for i in chunk]
        T = int(np.ceil(max(Ls) / DOWN) * DOWN)
        x = np.zeros((len(chunk), T, IN_DIM), np.float32)
        for k, i in enumerate(chunk):
            p = split["poses"][i][..., :2].astype(np.float32).copy()
            p[~split["masks"][i]] = 0.0
            x[k, :Ls[k]] = p.reshape(Ls[k], -1)
        xt = torch.from_numpy(x).to(device)
        codes = model.encode(xt)
        rec = model.decode(codes).float().cpu().numpy().reshape(len(chunk), T, B.N_POINTS, 2)
        codes = codes.cpu().numpy()
        for k, L in enumerate(Ls):
            nq = int(np.ceil(L / DOWN))
            codes_out.append(codes[k, :nq].astype(np.int16))
            rec_out.append(rec[k, :L])
    return codes_out, rec_out


def aligned_mje(preds, split, idx):
    """Frame-by-frame mean joint error (no time warping), real points only. For reconstructions,
    which have exactly the real clip's timing, this is the plain 'how far off is each point'."""
    tot, n = 0.0, 0
    for p, i in zip(preds, idx):
        g, m = split["poses"][i][..., :2].astype(np.float32), split["masks"][i]
        d = np.linalg.norm(p[:len(g)] - g, axis=-1)
        tot += float((d * m).sum()); n += int(m.sum())
    return tot / max(n, 1)
