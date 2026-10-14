"""BahuSanket stage 2: text -> pose codes.

A transformer reads the English sentence and writes pose codes (from the vq_v1 vocabulary) one
at a time, then an End token; the VQ decoder turns the codes back into poses.

Recipe:
  * T2M-GPT (Zhang et al., CVPR 2023, https://arxiv.org/abs/2301.06052), Sec. 3.2: next-code
    prediction with cross-entropy, a learned End token, and "corruption": during training a
    fraction tau ~ U[0,1] of the input codes is replaced by random codes, so the model learns to
    keep going sensibly after its own mistakes (at test time it only sees its own guesses).
  * T2S-GPT (Yin et al., ACL 2024, https://arxiv.org/abs/2406.07119), Sec. 4.3: for sign
    language the model is conditioned on the *words* of the sentence, not one sentence vector
    (T2M-GPT uses one CLIP vector). We do the same with cross-attention over the word vectors
    of a frozen DistilBERT (the tokenizer the project already uses), plus a few trainable layers.
Smaller than the papers' 18-layer GPT: we have ~15K sentences, and big models overfit that.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bahusanket_lib as B


class Text2Codes(nn.Module):
    def __init__(self, n_codes=512, d_model=512, n_heads=8, n_enc=2, n_dec=6, ff=2048,
                 dropout=0.1, max_codes=80, text_encoder="distilbert", vocab_size=30522):
        super().__init__()
        self.n_codes, self.max_codes = n_codes, max_codes
        self.END, self.BOS, self.PAD = n_codes, n_codes + 1, n_codes + 2
        self.text_encoder = text_encoder
        if text_encoder == "distilbert":
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
            self.bert.requires_grad_(False)
            self.bert.eval()
            self.txt_in = nn.Linear(self.bert.config.dim, d_model)
        else:                                   # (small local tests only)
            self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.txt_pos = nn.Embedding(B.MAX_TOKENS, d_model)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model, n_heads, ff, dropout, batch_first=True, norm_first=True), n_enc)
        self.code_emb = nn.Embedding(n_codes + 3, d_model, padding_idx=self.PAD)
        self.code_pos = nn.Embedding(max_codes + 2, d_model)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model, n_heads, ff, dropout, batch_first=True, norm_first=True), n_dec)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_codes + 1)        # codes 0..n_codes-1 plus End
        self.drop = nn.Dropout(dropout)

    def train(self, mode=True):
        super().train(mode)
        if self.text_encoder == "distilbert":
            self.bert.eval()                    # frozen: keep its dropout off
        return self

    def encode_text(self, ids, attn):
        if self.text_encoder == "distilbert":
            with torch.no_grad():
                h = self.bert(input_ids=ids, attention_mask=attn).last_hidden_state
            h = self.txt_in(h.float())
        else:
            h = self.tok(ids)
        h = self.drop(h + self.txt_pos(torch.arange(ids.shape[1], device=ids.device)))
        pad = attn == 0
        return self.enc(h, src_key_padding_mask=pad), pad

    def decode_step(self, inp, mem, mem_pad):
        T = inp.shape[1]
        h = self.drop(self.code_emb(inp) + self.code_pos(torch.arange(T, device=inp.device)))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=inp.device), 1)   # True = may not look
        h = self.dec(h, mem, tgt_mask=causal, tgt_key_padding_mask=inp == self.PAD,
                     memory_key_padding_mask=mem_pad)
        return self.head(self.norm(h))

    def forward(self, ids, attn, inp):
        mem, mem_pad = self.encode_text(ids, attn)
        return self.decode_step(inp, mem, mem_pad)

    @torch.no_grad()
    def generate(self, ids, attn, mode="greedy", temperature=1.0, top_k=0, min_codes=2):
        """Returns a list of int arrays (codes before End). mode: 'greedy' or 'sample'."""
        mem, mem_pad = self.encode_text(ids, attn)
        n = ids.shape[0]
        seq = torch.full((n, 1), self.BOS, dtype=torch.long, device=ids.device)
        done = torch.zeros(n, dtype=torch.bool, device=ids.device)
        lengths = torch.full((n,), self.max_codes, dtype=torch.long, device=ids.device)
        for t in range(self.max_codes):
            logits = self.decode_step(seq, mem, mem_pad)[:, -1].float()
            if t < min_codes:
                logits[:, self.END] = float("-inf")
            if mode == "greedy":
                nxt = logits.argmax(-1)
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(top_k, -1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            ended = (nxt == self.END) & ~done
            lengths[ended] = t
            done |= ended
            nxt = nxt.masked_fill(done, self.PAD)
            seq = torch.cat([seq, nxt[:, None]], 1)
            if bool(done.all()):
                break
        seq = seq[:, 1:].cpu().numpy()
        return [seq[k, :int(lengths[k])].astype(np.int64) for k in range(n)]


def make_batch(ids, attn, codes_list, model, tau_max=1.0, train=True):
    """Teacher-forcing batch. Input = [BOS] + codes (with a random fraction tau of them replaced
    by random codes when training, T2M-GPT's corruption); target = codes + [End]."""
    n, L = len(codes_list), max(len(c) for c in codes_list) + 1
    inp = np.full((n, L), model.PAD, np.int64)
    tgt = np.full((n, L), -100, np.int64)               # -100 = ignored by the loss
    for k, c in enumerate(codes_list):
        c = np.asarray(c, np.int64)
        x = c.copy()
        if train and tau_max > 0 and len(c):
            tau = np.random.uniform(0, tau_max)
            hit = np.random.rand(len(c)) < tau
            x[hit] = np.random.randint(0, model.n_codes, hit.sum())
        inp[k, 0] = model.BOS
        inp[k, 1:len(c) + 1] = x
        tgt[k, :len(c)] = c
        tgt[k, len(c)] = model.END
    t = lambda a: torch.as_tensor(a)
    return t(ids), t(attn), t(inp), t(tgt)


class CodeTextData:
    """Sentences (tokenised once) and their code sequences, cut to max_codes."""
    def __init__(self, texts, codes, tokenizer, max_codes):
        enc = tokenizer(list(texts), padding="max_length", truncation=True,
                        max_length=B.MAX_TOKENS, return_tensors="np")
        self.ids, self.attn = enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)
        self.codes = [np.asarray(c, np.int64)[:max_codes] for c in codes]

    def __len__(self):
        return len(self.codes)

    def batches(self, batch_size, shuffle=True):
        order = np.random.permutation(len(self)) if shuffle else np.arange(len(self))
        for s in range(0, len(order), batch_size):
            yield order[s:s + batch_size]


def generate_codes(model, data, idx, device, batch_size=64, **kw):
    model.eval()
    out = []
    for s in range(0, len(idx), batch_size):
        ch = np.asarray(idx[s:s + batch_size])
        ids = torch.as_tensor(data.ids[ch], device=device)
        attn = torch.as_tensor(data.attn[ch], device=device)
        out += model.generate(ids, attn, **kw)
    return out


@torch.no_grad()
def codes_to_poses(vq, codes_list, device, batch_size=64):
    """Decode code sequences with the VQ decoder -> list of (frames, 103, 2) arrays (4 frames per code)."""
    vq.eval()
    out = []
    for s in range(0, len(codes_list), batch_size):
        ch = codes_list[s:s + batch_size]
        T = max(1, max(len(c) for c in ch))
        x = np.zeros((len(ch), T), np.int64)
        for k, c in enumerate(ch):
            x[k, :len(c)] = c
        rec = vq.decode(torch.as_tensor(x, device=device)).float().cpu().numpy()
        rec = rec.reshape(len(ch), -1, B.N_POINTS, 2)
        for k, c in enumerate(ch):
            out.append(rec[k, :max(1, len(c)) * 4])
    return out


def score_both(preds, split, idx):
    """'own length' = as generated (the model chose when to stop);
    'real length' = stretched/squeezed to the real clip's length (fair across models)."""
    own, _ = B.score_predictions(preds, split, idx)
    real_len = [B.resample_to_length(p, len(split["poses"][i])) for p, i in zip(preds, idx)]
    real, _ = B.score_predictions(real_len, split, idx)
    return own, real
