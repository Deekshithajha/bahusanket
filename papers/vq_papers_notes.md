# Pose-code (VQ) papers: settings we copy

Read 2026-10-13 from the arXiv PDFs.

## T2M-GPT (Zhang et al., CVPR 2023): https://arxiv.org/abs/2301.06052
- VQ-VAE: 1-D CNN, 2 stride-2 downsamplings (1 code = 4 frames), dilated res blocks (9, 3, 1), width 512, codebook 512 x 512.
- EMA (0.99) codebook plus code reset. Loss: smooth-L1 recon + 0.5 x smooth-L1 velocity + 0.02 x commitment (Eq. 2-3).
- Training: 64-frame crops, AdamW (0.9, 0.99), batch 256, 200K its at lr 2e-4, then 100K at 1e-5.
- GPT: 18 layers, d 1024, 16 heads. A learned End token stops generation. Corruption: replace tau x 100% of input codes with random ones, with tau ~ U[0,1].

## T2S-GPT (Yin et al., ACL 2024): https://arxiv.org/abs/2406.07119
- Same two-stage idea for sign language, with a *dynamic* VQ-VAE (DVA-VAE) whose code length varies with information density, plus a duration transformer.
- PHOENIX14T back-translation BLEU-4 (Table 2): PT 4.38, T2M-GPT 8.01, MDM 7.55, T2S-GPT 11.87, ground truth 14.34.
- Ablation (Table 3): replacing the dynamic VQ with T2M-GPT's fixed VQ (rate 4) gives BLEU-4 8.39. Most of the gain over PT comes from discrete codes.
- Settings: K = 1024 codes, d = 512, 100K its, lr 2e-4 cosine, batch 256. GPT: d 1024, 16 heads, 18 code layers + 6 duration layers.
