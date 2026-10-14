# Literature survey

Papers BahuSanket has used or built on, with what each one gave the project. PDFs are in [`papers/`](papers/) (downloaded from arXiv, for personal study). Reading notes are in [`../papers/vq_papers_notes.md`](../papers/vq_papers_notes.md).

## 1. Dataset and task

| # | Paper | Venue | Link | How it helped |
| --- | --- | --- | --- | --- |
| 01 | **iSign: A Benchmark for Indian Sign Language Processing** (Joshi et al.) | ACL 2024 | [arXiv 2407.05404](https://arxiv.org/abs/2407.05404) | Our primary dataset (127K English-ISL pairs, CC-BY-NC-SA-4.0) and the Text2Pose task we work on. Our 854-video / 18,026-clip subset comes from it. |

## 2. Sign language production (text to signing)

| # | Paper | Venue | Link | How it helped |
| --- | --- | --- | --- | --- |
| 02 | **Progressive Transformers for End-to-End Sign Language Production** (Saunders et al.) | ECCV 2020 | [arXiv 2004.14874](https://arxiv.org/abs/2004.14874) | Anchor paper. The first end-to-end text-to-continuous-pose model. Our Baseline 1 (notebooks 02-04) follows its design, and its DTW-based scoring and back-translation evaluation shaped ours. |
| 03 | **Signing at Scale: Learning to Co-Articulate Signs for Large-Scale Photo-Realistic SLP** (Saunders et al.) | CVPR 2022 | [arXiv 2203.15354](https://arxiv.org/abs/2203.15354) | Shows how to learn smooth transitions (co-articulation) between signs, the core of our "continuous" goal. Cited in the Review 1 survey. |
| 04 | **Ham2Pose: Animating Sign Language Notation into Pose Sequences** (Shalev-Arkushin et al.) | CVPR 2023 | [arXiv 2211.13613](https://arxiv.org/abs/2211.13613) | Language-agnostic pose generation from HamNoSys notation; a reference point for pose-generation metrics. Cited in the Review 1 survey. |
| 07 | **T2S-GPT: Dynamic Vector Quantization for Autoregressive Sign Language Production from Text** (Yin et al.) | ACL 2024 | [arXiv 2406.07119](https://arxiv.org/abs/2406.07119) | Evidence that discrete pose codes beat Progressive Transformers on sign language (PHOENIX14T BLEU-4 11.87 vs 4.38). Main reason we moved to pose codes (notebooks 05-06). |

## 3. Discrete pose codes (VQ-VAE + GPT)

| # | Paper | Venue | Link | How it helped |
| --- | --- | --- | --- | --- |
| 05 | **Neural Discrete Representation Learning** (van den Oord et al.) | NeurIPS 2017 | [arXiv 1711.00937](https://arxiv.org/abs/1711.00937) | The original VQ-VAE: learns a fixed "vocabulary" of codes. Basis of `code/bahusanket_vq.py`. |
| 06 | **T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations** (Zhang et al.) | CVPR 2023 | [arXiv 2301.06052](https://arxiv.org/abs/2301.06052) | The recipe we copied for the pose vocabulary (512 codes, 4 frames per code, EMA + code reset, velocity loss) and for text-to-codes (End token, input-code corruption). |

## 4. Text encoders

| # | Paper | Venue | Link | How it helped |
| --- | --- | --- | --- | --- |
| 08 | **DistilBERT, a distilled version of BERT** (Sanh et al.) | NeurIPS 2019 EMC² workshop | [arXiv 1910.01108](https://arxiv.org/abs/1910.01108) | Our text tokenizer (28 tokens) and the frozen word vectors used in notebook 06. |
| 09 | **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** (Reimers & Gurevych) | EMNLP 2019 | [arXiv 1908.10084](https://arxiv.org/abs/1908.10084) | The sentence-transformers method behind Baseline 0 (retrieval). |
| 10 | **MiniLM: Deep Self-Attention Distillation** (Wang et al.) | NeurIPS 2020 | [arXiv 2002.10957](https://arxiv.org/abs/2002.10957) | The `all-MiniLM-L6-v2` model Baseline 0 uses to embed sentences. |

## 5. Pose extraction and tooling

| # | Paper | Venue | Link | How it helped |
| --- | --- | --- | --- | --- |
| 11 | **MediaPipe: A Framework for Building Perception Pipelines** (Lugaresi et al.) | CVPR 2019 workshop | [arXiv 1906.08172](https://arxiv.org/abs/1906.08172) | iSign poses are MediaPipe Holistic keypoints (543 points), which we trim to 103. |
| 12 | **pose-format: Library for Viewing, Augmenting, and Handling .pose Files** (Moryossef & Müller) | arXiv 2023 | [arXiv 2310.09066](https://arxiv.org/abs/2310.09066) | The library and file format used to read iSign `.pose` files (`code/isign_stream.py`). |

## 6. Related ISL and multilingual work

| # | Paper | Venue | Link | How it helped |
| --- | --- | --- | --- | --- |
| 13 | **Multilingual Sign Language Translation with Unified Datasets and Pose-Based Transformers** (Dal Bianco et al.) | WSLP 2025 (IJCNLP-AACL) | [ACL Anthology 2025.wslp-main.5](https://aclanthology.org/2025.wslp-main.5/) | Multilingual across 4 sign languages (including ISL), but translates signing to text, not text to signing. Shows the multilingual gap on the generation side. |
| 14 | **Towards Scalable Sign Production: Leveraging Co-Articulated Gloss Dictionary for Fluid Sign Synthesis** (SanketVaani-1K; Agrawal, Mazumder, Jawahar, Namboodiri) | ICVGIP 2025 | [IIIT Hyderabad PDF](https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2025/Sign_LangSynthesis_ICVGIP.pdf) | Rival: gloss-driven, single-language ISL production with co-articulated transitions (1,000 interpreted sentences, FramePack interpolation). |
| 15 | **A Gloss-driven Indian Sign Language Production System Using Learned Pose Representations** (Patra, Maitra, Punyeshwarananda, Samanta) | IJCAI 2026 | [ijcai.org/proceedings/2026/823](https://www.ijcai.org/proceedings/2026/823) | Closest existing ISL production system and our main rival: a Sign-Pose-VQ-VAE with separate body/hand codes, interpreter-rated 4.33/5. Gloss-driven, Hindi input. |
| 16 | **An Automatic Conversion of Punjabi Text to Indian Sign Language** (Dhanjal & Singh) | EAI Trans. Scalable Information Systems 2020 | [EUDL](https://eudl.eu/doi/10.4108/eai.13-7-2018.165279) | Earlier rule-based Punjabi-text-to-ISL 3D animation by Dhanjal & Singh (the 2025 Springer system below is cited under the same names). Shows the rule-based approach we move away from. |

## Not yet in `papers/` (Deekshitha to download)

| Paper | Venue | Where to get it | Why it matters |
| --- | --- | --- | --- |
| **Towards Fast and High-Quality Sign Language Production** (Huang et al.) | ACM MM 2021, pp. 3172-3181 | [ACM Digital Library](https://dl.acm.org) (search the title; PES library login) or [Google Scholar](https://scholar.google.com) | Introduced the DTW-MJE metric we use for scoring. |
| Multilingual speech to ISL with synthetic animation (Dhanjal & Singh) | Springer 2025 | Not found online yet; exact title needed | English/Hindi/Punjabi to ISL avatar, rule-based rather than learned. |
