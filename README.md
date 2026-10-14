# BahuSanket

Multilingual (English / Hindi / Kannada) text to continuous Indian Sign Language (ISL) pose generation, including non-manual features (eyebrows, mouth, head), gloss-free at inference.

M.Tech CSE capstone, PES University (PW_25_CGM_05), guide Prof. Chitra G M, 2025-27.

Anchor papers: [iSign](https://arxiv.org/abs/2407.05404) (dataset and benchmark) and [Progressive Transformers](https://arxiv.org/abs/2004.14874) (text-to-pose model).

## What is in this repo

Code only. Data, pose files, splits, indices and checkpoints stay in the Google Drive folder `BahuSanket` and are kept out of git by `.gitignore`.

| Path | What it is |
| --- | --- |
| `code/bahusanket_lib.py` | Shared data loading, the Progressive-Transformer-style model, and the DTW-MJE pose-distance metric |
| `code/bahusanket_vq.py` | Pose-code VQ-VAE tokenizer (turns pose sequences into discrete codes) |
| `code/bahusanket_t2c.py` | Text-to-pose-codes model |
| `code/isign_stream.py` | Streams single `.pose` files out of the iSign zip on Hugging Face with HTTP range requests (never downloads the 170 GB archive) |
| `02_text_to_pose_baseline.ipynb` | Baseline 1: text to pose |
| `03_velocity_and_rescore.ipynb` | Velocity/acceleration smoothness loss and rescoring |
| `04_fair_scoring.ipynb` | Fair comparison against the retrieval baseline |
| `05_pose_vocabulary.ipynb` | Training the pose-code VQ tokenizer |
| `06_text_to_codes.ipynb` | Text to pose codes |
| `make_notebook*.py` | Scripts that generate the notebooks above (edit these, then regenerate) |
| `papers/` | Reading notes |

## Running

The notebooks run on Colab with a GPU. Each one mounts Drive, finds the `BahuSanket` folder, puts `BahuSanket/code` on the Python path, and reloads `splits.json`, `model_point_indices.json` and `excluded_uids.json` from Drive at the start.

## Data

[iSign v1.1](https://huggingface.co/datasets/Exploration-Lab/iSign) (CC-BY-NC-SA-4.0). Working subset: 854 videos, 18,026 clips, video-level train/val/test split. Poses trimmed from 543 MediaPipe points to 103 (7 body, 54 face, 42 hands), normalized to shoulder midpoint and shoulder width.
