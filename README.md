# Multi-Disorder Mental Health AI Diagnosis
**Tri-Modal, Severity-Aware Fusion for Depression and PTSD**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](#)
[![License](https://img.shields.io/badge/license-UNSPECIFIED-lightgrey.svg)](#)

This repo contains a fast, reproducible trainer that fuses **Text (768)** + **Audio (256)** + **Face (512)** features into a **1536-D** vector and trains **seed-ensembled XGBoost** with stratified K-fold CV.  
It outputs paper-ready logs such as `logs_paper/summary_all.csv`.

---

## ✨ What’s inside
- **`fast_multimodal.py`** – end-to-end training (feature build → CV → seed ensemble → logs)
- **Tri-modal features**
  - Text: `all-mpnet-base-v2` mean pooled (768-D, cached per PID)
  - Audio: 64-mel log-spectrogram stats + deltas (256-D)
  - Face: OpenFace numeric columns, mean+std (512-D)
- **Outputs**: `summary_all.csv`, `summary_all.json`, out-of-fold NPZs for figures (PR/ROC/CM)

---

## 🛠️ Setup

### 1) Create env & install deps
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip wheel
pip install -r requirements.txt


Datasets you need

We train on E-DAIC and DAIC-WOZ. Obtain access from the official providers and place the data locally.

Required CSVs (repo root by default)

metadata_mapped.csv (must include: Participant_ID, PTSD Severity, and optionally PHQ_Score)

Detailed_PHQ8_Labels.csv (must include: Participant_ID, PHQ_8Total)

The script prefers PHQ_8Total for depression; falls back to PHQ_Score if missing. PTSD severity is taken from metadata_mapped.csv.

Expected folder layout

You can keep the two datasets in two separate roots (names customizable via flags):

repo/
├─ fast_multimodal.py
├─ requirements.txt
├─ run_paper.sh
├─ extractor.py
├─ controller.py
├─ metadata_mapped.csv
├─ Detailed_PHQ8_Labels.csv
├─ data/                      # E-DAIC root (example)
│  ├─ 300_P/
│  │  ├─ 300_AUDIO.wav
│  │  ├─ 300_TRANSCRIPT.csv               # any *transcript*.csv works
│  │  ├─ 300_openface_clnf.csv            # any PID + (clnf|openface|au|gaze|pose)
│  │  └─ ...
│  └─ ...
└─ data_daicwoz/              # DAIC-WOZ root (example)
   ├─ 302_P/
   │  ├─ 302_AUDIO.wav
   │  ├─ 302_TRANSCRIPT.csv
   │  ├─ 302_openface_clnf.csv
   │  └─ ...
   └─ ...


