#!/usr/bin/env python3
"""
Fast multimodal depression/PTSD trainer with caching + progress bars.

- Text: sentence-transformers/all-mpnet-base-v2 mean pooled (768 dims)
- Audio: log-Mel stats (mean/std + delta mean/std) with 64 mels → 256 dims
- Face: OpenFace CSV numeric features (mean+std, padded/truncated) → 512 dims
- Fusion: [Audio 256 | Face 512 | Text 768] = 1536 dims
- CV: Stratified K-fold with optional seed-ensemble per fold
- Caching: ./cache_fast/merged_cache.npz
"""

from __future__ import annotations
import os, re, json, time, warnings, argparse, pathlib
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio",
    category=UserWarning
)

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score

from xgboost import XGBClassifier

# ------------------------- filesystem helpers -------------------------

def get_participant_dirs(root: str) -> List[str]:
    if not root or not os.path.exists(root):
        return []
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d)) and d.endswith("_P")
    ]

def find_transcript_path(pdir: str) -> Optional[str]:
    try:
        for fn in os.listdir(pdir):
            low = fn.lower()
            if "transcript" in low and low.endswith(".csv"):
                return os.path.join(pdir, fn)
    except Exception:
        pass
    return None

# ------------------------- label helpers -------------------------

def load_meta(metadata_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(metadata_path):
        return None
    try:
        return pd.read_csv(metadata_path)
    except Exception:
        return None

def load_labels(labels_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(labels_path):
        return None
    try:
        return pd.read_csv(labels_path)
    except Exception:
        return None

def build_label_maps(meta: Optional[pd.DataFrame], labels: Optional[pd.DataFrame]) -> Tuple[Dict[str,float], Dict[str,float]]:
    """
    Returns:
      dep_map[pid]  -> PHQ score (PHQ_8Total preferred; fallback to PHQ_Score)
      ptsd_map[pid] -> PTSD Severity
    """
    meta_dep: Dict[str, float] = {}
    meta_ptsd: Dict[str, float] = {}
    if meta is not None:
        for _, r in meta.iterrows():
            pid = str(r.get("Participant_ID")).strip()
            if "PHQ_Score" in r:
                meta_dep[pid] = r.get("PHQ_Score", np.nan)
            meta_ptsd[pid] = r.get("PTSD Severity", np.nan)

    label_dep: Dict[str, float] = {}
    if labels is not None:
        for _, r in labels.iterrows():
            pid = str(r.get("Participant_ID")).strip()
            label_dep[pid] = r.get("PHQ_8Total", np.nan)

    dep_map: Dict[str, float] = {}
    for pid in set(list(meta_dep.keys()) + list(label_dep.keys())):
        v = label_dep.get(pid, np.nan)
        dep_map[pid] = v if not pd.isna(v) else meta_dep.get(pid, np.nan)

    return dep_map, meta_ptsd

def categorize_depression(score: float) -> int:
    if pd.isna(score): return -1
    x = float(score)
    if x <= 4:  return 0
    if x <= 9:  return 1
    if x <= 14: return 2
    if x <= 19: return 3
    return 4

def categorize_ptsd(score: float) -> int:
    if pd.isna(score): return -1
    x = float(score)
    if x <= 20: return 0
    if x <= 40: return 1
    return 2

# ------------------------- text features -------------------------

TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 768-dim

class TextEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
        self.model = AutoModel.from_pretrained(TEXT_MODEL).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, texts: List[str], max_len: int = 256, batch_size: int = 16) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state          # [B,T,H]
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.size()).float()
            summed = (out * mask).sum(dim=1)
            denom  = mask.sum(dim=1).clamp(min=1e-6)
            sent = summed / denom
            outs.append(sent.cpu().numpy())
        return np.vstack(outs) if outs else np.zeros((0, 768), dtype=np.float32)

def read_transcript_text(pdir: str) -> str:
    tp = find_transcript_path(pdir)
    if not tp or not os.path.exists(tp):
        return ""
    try:
        df = pd.read_csv(tp)
        for c in df.columns:
            if str(c).strip().lower() in {"text", "value", "utterance"}:
                return " ".join(map(str, df[c].dropna().tolist()))
        obj = df.select_dtypes(include=[object]).fillna("").astype(str)
        return " ".join(obj.values.flatten().tolist())
    except Exception:
        return ""

# ------------------------- audio features -------------------------

def audio_logmel_stats(wav: torch.Tensor, sr: int, n_mels=64) -> np.ndarray:
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=320, f_min=20, f_max=sr//2, n_mels=n_mels
    )(wav)
    mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)
    x = mel_db.squeeze(0).float()
    delta = torchaudio.functional.compute_deltas(x.unsqueeze(0)).squeeze(0)

    def stats(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t.mean(dim=1), t.std(dim=1)], dim=0)

    feat = torch.cat([stats(x), stats(delta)], dim=0)
    feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat.cpu().numpy().astype(np.float32)  # 256 dims

def safe_audio_features(audio_path: str) -> np.ndarray:
    try:
        if not os.path.exists(audio_path):
            return np.zeros(256, dtype=np.float32)
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        return audio_logmel_stats(wav, 16000, n_mels=64)
    except Exception:
        return np.zeros(256, dtype=np.float32)

# ------------------------- face features -------------------------

def read_csv_silent(path: str) -> Optional[pd.DataFrame]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            first = fh.readline()
        sep = ";" if ";" in first else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    except Exception:
        return None

def safe_face_features(pdir: str, pid: str) -> np.ndarray:
    feats = []
    try:
        for fn in os.listdir(pdir):
            low = fn.lower()
            if not low.startswith(f"{pid.lower()}_") or "clnf" not in low:
                continue
            if low.endswith(".bin"):
                continue
            df = read_csv_silent(os.path.join(pdir, fn))
            if df is None or df.empty:
                continue
            num = df.select_dtypes(include=[float, int])
            cols = [c for c in num.columns if re.search(r"^(au|gaze|pose|eye|p_)", str(c), re.I)]
            use = num[cols] if cols else num
            m = np.nan_to_num(use.mean().to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            s = np.nan_to_num(use.std().to_numpy(),  nan=0.0, posinf=0.0, neginf=0.0)
            feats.append(np.concatenate([m, s], axis=0))
    except Exception:
        pass

    target = 512
    if not feats:
        return np.zeros(target, dtype=np.float32)

    v = np.concatenate(feats, axis=0)
    if v.shape[0] < target:
        v = np.pad(v, (0, target - v.shape[0]))
    else:
        v = v[:target]
    return v.astype(np.float32)

# ------------------------- text embed cache -------------------------

_embedder: Optional[TextEmbedder] = None

def text_embed(text: str, cache_dir: str, pid: str, device: str) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    fp = os.path.join(cache_dir, f"{pid}_text.npy")
    if os.path.exists(fp):
        try:
            v = np.load(fp)
            if v.shape == (768,):
                return v.astype(np.float32)
        except Exception:
            pass

    global _embedder
    if _embedder is None:
        _embedder = TextEmbedder(device=device)
    v = _embedder.encode_batch([text or ""], max_len=256, batch_size=1)[0]
    try:
        np.save(fp, v.astype(np.float32))
    except Exception:
        pass
    return v.astype(np.float32)

# ------------------------- participant scan -------------------------

def gather_participants(edaic_root: str, daic_root: str, max_participants: Optional[int]) -> List[Tuple[str,str,str]]:
    items = []
    for root, tag in [(edaic_root, "edaic"), (daic_root, "daicwoz")]:
        for pdir in get_participant_dirs(root):
            pid = os.path.basename(pdir).split("_")[0]
            items.append((pid, pdir, tag))
    if max_participants:
        items = items[:max_participants]
    return items

# ------------------------- cache builder -------------------------

def build_or_load_cache(
    participants: List[Tuple[str,str,str]],
    dep_map: Dict[str, float],
    ptsd_map: Dict[str, float],
    cache_dir: str,
    device: str,
    rebuild_cache: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_fp = os.path.join(cache_dir, "merged_cache.npz")

    if (not rebuild_cache) and os.path.exists(cache_fp):
        data = np.load(cache_fp, allow_pickle=True)
        return data["X"], data["y_dep"], data["y_ptsd"], data["ids"].tolist()

    for f in pathlib.Path(cache_dir).glob("*.npy"):
        try: f.unlink()
        except Exception: pass
    if os.path.exists(cache_fp):
        try: os.remove(cache_fp)
        except Exception: pass

    print("Stage 1/4: Scanning participants…")
    items = participants
    print(f"Found {len(items)} participant folders across datasets.")

    print("Stage 2/4: Collecting transcripts + audio/face features…")
    X_list: List[np.ndarray] = []
    y_dep: List[int] = []
    y_ptsd: List[int] = []
    ids: List[str] = []

    pbar = tqdm(items, total=len(items), ncols=100, leave=False)
    for pid, pdir, _tag in pbar:
        pbar.set_description(f"PID {pid}")

        dep_score = dep_map.get(pid, np.nan)
        ptsd_score = ptsd_map.get(pid, np.nan)
        dep_c = categorize_depression(dep_score)
        ptsd_c = categorize_ptsd(ptsd_score)
        if dep_c < 0 or ptsd_c < 0:
            continue

        text = read_transcript_text(pdir)
        t_vec = text_embed(text, cache_dir=cache_dir, pid=pid, device=device)  # 768
        audio_path = os.path.join(pdir, f"{pid}_AUDIO.wav")
        a_vec = safe_audio_features(audio_path)                                # 256
        f_vec = safe_face_features(pdir, pid)                                  # 512

        vec = np.concatenate([a_vec, f_vec, t_vec], axis=0)  # 1536
        if vec.shape[0] != 1536:
            z = np.zeros(1536, dtype=np.float32)
            k = min(1536, vec.shape[0]); z[:k] = vec[:k]; vec = z

        X_list.append(vec.astype(np.float32))
        y_dep.append(dep_c)
        y_ptsd.append(ptsd_c)
        ids.append(pid)

    if not X_list:
        raise RuntimeError("No valid participants found after filtering (labels/features).")

    print("Stage 3/4: Building text embeddings… (cached during collection)")
    print("Stage 4/4: Assembling features + writing caches…")
    X = np.vstack(X_list).astype(np.float32)
    y_dep = np.asarray(y_dep, dtype=np.int64)
    y_ptsd = np.asarray(y_ptsd, dtype=np.int64)

    print(f"Final feature matrix: X.shape={X.shape} | dep classes={np.unique(y_dep)} | ptsd classes={np.unique(y_ptsd)}")
    np.savez_compressed(cache_fp, X=X, y_dep=y_dep, y_ptsd=y_ptsd, ids=np.array(ids, dtype=object))
    return X, y_dep, y_ptsd, ids

# ------------------------- seed-ensembled CV -------------------------

def build_xgb(num_class: int, seed: int, use_gpu: bool):
    params = dict(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=2.0,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    try:
        if use_gpu and torch.cuda.is_available():
            params["device"] = "cuda"
    except Exception:
        pass
    return XGBClassifier(**params)

def run_cv_ensembled(X: np.ndarray,
                     y: np.ndarray,
                     num_class: int,
                     seeds: List[int],
                     folds: int = 5,
                     use_gpu: bool = False,
                     tag: str = "",
                     outdir: str = "logs_paper"):
    os.makedirs(outdir, exist_ok=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=min(seeds) if seeds else 42)
    oof_proba = np.zeros((len(y), num_class), dtype=np.float32)

    for fi, (tr, va) in enumerate(skf.split(X, y), 1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        ytr = y[tr]
        yva = y[va]

        bc = np.bincount(ytr, minlength=num_class).astype(np.float32)
        sw = (bc.sum() / np.maximum(bc, 1.0))[ytr]

        proba_accum = np.zeros((len(va), num_class), dtype=np.float32)
        for s in seeds:
            mdl = build_xgb(num_class=num_class, seed=s, use_gpu=use_gpu)
            try:
                mdl.fit(
                    Xtr, ytr,
                    sample_weight=sw,
                    eval_set=[(Xva, yva)],
                    early_stopping_rounds=20,
                    verbose=False
                )
            except TypeError:
                mdl.set_params(n_estimators=400)
                mdl.fit(Xtr, ytr, sample_weight=sw, eval_set=[(Xva, yva)], verbose=False)

            proba_accum += mdl.predict_proba(Xva).astype(np.float32)

        proba_accum /= max(1, len(seeds))
        oof_proba[va] = proba_accum

    oof_pred = oof_proba.argmax(axis=1)
    results = {
        "acc": float(accuracy_score(y, oof_pred)),
        "f1_weighted": float(f1_score(y, oof_pred, average="weighted")),
        "recall_weighted": float(recall_score(y, oof_pred, average="weighted")),
    }
    np.savez_compressed(os.path.join(outdir, f"oof_{tag or 'task'}.npz"),
                        proba=oof_proba, pred=oof_pred, y=y.astype(int))
    return results, oof_proba

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edaic_root", type=str, default="data")
    ap.add_argument("--daicwoz_root", type=str, default="data daicwoz")
    ap.add_argument("--metadata", type=str, default="metadata_mapped.csv")
    ap.add_argument("--labels", type=str, default="Detailed_PHQ8_Labels.csv")
    ap.add_argument("--max_participants", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rebuild_cache", type=int, default=0)
    ap.add_argument("--cache_dir", type=str, default="cache_fast")
    ap.add_argument("--use_gpu", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="logs_paper")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = ap.parse_args()

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")

    meta = load_meta(args.metadata)
    labels = load_labels(args.labels)
    dep_map, ptsd_map = build_label_maps(meta, labels)

    participants = gather_participants(args.edaic_root, args.daicwoz_root, args.max_participants)

    X, y_dep, y_ptsd, ids = build_or_load_cache(
        participants, dep_map, ptsd_map, args.cache_dir, device, rebuild_cache=bool(args.rebuild_cache)
    )

    print(f"Feature matrix: {X.shape}, dep classes={np.unique(y_dep)}, ptsd classes={np.unique(y_ptsd)}")
    
    # --- sanity checks ---
    def pct_zero_rows(Xsub):
        return float((np.all(np.isclose(Xsub, 0), axis=1)).mean())
    
    print("\n[Sanity] Label distributions")
    for name, y in [("DEP", y_dep), ("PTSD", y_ptsd)]:
        uniq, cnt = np.unique(y, return_counts=True)
        maj = cnt.max() / cnt.sum()
        print(f"{name}: {dict(zip(uniq.tolist(), cnt.tolist()))} | majority baseline≈{maj:.4f}")
    
    print("\n[Sanity] Percent of rows that are all-zeros by modality")
    print(f"audio zeros: {pct_zero_rows(X[:, slice(0,256)]):.3f}")
    print(f"face  zeros: {pct_zero_rows(X[:, slice(256,256+512)]):.3f}")
    print(f"text  zeros: {pct_zero_rows(X[:, slice(256+512,1536)]):.3f}")
    

    # modality slices
    sl_audio = slice(0, 256)
    sl_face  = slice(256, 256+512)
    sl_text  = slice(256+512, 1536)

    X_audio = X[:, sl_audio]
    X_face  = X[:, sl_face]
    X_text  = X[:, sl_text]
    X_AT    = np.concatenate([X_audio, X_text], axis=1)
    X_AF    = np.concatenate([X_audio, X_face], axis=1)
    X_TF    = np.concatenate([X_text,  X_face], axis=1)
    X_all   = X

    summary_rows = []

    def run_and_log(task_name, Xsub, y, K, tag_label):
        res, _ = run_cv_ensembled(Xsub, y, num_class=K, seeds=args.seeds,
                                  folds=args.folds, use_gpu=bool(args.use_gpu),
                                  tag=tag_label, outdir=args.outdir)
        print(f"{task_name} – {tag_label}: acc={res['acc']:.4f}  f1w={res['f1_weighted']:.4f}  rec={res['recall_weighted']:.4f}")
        summary_rows.append({
            "task": task_name, "modality": tag_label,
            "acc": res["acc"], "f1_weighted": res["f1_weighted"], "recall_weighted": res["recall_weighted"]
        })

    # ===== DEP =====
    print("\n==== DEP ====")
    run_and_log("DEP", X_text,  y_dep, 5, "TEXT")
    run_and_log("DEP", X_audio, y_dep, 5, "AUDIO")
    run_and_log("DEP", X_face,  y_dep, 5, "FACE")
    run_and_log("DEP", X_AT,    y_dep, 5, "AUDIO+TEXT")
    run_and_log("DEP", X_AF,    y_dep, 5, "AUDIO+FACE")
    run_and_log("DEP", X_TF,    y_dep, 5, "TEXT+FACE")
    run_and_log("DEP", X_all,   y_dep, 5, "ALL")

    # ===== PTSD =====
    print("\n==== PTSD ====")
    run_and_log("PTSD", X_text,  y_ptsd, 3, "TEXT")
    run_and_log("PTSD", X_audio, y_ptsd, 3, "AUDIO")
    run_and_log("PTSD", X_face,  y_ptsd, 3, "FACE")
    run_and_log("PTSD", X_AT,    y_ptsd, 3, "AUDIO+TEXT")
    run_and_log("PTSD", X_AF,    y_ptsd, 3, "AUDIO+FACE")
    run_and_log("PTSD", X_TF,    y_ptsd, 3, "TEXT+FACE")
    run_and_log("PTSD", X_all,   y_ptsd, 3, "ALL")

    # Print a neat table and save CSV
    df_sum = pd.DataFrame(summary_rows)
    print("\n=== Summary (all streams) ===")
    print(df_sum.pivot(index="modality", columns="task", values="acc").round(4).fillna(""))

    os.makedirs(args.outdir, exist_ok=True)
    df_sum.to_csv(os.path.join(args.outdir, "summary_all.csv"), index=False)
    with open(os.path.join(args.outdir, "summary_all.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nSaved per-stream results to {args.outdir}/summary_all.csv and summary_all.json")
    print("\nDone.")

if __name__ == "__main__":
    main()
