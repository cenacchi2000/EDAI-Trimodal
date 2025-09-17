import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
import torch.nn.functional as F

# ─── LOAD MODELS ONCE ───────────────────────────────────────────────────────
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)
deberta_model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
deberta_model.eval()

emotion_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotion_model.eval()

# ─── TRANSCRIPT ENCODING ────────────────────────────────────────────────────

def get_speaker_turns(df):
    df = df.dropna(subset=["value", "speaker"])
    speaker_turns = []
    for _, row in df.iterrows():
        speaker = str(row["speaker"])
        sentences = sent_tokenize(str(row["value"]))
        for sentence in sentences:
            if sentence.strip():
                speaker_turns.append((speaker, sentence.strip()))
    return speaker_turns

def get_emotion_logits(text):
    try:
        inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
        return logits.squeeze().numpy()  # shape: (7,)
    except Exception:
        return np.zeros(7)

def encode_transcript_advanced(transcript_path):
    if not os.path.exists(transcript_path):
        return np.zeros(1032)

    try:
        df = pd.read_csv(transcript_path)
        speaker_turns = get_speaker_turns(df)
        all_vectors = []

        for speaker, sentence in speaker_turns:
            try:
                inputs = deberta_tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = deberta_model(**inputs)
                    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # (1024,)
            except Exception:
                cls_embedding = np.zeros(1024)

            emo_logits = get_emotion_logits(sentence)  # (7,)
            speaker_flag = np.array([1.0 if speaker.lower().startswith("participant") else 0.0])  # (1,)

            full_vector = np.concatenate([cls_embedding, emo_logits, speaker_flag])  # (1032,)
            all_vectors.append(full_vector)

        if not all_vectors:
            return np.zeros(1032)

        return np.mean(all_vectors, axis=0)

    except Exception as e:
        print(f"[Transcript Error] {transcript_path}: {e}")
        return np.zeros(1032)

# ─── FEATURE FILE LOADER ────────────────────────────────────────────────────

def load_feature_file(path):
    """Load CSV features with mean pooling fallback"""
    if os.path.exists(path):
        try:
            return pd.read_csv(path).mean().values
        except Exception:
            return np.zeros(50)
    else:
        return np.zeros(50)

# ─── MAIN FEATURE EXTRACTOR ─────────────────────────────────────────────────

def extract_all_features(meta_df, labels_df, participant_id, data_root="data"):
    base_path = os.path.join(data_root, f"{participant_id}_P", "features")
    
    feature_files = {
        "boaw_mfcc": f"{base_path}/{participant_id}_BoAW_openSMILE_2.3.0_MFCC.csv",
        "boaw_egemaps": f"{base_path}/{participant_id}_BoAW_openSMILE_2.3.0_eGeMAPS.csv",
        "densenet201": f"{base_path}/{participant_id}_densenet201.csv",
        "vgg16": f"{base_path}/{participant_id}_vgg16.csv",
        "bovw_posegazeau": f"{base_path}/{participant_id}_BoVW_openFace_2.1.0_Pose_Gaze_AUs.csv",
        "openface_posegazeau": f"{base_path}/{participant_id}_OpenFace2.1.0_Pose_gaze_AUs.csv",
    }

    audio_features = []
    visual_features = []

    for key, filepath in feature_files.items():
        if not os.path.exists(filepath):
            print(f"[Warning] Missing file: {filepath}")
            continue

        try:
            try:
                df = pd.read_csv(filepath, sep=';')
                if df.shape[1] == 1:
                    df = pd.read_csv(filepath, sep=',')
            except Exception:
                df = pd.read_csv(filepath)

            df_numeric = df.select_dtypes(include=[np.number])
            print(f"[Loaded] {key}: {filepath} -> shape {df_numeric.shape}")

            if df_numeric.empty:
                raise ValueError(f"No numeric data extracted from {filepath}")

            mean_features = df_numeric.mean().values

            if key in ["boaw_mfcc", "boaw_egemaps", "densenet201", "vgg16"]:
                audio_features.append(mean_features)
            else:
                visual_features.append(mean_features)

        except Exception as e:
            raise RuntimeError(f"[Error] Failed to process {filepath}: {e}")

    if not audio_features:
        raise ValueError("❌ No usable audio features found.")
    if not visual_features:
        raise ValueError("❌ No usable visual features found.")

    X_audio = np.concatenate(audio_features)
    X_face = np.concatenate(visual_features)
    transcript_path = f"data/{participant_id}_P/transcripts/{participant_id}_TRANSCRIPT.csv"
    X_text = encode_transcript_advanced(transcript_path)

    labels_df.columns = labels_df.columns.str.strip()
    row = labels_df[labels_df['Participant_ID'] == int(participant_id)]
    if row.empty:
        raise ValueError(f"❌ Label not found for participant: {participant_id}")
    y = row['PHQ_Binary'].values[0]

    print(f"\n✅ Final shapes: X_face={X_face.shape}, X_audio={X_audio.shape}, X_text={X_text.shape}")
    return X_face, X_audio, X_text, y

def extract_all_features_batch(meta_df, labels_df, participant_ids):
    
    if "PHQ_Binary" not in labels_df.columns:
            labels_df = pd.read_csv("Detailed_PHQ8_Labels.csv")
            labels_df.columns = labels_df.columns.str.strip()
            labels_df["Participant_ID"] = labels_df["Participant_ID"].astype(str).str.strip()
    
    X, y = [], []
    
    for pid in participant_ids:
            pid_str = str(pid).strip()
            # Lookup label row
            row = labels_df[labels_df["Participant_ID"].astype(str).str.strip() == pid_str]
            if row.empty:
                print(f"[Skip] Participant {pid_str}: ID not found in labels")
                continue
            # Extract binary label
            try:
                dep_binary = int(row["PHQ_Binary"].values[0])
            except Exception:
                print(f"[Skip] Participant {pid_str}: 'PHQ_Binary' missing or invalid")
                continue
    
            # Try both possible data roots for feature lookup
            features = None
            for root in ("data", "data daicwoz"):
                try:
                    features = extract_all_features(meta_df, labels_df, pid_str, data_root=root)
                    break
                except Exception:
                    features = None
            if features is None:
                print(f"[Skip] Participant {pid_str}: no feature files found in data roots")
                continue
    
            # Concatenate and collect
            X_face, X_audio, X_text, _ = features
            X.append(np.concatenate([X_face, X_audio, X_text]))
            y.append(dep_binary)
    
    if not X:
        raise ValueError("❌ No features extracted — check feature files or IDs")
    
    return np.vstack(X), np.array(y)




# ─── COMPATIBILITY PATCH ─────────────────────────────────────────────────────
def extract_text_features(transcript_path):
    return encode_transcript_advanced(transcript_path)
