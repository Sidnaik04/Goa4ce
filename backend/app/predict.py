# app/predict.py
import os
import io
import datetime
import hashlib
import numpy as np
import torch
import librosa
import soundfile as sf
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import webrtcvad
from concurrent.futures import ThreadPoolExecutor

# Use your report generator (adjust import path if needed)
from app.reporting import generate_report

router = APIRouter()

# ---------------- Models ---------------- #
MODEL_NAMES = [
    "mo-thecreator/Deepfake-audio-detection",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[predict] device: {device}")

# load models & extractors (module-level cache)
feature_extractors = []
models = []
loaded_names = []
for name in MODEL_NAMES:
    try:
        extractor = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModelForAudioClassification.from_pretrained(name).to(device)
        model.eval()
        feature_extractors.append(extractor)
        models.append(model)
        loaded_names.append(name)
        print(f"[predict] Loaded model: {name}")
    except Exception as e:
        print(f"[predict] ⚠️ Could not load {name}: {e}")

# if nothing loaded, we'll still return a safe response
if len(models) == 0:
    print("[predict] WARNING: No deepfake models loaded!")

# ---------------- Folders ---------------- #
os.makedirs("uploads", exist_ok=True)

# ---------------- Settings (tune these) ---------------- #
SAMPLE_RATE = 16000
CHUNK_DURATION = 3.0       # seconds per chunk
CHUNK_OVERLAP = 0.5        # 50% overlap
MIN_CHUNK_LEN = 0.5        # discard very small segments (seconds)
AVG_THRESHOLD = 0.56       # average deepfake threshold
MAX_THRESHOLD = 0.72       # any chunk above this -> likely synthetic
VOTE_THRESHOLD = 0.45      # fraction of chunks flagged to trigger synthetic
ENSEMBLE_WEIGHTS = None    # e.g. [0.7, 0.3] or None for equal weights
TEMPERATURE = 1.0          # 1.0 = no scaling; tune with dev set
VAD_AGGRESSIVENESS = 2     # 0-3 (higher = more aggressive at trimming silence)

# threadpool for CPU-bound tasks if needed
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="predict_worker")

# ---------------- Utilities ---------------- #
def _load_audio_bytes_fallback(file_bytes: bytes, target_sr: int = SAMPLE_RATE):
    """Robust audio loader: try soundfile then librosa fallback."""
    try:
        y, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
    except Exception:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    if y is None:
        return np.array([], dtype=np.float32), target_sr
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr

def _frame_generator(frame_duration_ms, audio, sample_rate):
    """Yield successive frames (bytes) from PCM int16 audio"""
    n = int(sample_rate * (frame_duration_ms / 1000.0)) * 2  # bytes for int16
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset:offset + n]
        offset += n

def vad_trim(y: np.ndarray, sr: int, aggressiveness=VAD_AGGRESSIVENESS):
    """
    Return trimmed waveform containing voiced frames using webrtcvad.
    If no voiced frames, returns empty array.
    This is a simple conservative approach; can be improved to return voice segments.
    """
    try:
        vad = webrtcvad.Vad(aggressiveness)
        # convert float waveform (-1..1) to 16-bit PCM bytes
        pcm16 = (y * 32767).astype(np.int16).tobytes()
        frames = list(_frame_generator(30, pcm16, sr))
        voiced = []
        for f in frames:
            try:
                if vad.is_speech(f, sample_rate=sr):
                    voiced.append(True)
                else:
                    voiced.append(False)
            except Exception:
                voiced.append(False)
        if not any(voiced):
            return np.array([], dtype=np.float32)
        # reconstruct audio keeping frames flagged voiced and a small padding window
        # simple: if any voiced -> return whole audio (safe); better: reconstruct segments
        # For now, return the original audio when there is speech to avoid chopping
        return y
    except Exception:
        # if vad fails, return original audio
        return y

def _chunk_audio(y: np.ndarray, sr: int, chunk_duration=CHUNK_DURATION, overlap=CHUNK_OVERLAP, min_len=MIN_CHUNK_LEN):
    """Split waveform into fixed-length chunks with overlap and pad last chunk."""
    if y is None or len(y) == 0:
        return []
    chunk_size = int(chunk_duration * sr)
    step = int(chunk_size * (1 - overlap)) if overlap < 1.0 else chunk_size
    if len(y) <= chunk_size:
        if len(y) >= int(min_len * sr):
            if len(y) < chunk_size:
                pad = np.zeros(chunk_size - len(y), dtype=np.float32)
                return [np.concatenate([y, pad])]
            return [y]
        return []
    chunks = []
    for start in range(0, len(y), step):
        end = start + chunk_size
        chunk = y[start:end]
        if len(chunk) < int(min_len * sr):
            continue
        if len(chunk) < chunk_size:
            pad = np.zeros(chunk_size - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        chunks.append(chunk)
    return chunks

def temperature_scale_probs(logits: np.ndarray, T: float = TEMPERATURE):
    """Apply temperature scaling to logits (numpy array shape (N, C)) and return probs."""
    if T == 1.0:
        exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    else:
        scaled = logits / float(T)
        exps = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
    probs = exps / np.sum(exps, axis=-1, keepdims=True)
    return probs

def ensemble_average(probs_list, weights=None):
    """Average an array of probability vectors (list of numpy arrays)."""
    arr = np.array(probs_list, dtype=float)
    if weights is None:
        return np.mean(arr, axis=0)
    weights = np.array(weights, dtype=float)
    weights = weights / np.sum(weights)
    return np.average(arr, axis=0, weights=weights)

def classify_label_from_probs(ensemble_probs, threshold=AVG_THRESHOLD):
    """Assume index 1 = deepfake. Return (label, deepfake_score, genuine_score)."""
    if len(ensemble_probs) >= 2:
        genuine_score = float(ensemble_probs[0])
        deepfake_score = float(ensemble_probs[1])
    else:
        genuine_score = float(1.0 - ensemble_probs[0])
        deepfake_score = float(ensemble_probs[0])
    if deepfake_score >= threshold:
        label = "AI / Synthetic Voice"
    elif genuine_score >= (1 - threshold):
        label = "Human Voice"
    elif max(genuine_score, deepfake_score) < 0.4:
        label = "Uncertain"
    else:
        label = "Noise / Other"
    return label, deepfake_score, genuine_score

def aggregate_synth_decision(chunk_results, avg_thresh=AVG_THRESHOLD, max_thresh=MAX_THRESHOLD, vote_thresh=VOTE_THRESHOLD):
    deepfake_scores = [c["deepfake_score"] for c in chunk_results if "deepfake_score" in c]
    if not deepfake_scores:
        return False, 0.0, "no_chunks"
    avg_score = float(np.mean(deepfake_scores))
    max_score = float(np.max(deepfake_scores))
    votes = [1 if c["deepfake_score"] >= avg_thresh else 0 for c in chunk_results]
    vote_fraction = float(sum(votes)) / len(votes)
    is_synth = (avg_score >= avg_thresh) or (max_score >= max_thresh) or (vote_fraction >= vote_thresh)
    confidence = max_score if is_synth else avg_score
    method = "max" if max_score >= max_thresh else ("avg" if avg_score >= avg_thresh else ("vote" if vote_fraction >= vote_thresh else "none"))
    return bool(is_synth), float(confidence), method

# ---------------- Endpoint ---------------- #
@router.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """
    Improved deepfake detection endpoint:
      - returns JSON: report_data + stored report files (from generate_report)
      - uses batching, VAD, ensemble, aggregation
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return JSONResponse({"error": "Empty file"}, status_code=400)

        # Save uploaded file (optional)
        audio_path = os.path.join("uploads", file.filename)
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        # Load audio robustly
        y, sr = _load_audio_bytes_fallback(file_bytes, target_sr=SAMPLE_RATE)

        # VAD trim (conservative)
        y_voiced = vad_trim(y, sr, aggressiveness=VAD_AGGRESSIVENESS)
        if y_voiced is None or len(y_voiced) == 0:
            # nothing to analyze
            return JSONResponse({
                "report_data": {
                    "report_id": f"DFA-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "date_of_analysis": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "analyzed_by": "Automated AI Tool",
                    "tool_used": MODEL_NAMES,
                    "audio_metadata": {"duration": float(len(y) / sr), "sampling_rate": sr},
                    "final_verdict": "Unclear (no voiced frames detected)",
                    "average_deepfake_score": None,
                    "average_genuine_score": None,
                    "device_used": str(device),
                    "chunks_analyzed": 0,
                    "chunk_results": []
                },
                "report_files": []
            })

        # chunk audio
        chunks = _chunk_audio(y_voiced, sr, chunk_duration=CHUNK_DURATION, overlap=CHUNK_OVERLAP)
        if not chunks:
            # fallback: use whole audio padded to chunk length
            if len(y_voiced) < int(MIN_CHUNK_LEN * sr):
                return JSONResponse({"error": "Audio too short after VAD trimming"}, status_code=400)
            chunks = _chunk_audio(y_voiced, sr, chunk_duration=CHUNK_DURATION, overlap=0.0)

        # Prepare model inference: batch chunks per model
        chunk_results = []
        per_chunk_model_probs = [[] for _ in range(len(chunks))]  # collect per-model probs for each chunk

        for extractor, model in zip(feature_extractors, models):
            # batch all chunks at once through extractor -> avoids per-chunk overhead
            try:
                inputs = extractor(chunks, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception as e:
                # If extractor can't handle lists (some older extractors expect single array), fallback to per-chunk
                inputs = None

            with torch.no_grad():
                if inputs is not None:
                    # batch infer
                    if device.type == "cuda":
                        from torch.cuda.amp import autocast
                        with autocast():
                            logits = model(**inputs).logits
                    else:
                        logits = model(**inputs).logits
                    logits = logits.cpu().numpy()  # shape (batch, classes)
                    probs = temperature_scale_probs(logits, T=TEMPERATURE)  # returns numpy array
                    # assign each chunk
                    for i in range(len(chunks)):
                        per_chunk_model_probs[i].append(probs[i])
                else:
                    # fallback per-chunk infer
                    for i, chunk in enumerate(chunks):
                        try:
                            inp = extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                            inp = {k: v.to(device) for k, v in inp.items()}
                            if device.type == "cuda":
                                from torch.cuda.amp import autocast
                                with autocast():
                                    l = model(**inp).logits
                            else:
                                l = model(**inp).logits
                            lp = l.cpu().numpy()
                            p = temperature_scale_probs(lp, T=TEMPERATURE)[0]
                        except Exception:
                            p = np.array([0.5, 0.5])
                        per_chunk_model_probs[i].append(p)

        # Ensemble per chunk and classify
        for i, chunk in enumerate(chunks):
            model_probs = per_chunk_model_probs[i]
            if len(model_probs) == 0:
                ensemble = np.array([0.5, 0.5])
            else:
                try:
                    if ENSEMBLE_WEIGHTS is None:
                        ensemble = ensemble_average(model_probs)
                    else:
                        ensemble = ensemble_average(model_probs, weights=ENSEMBLE_WEIGHTS)
                except Exception:
                    ensemble = np.mean(np.array(model_probs), axis=0)
            label, deepfake_score, genuine_score = classify_label_from_probs(ensemble, threshold=AVG_THRESHOLD)
            chunk_results.append({
                "chunk_index": i + 1,
                "genuine_score": float(genuine_score),
                "deepfake_score": float(deepfake_score),
                "prediction": label
            })

        # Aggregate across chunks
        is_synth, confidence, method = aggregate_synth_decision(chunk_results, avg_thresh=AVG_THRESHOLD, max_thresh=MAX_THRESHOLD, vote_thresh=VOTE_THRESHOLD)
        final_verdict = "🔴 AI / Synthetic Detected" if is_synth else ("🟢 Human Voice" if confidence < AVG_THRESHOLD else "🟠 Uncertain")

        avg_deepfake = float(np.mean([c["deepfake_score"] for c in chunk_results])) if chunk_results else None
        avg_genuine = float(np.mean([c["genuine_score"] for c in chunk_results])) if chunk_results else None

        # Build enriched report payload
        report_payload = {
            "report_id": f"DFA-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "date_of_analysis": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyzed_by": "Automated AI Tool",
            "tool_used": loaded_names,
            "audio_metadata": {"duration": float(len(y)/sr), "sampling_rate": sr},
            "final_verdict": final_verdict,
            "average_deepfake_score": avg_deepfake,
            "average_genuine_score": avg_genuine,
            "device_used": str(device),
            "chunks_analyzed": len(chunk_results),
            "chunk_results": chunk_results,
            "aggregation_method": method,
            "confidence": float(confidence),
        }

        # optionally generate files (json/pdf)
        try:
            report_files = generate_report(audio_path, report_payload)
        except Exception as e:
            print("[predict] generate_report failed:", e)
            report_files = []

        return JSONResponse({"report_data": report_payload, "report_files": report_files})

    except Exception as exc:
        print("[predict] error:", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
