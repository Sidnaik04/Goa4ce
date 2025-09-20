from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import librosa
import numpy as np
import os
import hashlib
import datetime
import io
import soundfile as sf
import matplotlib.pyplot as plt
import base64
import librosa.display
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

router = APIRouter()

# ---------------- Models ---------------- #
model_names = [
    "mo-thecreator/Deepfake-audio-detection",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
]

models = []
feature_extractors = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for name in model_names:
    try:
        extractor = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModelForAudioClassification.from_pretrained(name).to(device)
        models.append(model)
        feature_extractors.append(extractor)
        print(f"Loaded model: {name}")
    except Exception as e:
        print(f"⚠️ Could not load {name}: {e}")

# ---------------- Folders ---------------- #
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ---------------- Settings ---------------- #
SAMPLE_RATE = 16000
CHUNK_DURATION = 3        # seconds per chunk
CHUNK_OVERLAP = 0.5       # 50% overlap

# ---------------- Utilities ---------------- #
def load_audio(file_bytes, target_sr=SAMPLE_RATE):
    try:
        y, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
    except:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr

def preprocess_audio(file_bytes):
    y, sr = load_audio(file_bytes, target_sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=20)
    step = int(CHUNK_DURATION * SAMPLE_RATE * (1 - CHUNK_OVERLAP))
    chunks = []
    for start in range(0, len(y), step):
        end = start + int(CHUNK_DURATION * SAMPLE_RATE)
        chunk = y[start:end]
        if len(chunk) > SAMPLE_RATE:
            chunks.append(chunk)
    return chunks

def extract_metadata(file_bytes, filename):
    y, sr = load_audio(file_bytes, target_sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)
    fmt = os.path.splitext(filename)[1][1:]
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return {
        "file_name": filename,
        "file_format": fmt,
        "duration": f"{int(duration//60)} min {int(duration%60)} sec",
        "sampling_rate": f"{sr} Hz",
        "md5_hash": file_hash,
        "date_of_file_creation": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M GMT")
    }

def generate_heatmap(audio, sr, probs, chunk_index, filename):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Chunk {chunk_index+1} Deepfake Score: {probs[1]:.2f}")
    img_path = os.path.join("reports", f"{filename}_heatmap_chunk{chunk_index+1}.png")
    plt.savefig(img_path)
    plt.close()
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# ---------------- Voice Classification ---------------- #
def classify_voice(probs):
    genuine, deepfake = probs

    # Clearly AI
    if deepfake > 0.55:
        return "AI / Synthetic Voice"
    
    # Clearly Human or borderline (anything not AI and not extremely low confidence)
    elif genuine >= 0.4 and deepfake <= 0.55:
        return "Human Voice"
    
    # Low confidence, unclear
    elif max(probs) < 0.4:
        return "Uncertain"
    
    # Any other rare case
    else:
        return "Noise / Other"



def aggregate_verdict(chunk_results):
    # If any chunk is AI, mark AI
    if any(c['prediction'] == "AI / Synthetic Voice" for c in chunk_results):
        return "🔴 AI / Synthetic Detected"
    if any(c['prediction'] == "Human Voice" for c in chunk_results):
        return "🟢 Human Voice"
    if any(c['prediction'] == "Noise / Other" for c in chunk_results):
        return "🟡 Noise / Other"
    return "🟠 Uncertain"

# ---------------- Predict Endpoint ---------------- #
@router.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        audio_path = os.path.join("uploads", file.filename)
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        metadata = extract_metadata(file_bytes, file.filename)
        chunks = preprocess_audio(file_bytes)
        chunk_results = []

        for idx, chunk in enumerate(chunks):
            model_probs = []
            for extractor, model in zip(feature_extractors, models):
                inputs = extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                model_probs.append(probs)

            ensemble_probs = np.mean(model_probs, axis=0)
            predicted_label = classify_voice(ensemble_probs)
            heatmap_b64 = generate_heatmap(chunk, SAMPLE_RATE, ensemble_probs, idx, file.filename)

            chunk_results.append({
                "chunk_index": idx+1,
                "prediction": predicted_label,
                "genuine_score": float(ensemble_probs[0]),
                "deepfake_score": float(ensemble_probs[1]),
                "heatmap": heatmap_b64
            })

        final_verdict = aggregate_verdict(chunk_results)

        report = {
            "report_id": f"DFA-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "date_of_analysis": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyzed_by": "Automated AI Tool",
            "tool_used": model_names,
            "audio_metadata": metadata,
            "chunk_results": chunk_results,
            "final_verdict": final_verdict
        }

        return JSONResponse(report)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
