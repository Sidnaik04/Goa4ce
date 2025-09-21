from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import torch, librosa, numpy as np, os, hashlib, datetime, io, soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from app.reporting import generate_report

router = APIRouter()

# ---------------- Models ---------------- #
model_names = [
    "mo-thecreator/Deepfake-audio-detection",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
]

models, feature_extractors = [], []
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
CHUNK_DURATION = 3
CHUNK_OVERLAP = 0.5

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

# ---------------- Classification ---------------- #
def classify_voice(probs):
    genuine, deepfake = probs
    if deepfake > 0.55:
        return "AI / Synthetic Voice"
    elif genuine >= 0.4 and deepfake <= 0.55:
        return "Human Voice"
    elif max(probs) < 0.4:
        return "Uncertain"
    else:
        return "Noise / Other"

def aggregate_verdict(chunk_results):
    # Use the classify_voice result of the first chunk (or main chunk) directly
    if not chunk_results:
        return "Uncertain"
    
    first_chunk = chunk_results[0]
    # Final verdict is strictly based on classify_voice for the first chunk
    return classify_voice([first_chunk['genuine_score'], first_chunk['deepfake_score']])
@router.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        audio_path = os.path.join("uploads", file.filename)
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        metadata = extract_metadata(file_bytes, file.filename)

        # Preprocess audio
        chunks = preprocess_audio(file_bytes)
        if not chunks:
            chunks = [load_audio(file_bytes)[0]]  # fallback for very short audio

        # Compute first chunk scores
        first_chunk_scores = None
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
            if idx == 0:
                first_chunk_scores = ensemble_probs

        final_verdict = classify_voice(first_chunk_scores) if first_chunk_scores is not None else "Uncertain"

        # Build report (no chunk_results)
        report = {
            "report_id": f"DFA-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
            "date_of_analysis": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyzed_by": "Automated AI Tool",
            "tool_used": model_names,
            "audio_metadata": metadata,
            "final_verdict": final_verdict
        }

        report_files = generate_report(audio_path, report)

        return JSONResponse({
            "report_data": report,
            "report_files": report_files
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
