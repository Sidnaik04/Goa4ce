from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import jwt
import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import shutil
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import librosa
import numpy as np
import io
import soundfile as sf
import matplotlib.pyplot as plt
import base64
import librosa.display
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app import models, schemas, crud, ml_stubs, utils
from app.database import SessionLocal, engine

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="ps3 ka Backend", version="1.0.0")

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET", "hackathon-secret-key-2025")
HMAC_KEY = os.getenv("HMAC_KEY", "hmac-signing-key-2024")
DEFAULT_USER = os.getenv("DEFAULT_USER", "officer")
DEFAULT_PASS = os.getenv("DEFAULT_PASS", "police123")

# make directories
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/spectrograms", exist_ok=True)
os.makedirs("data/reports", exist_ok=True)
os.makedirs("reports", exist_ok=True)

ml_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml_worker")

print("🤖 Loading deepfake detection models...")
deepfake_models = []
deepfake_extractors = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_names = [
    "mo-thecreator/Deepfake-audio-detection",
    "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
]

for name in model_names:
    try:
        extractor = AutoFeatureExtractor.from_pretrained(name)
        model = AutoModelForAudioClassification.from_pretrained(name).to(device)
        deepfake_models.append(model)
        deepfake_extractors.append(extractor)
        print(f"✅ Loaded deepfake model: {name}")
    except Exception as e:
        print(f"⚠️ Could not load deepfake model {name}: {e}")

# Deepfake detection settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 3
CHUNK_OVERLAP = 0.5


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Auth dependency
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# load and preprocess audio for deepfake detection
def load_audio_for_deepfake(file_bytes, target_sr=SAMPLE_RATE):
    try:
        y, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
    except:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)

    if len(y.shape) > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr


# preprocess audio into chunks for analysis
def preprocess_audio_chunks(file_bytes):
    y, sr = load_audio_for_deepfake(file_bytes, target_sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=20)

    step = int(CHUNK_DURATION * SAMPLE_RATE * (1 - CHUNK_OVERLAP))
    chunks = []

    for start in range(0, len(y), step):
        end = start + int(CHUNK_DURATION * SAMPLE_RATE)
        chunk = y[start:end]
        if len(chunk) > SAMPLE_RATE:
            chunks.append(chunk)

    return chunks


# classify voice based on deepfake probabilities
def classify_voice_deepfake(probs):
    genuine, deepfake = probs

    if deepfake > 0.55:
        return "AI / Synthetic Voice"
    elif genuine >= 0.4 and deepfake <= 0.55:
        return "Human Voice"
    elif max(probs) < 0.4:
        return "Uncertain"
    else:
        return "Noise / Other"


# aggregate deepfake detection results
def aggregate_deepfake_verdict(chunk_results):
    if any(c["prediction"] == "AI / Synthetic Voice" for c in chunk_results):
        return "🔴 AI / Synthetic Detected"
    if any(c["prediction"] == "Human Voice" for c in chunk_results):
        return "🟢 Human Voice"
    if any(c["prediction"] == "Noise / Other" for c in chunk_results):
        return "🟡 Noise / Other"
    return "🟠 Uncertain"


# heatmap for deepfake analysis
def generate_deepfake_heatmap(audio, sr, probs, chunk_index, filename):
    try:
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(8, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Chunk {chunk_index+1} Deepfake Score: {probs[1]:.2f}")

        # ✅ Fixed: Ensure directory exists and use proper path
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        img_path = os.path.join(
            "reports", f"{safe_filename}_heatmap_chunk{chunk_index+1}.png"
        )

        # Ensure the reports directory exists
        os.makedirs("reports", exist_ok=True)

        plt.savefig(img_path)
        plt.close()

        # Read and encode the image
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    except Exception as e:
        print(f"Failed to generate heatmap: {e}")
        plt.close()  # Make sure to close the figure even if it fails
        return ""


# run comprehensive deepfake detection
async def run_deepfake_detection(file_bytes, filename):
    try:
        chunks = preprocess_audio_chunks(file_bytes)
        chunk_results = []

        for idx, chunk in enumerate(chunks):
            model_probs = []

            for extractor, model in zip(deepfake_extractors, deepfake_models):
                inputs = extractor(
                    chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                model_probs.append(probs)

            ensemble_probs = np.mean(model_probs, axis=0)
            predicted_label = classify_voice_deepfake(ensemble_probs)
            heatmap_b64 = generate_deepfake_heatmap(
                chunk, SAMPLE_RATE, ensemble_probs, idx, filename
            )

            chunk_results.append(
                {
                    "chunk_index": idx + 1,
                    "prediction": predicted_label,
                    "genuine_score": float(ensemble_probs[0]),
                    "deepfake_score": float(ensemble_probs[1]),
                    "heatmap": heatmap_b64,
                }
            )

        final_verdict = aggregate_deepfake_verdict(chunk_results)

        # Calculate overall confidence
        avg_deepfake_score = np.mean([c["deepfake_score"] for c in chunk_results])
        is_synthetic = final_verdict in ["🔴 AI / Synthetic Detected"]

        return {
            "is_synthetic": is_synthetic,
            "confidence": avg_deepfake_score,
            "verdict": final_verdict,
            "chunk_results": chunk_results,
            "model_info": {
                "models_used": model_names,
                "device": str(device),
                "chunks_analyzed": len(chunk_results),
            },
        }

    except Exception as e:
        print(f"Deepfake detection error: {e}")
        return {
            "is_synthetic": False,
            "confidence": 0.5,
            "verdict": "🟠 Error in Detection",
            "chunk_results": [],
            "error": str(e),
        }


# login
@app.post("/auth/login", response_model=schemas.Token)
async def login(credentials: schemas.UserCredentials):
    if credentials.username != DEFAULT_USER or credentials.password != DEFAULT_PASS:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token_expires = timedelta(hours=24)
    access_token = jwt.encode(
        {
            "sub": credentials.username,
            "exp": datetime.now(timezone.utc) + access_token_expires,
        },
        JWT_SECRET,
        algorithm="HS256",
    )

    return {"access_token": access_token, "token_type": "bearer"}


# create new inmate with reference audio
@app.post("/inmates", response_model=schemas.InmateResponse)
async def create_inmate(
    name: str = Form(...),
    inmate_code: str = Form(...),
    reference_audio: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    print(f"Received: name={name}, inmate_code={inmate_code}")
    print(f"File: {reference_audio.filename if reference_audio else 'None'}")
    print(
        f"Content-Type: {reference_audio.content_type if reference_audio else 'None'}"
    )

    existing = crud.get_inmate_by_code(db, inmate_code)
    if existing:
        raise HTTPException(status_code=400, detail="Inmate code already exists")

    if not reference_audio.content_type or not reference_audio.content_type.startswith(
        "audio/"
    ):
        if not reference_audio.filename or not any(
            reference_audio.filename.lower().endswith(ext)
            for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
        ):
            raise HTTPException(status_code=400, detail="Invalid audio file")

    # Save audio file
    file_id = str(uuid.uuid4())
    file_extension = reference_audio.filename.split(".")[-1]
    audio_filename = f"{file_id}_{reference_audio.filename}"
    audio_path = f"data/uploads/{audio_filename}"

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(reference_audio.file, buffer)

    # create inmate
    inmate = crud.create_inmate(db, name=name, inmate_code=inmate_code)

    # create embedding
    embedding = ml_stubs.get_embedding(audio_path)

    # create voiceprint
    voiceprint = crud.create_voiceprint(
        db, inmate_id=inmate.id, embedding=embedding, sample_audio_path=audio_path
    )

    return schemas.InmateResponse(
        id=inmate.id,
        name=inmate.name,
        inmate_code=inmate.inmate_code,
        created_at=inmate.created_at,
        voiceprint_id=voiceprint.id,
    )


# get all inmates
@app.get("/inmates", response_model=List[schemas.InmateBase])
async def get_inmates(
    current_user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    return crud.get_inmates(db)


# get specific inmate
@app.get("/inmates/{inmate_id}", response_model=schemas.InmateBase)
async def get_inmate(
    inmate_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    inmate = crud.get_inmate(db, inmate_id)
    if not inmate:
        raise HTTPException(status_code=404, detail="Inmate not found")
    return inmate


# upload and analyse audio file
@app.post("/upload", response_model=schemas.AnalysisResponse)
async def upload_audio(
    audio: UploadFile = File(...),
    inmate_code: Optional[str] = Form(None),
    claimed_caller: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    provided_by: Optional[str] = Form(None),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Validate audio file
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        if not audio.filename or not any(
            audio.filename.lower().endswith(ext)
            for ext in [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
        ):
            raise HTTPException(status_code=400, detail="Invalid audio file")

    # Save audio file
    file_id = str(uuid.uuid4())
    file_extension = audio.filename.split(".")[-1] if "." in audio.filename else "wav"
    audio_filename = f"{file_id}_{audio.filename}"
    audio_path = f"data/uploads/{audio_filename}"

    file_bytes = await audio.read()

    # Write the bytes directly to file
    with open(audio_path, "wb") as buffer:
        buffer.write(file_bytes)

    # Compute hashes
    sha256_hash = utils.compute_sha256(audio_path)
    md5_hash = utils.compute_md5(audio_path)

    # Get audio metadata
    audio_metadata = utils.get_audio_metadata(audio_path)  # Changed variable name

    # Debug: Verify it's a dictionary
    print(f"Audio metadata type: {type(audio_metadata)}")
    print(f"Audio metadata: {audio_metadata}")

    # Ensure it's JSON serializable
    import json

    try:
        json.dumps(audio_metadata)
        print("✅ Audio metadata is JSON serializable")
    except Exception as e:
        print(f"❌ Audio metadata is not JSON serializable: {e}")
        audio_metadata = {
            "file_name": os.path.basename(audio_path),
            "error": "Failed to serialize metadata",
        }

    # Continue with ML analysis after metadata extraction
    try:
        # Run both ML analyses in parallel
        loop = asyncio.get_event_loop()

        # 1. Advanced synthetic detection (our ML module)
        synthetic_result = await loop.run_in_executor(
            ml_executor, ml_stubs.detect_synthetic, audio_path
        )
        is_synthetic_ml, synthetic_confidence_ml = synthetic_result

        # 2. Deepfake detection (transformer models)
        file_content = await audio.read()
        await audio.seek(0)  # Reset file pointer
        deepfake_result = await run_deepfake_detection(file_content, audio.filename)

        # 3. Generate spectrogram
        spectrogram_path = await loop.run_in_executor(
            ml_executor, ml_stubs.generate_spectrogram, audio_path
        )

        # 4. Segment-wise analysis
        segments_result = await loop.run_in_executor(
            ml_executor, ml_stubs.detect_synthetic_segments, audio_path
        )

        # Combine results - if either system detects synthetic, flag as synthetic
        is_synthetic_combined = is_synthetic_ml or deepfake_result["is_synthetic"]
        synthetic_confidence_combined = max(
            synthetic_confidence_ml, deepfake_result["confidence"]
        )

    except Exception as e:
        print(f"ML analysis failed: {e}")
        # Fallback to original stub behavior
        is_synthetic_combined, synthetic_confidence_combined = (
            ml_stubs.detect_synthetic(audio_path)
        )
        spectrogram_path = ml_stubs.generate_spectrogram(audio_path)
        synthetic_details_ml = {
            "model": "Fallback-Stub",
            "overall_score": synthetic_confidence_combined,
            "error": str(e),
        }
        deepfake_result = {
            "verdict": "🟠 Error in Detection",
            "confidence": 0.5,
            "error": str(e),
        }
        segments_result = []

    # Voice matching (existing logic)
    matched_inmate = None
    speaker_match = False
    speaker_match_confidence = 0.0

    # Get embedding for uploaded audio
    uploaded_embedding = ml_stubs.get_embedding(audio_path)

    # If inmate_code provided, compare with specific inmate
    if inmate_code:
        inmate = crud.get_inmate_by_code(db, inmate_code)
        if inmate:
            voiceprint = crud.get_voiceprint_by_inmate(db, inmate.id)
            if voiceprint:
                similarity = ml_stubs.compare_embeddings(
                    uploaded_embedding, voiceprint.embedding
                )
                if similarity >= 0.6:  # Demo threshold (lowered for better matching)
                    speaker_match = True
                    speaker_match_confidence = similarity
                    matched_inmate = inmate
    else:
        # Search all inmates for best match
        best_match = crud.find_best_voice_match(db, uploaded_embedding)
        if best_match and best_match["similarity"] >= 0.6:
            speaker_match = True
            speaker_match_confidence = best_match["similarity"]
            matched_inmate = best_match["inmate"]

    # Generate report code
    report_code = utils.generate_report_code()

    # Enhanced report creation with both ML analyses
    report = crud.create_report(
        db=db,
        report_code=report_code,
        inmate_id=matched_inmate.id if matched_inmate else None,
        audio_path=audio_path,
        spectrogram_path=spectrogram_path,
        is_synthetic=is_synthetic_combined,
        synthetic_confidence=synthetic_confidence_combined,
        speaker_match=speaker_match,
        speaker_match_confidence=speaker_match_confidence,
        sha256=sha256_hash,
        md5=md5_hash,
        call_metadata=audio_metadata,
        claimed_caller=claimed_caller,
        context=context,
        provided_by=provided_by,
    )

    # Generate reports with available data
    json_path, pdf_path = utils.generate_reports(
        report=report,
        metadata=audio_metadata,
        spectrogram_path=spectrogram_path,
        matched_inmate=matched_inmate,
        hmac_key=HMAC_KEY,
    )

    # Update report with file paths
    crud.update_report_paths(db, report.id, json_path, pdf_path)

    return schemas.AnalysisResponse(
        report_id=report.id,
        report_code=report_code,
        is_synthetic=is_synthetic_combined,
        synthetic_confidence=synthetic_confidence_combined,
        speaker_match=speaker_match,
        speaker_match_confidence=speaker_match_confidence,
        matched_inmate_code=matched_inmate.inmate_code if matched_inmate else None,
        json_report_url=f"/reports/{report.id}",
        pdf_report_url=f"/reports/{report.id}/pdf",
        # Enhanced ML analysis summary
        ml_analysis_summary={
            "biometric_model": synthetic_details_ml.get("model", "Unknown"),
            "deepfake_models": deepfake_result.get("model_info", {}).get(
                "models_used", []
            ),
            "segments_analyzed": len(segments_result),
            "confidence_level": synthetic_details_ml.get("confidence_level", "medium"),
            "deepfake_verdict": deepfake_result.get("verdict", "Unknown"),
            "device_used": str(device),
        },
    )


# deepfake detection endpoint
@app.post("/api/v1/predict/")
async def predict_deepfake_only(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        os.makedirs("data/uploads", exist_ok=True)
        audio_path = os.path.join("data/uploads", file.filename)

        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        # Extract metadata
        audio_metadata = utils.get_audio_metadata(audio_path)

        # Run deepfake detection
        deepfake_result = await run_deepfake_detection(file_bytes, file.filename)

        report = {
            "report_id": f"DFA-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "date_of_analysis": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analyzed_by": "Automated AI Tool",
            "tool_used": model_names,
            "audio_metadata": audio_metadata,
            "deepfake_analysis": deepfake_result,
            "final_verdict": deepfake_result["verdict"],
        }

        return JSONResponse(report)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# get all reports
@app.get("/reports", response_model=List[schemas.ReportSummary])
async def get_reports(
    current_user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    return crud.get_reports(db)


# health check
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "deepfake_models_loaded": len(deepfake_models),
        "biometric_ml_available": True,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }


# get information about loaded ml models
@app.get("/ml/info")
async def get_ml_info(current_user: str = Depends(get_current_user)):
    try:
        # Get biometric ML info
        biometric_info = ml_stubs.get_model_info()

        # Get deepfake ML info
        deepfake_info = {
            "models_loaded": len(deepfake_models),
            "model_names": model_names,
            "device": str(device),
            "sample_rate": SAMPLE_RATE,
        }

        return {
            "biometric_recognition": biometric_info,
            "deepfake_detection": deepfake_info,
            "combined_system": {
                "status": "operational",
                "features": [
                    "Voice biometric identification",
                    "Inmate database matching",
                    "Advanced synthetic detection (AASIST)",
                    "Transformer-based deepfake detection",
                    "Segment-wise analysis",
                    "Forensic report generation",
                ],
            },
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": "partial_failure",
            "recommendations": ["Check ML module installation and dependencies"],
        }


# get all reports
@app.get("/reports", response_model=List[schemas.ReportSummary])
async def get_reports(
    current_user: str = Depends(get_current_user), db: Session = Depends(get_db)
):
    return crud.get_reports(db)


# get json report
@app.get("/reports/{report_id}")
async def get_report_json(
    report_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = crud.get_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if not report.report_json_path or not os.path.exists(report.report_json_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    with open(report.report_json_path, "r") as f:
        return json.load(f)


# download pdf report
@app.get("/reports/{report_id}/pdf")
async def download_report_pdf(
    report_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = crud.get_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if not report.report_pdf_path or not os.path.exists(report.report_pdf_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        report.report_pdf_path,
        media_type="application/pdf",
        filename=f"forensic_report_{report.report_code}.pdf",
    )


# verify report signature
@app.get("/reports/{report_id}/verify")
async def verify_report(
    report_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    report = crud.get_report(db, report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    if not report.report_json_path or not os.path.exists(report.report_json_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    with open(report.report_json_path, "r") as f:
        report_data = json.load(f)

    is_valid = utils.verify_signature(report_data, HMAC_KEY)

    return {
        "report_id": report_id,
        "report_code": report.report_code,
        "is_valid": is_valid,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "message": (
            "Signature is valid" if is_valid else "Signature verification failed"
        ),
    }


@app.get("/")
async def root():
    return {
        "message": "Voice Biometric Recognition API",
        "version": "1.0.0",
        "endpoints": [
            "POST /auth/login - Login",
            "POST /inmates - Create inmate",
            "GET /inmates - List inmates",
            "POST /upload - Analyze audio",
            "GET /reports - List reports",
            "GET /reports/{id} - Get JSON report",
            "GET /reports/{id}/pdf - Download PDF report",
            "GET /reports/{id}/verify - Verify report signature",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Voice Biometric Recognition & Deepfake Detection System")
    print("=" * 60)
    print("🌐 Web UI: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("🤖 ML Status: http://localhost:8000/ml/info")
    print("🔍 Deepfake Test: http://localhost:8000/api/v1/predict/")
    print("=" * 60)

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
