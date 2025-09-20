from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
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
    print(f"Content-Type: {reference_audio.content_type if reference_audio else 'None'}")
    
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

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # Compute hashes
    sha256_hash = utils.compute_sha256(audio_path)
    md5_hash = utils.compute_mdf5(audio_path)

    # Get audio metadata
    metadata = utils.get_audio_metadata(audio_path)

    # Generate spectrogram
    spectrogram_path = ml_stubs.generate_spectrogram(audio_path)

    # Detect synthetic
    is_synthetic, synthetic_confidence = ml_stubs.detect_synthetic(audio_path)

    # Voice matching
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
                if similarity >= 0.75:  # Demo threshold
                    speaker_match = True
                    speaker_match_confidence = similarity
                    matched_inmate = inmate
    else:
        # Search all inmates for best match
        best_match = crud.find_best_voice_match(db, uploaded_embedding)
        if best_match and best_match["similarity"] >= 0.75:
            speaker_match = True
            speaker_match_confidence = best_match["similarity"]
            matched_inmate = best_match["inmate"]

    # generate report code
    report_code = utils.generate_report_code()

    # Create report record
    report = crud.create_report(
        db=db,
        report_code=report_code,
        inmate_id=matched_inmate.id if matched_inmate else None,
        audio_path=audio_path,
        spectrogram_path=spectrogram_path,
        is_synthetic=is_synthetic,
        synthetic_confidence=synthetic_confidence,
        speaker_match=speaker_match,
        speaker_match_confidence=speaker_match_confidence,
        sha256=sha256_hash,
        md5=md5_hash,
        metadata=metadata,
        claimed_caller=claimed_caller,
        context=context,
        provided_by=provided_by,
    )

    # Generate JSON and PDF reports
    json_path, pdf_path = utils.generate_reports(
        report=report,
        metadata=metadata,
        spectrogram_path=spectrogram_path,
        matched_inmate=matched_inmate,
        hmac_key=HMAC_KEY,
    )

    # Update report with file paths
    crud.update_report_paths(db, report.id, json_path, pdf_path)

    return schemas.AnalysisResponse(
        report_id=report.id,
        report_code=report_code,
        is_synthetic=is_synthetic,
        synthetic_confidence=synthetic_confidence,
        speaker_match=speaker_match,
        speaker_match_confidence=speaker_match_confidence,
        matched_inmate_code=matched_inmate.inmate_code if matched_inmate else None,
        json_report_url=f"/reports/{report.id}",
        pdf_report_url=f"/reports/{report.id}/pdf",
    )


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
