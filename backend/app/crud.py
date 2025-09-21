from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import uuid
from app import models, ml_stubs
from app.utils import make_json_serializable


# create a new inmate
def create_inmate(db: Session, name: str, inmate_code: str) -> models.Inmate:
    db_inmate = models.Inmate(name=name, inmate_code=inmate_code)
    db.add(db_inmate)
    db.commit()
    db.refresh(db_inmate)
    return db_inmate


# get inmate by id
def get_inmate(db: Session, inmate_id: str) -> Optional[models.Inmate]:
    try:
        uuid_id = uuid.UUID(inmate_id)
        return db.query(models.Inmate).filter(models.Inmate.id == uuid_id).first()
    except ValueError:
        return None


# get inmate by code
def get_inmate_by_code(db: Session, inmate_code: str) -> Optional[models.Inmate]:
    return (
        db.query(models.Inmate).filter(models.Inmate.inmate_code == inmate_code).first()
    )


# get all inmates
def get_inmates(db: Session, skip: int = 0, limit: int = 100) -> List[models.Inmate]:
    return db.query(models.Inmate).offset(skip).limit(limit).all()


# create a new voiceprint
def create_voiceprint(
    db: Session, inmate_id: uuid.UUID, embedding: List[float], sample_audio_path: str
) -> models.Voiceprint:
    db_voiceprint = models.Voiceprint(
        inmate_id=inmate_id, embedding=embedding, sample_audio_path=sample_audio_path
    )
    db.add(db_voiceprint)
    db.commit()
    db.refresh(db_voiceprint)
    return db_voiceprint


# get voiceprint by inmate id
def get_voiceprint_by_inmate(
    db: Session, inmate_id: uuid.UUID
) -> Optional[models.Voiceprint]:
    return (
        db.query(models.Voiceprint)
        .filter(models.Voiceprint.inmate_id == inmate_id)
        .first()
    )


# Find best matching voice from all stored voiceprints
def find_best_voice_match(
    db: Session, uploaded_embedding: List[float]
) -> Optional[Dict[str, Any]]:
    voiceprints = db.query(models.Voiceprint).all()

    best_match = None
    best_similarity = 0.0

    for voiceprint in voiceprints:
        similarity = ml_stubs.compare_embeddings(
            uploaded_embedding, voiceprint.embedding
        )

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = voiceprint

    if best_match and best_similarity >= 0.75:
        inmate = (
            db.query(models.Inmate)
            .filter(models.Inmate.id == best_match.inmate_id)
            .first()
        )
        return {
            "similarity": best_similarity,
            "voiceprint": best_match,
            "inmate": inmate,
        }

    return None


# create analysis report
def create_report(
    db: Session,
    report_code: str,
    inmate_id: Optional[uuid.UUID],
    audio_path: str,
    spectrogram_path: str,
    is_synthetic: bool,
    synthetic_confidence: float,
    speaker_match: bool,
    speaker_match_confidence: float,
    sha256: str,
    md5: Optional[str] = None,
    call_metadata: Optional[Dict] = None,
    claimed_caller: Optional[str] = None,
    context: Optional[str] = None,
    provided_by: Optional[str] = None,
) -> models.Report:

    safe_call_metadata = (
        make_json_serializable(call_metadata) if call_metadata is not None else None
    )

    db_report = models.Report(
        report_code=report_code,
        inmate_id=inmate_id,
        audio_path=audio_path,
        spectrogram_path=spectrogram_path,
        is_synthetic=is_synthetic,
        synthetic_confidence=synthetic_confidence,
        speaker_match=speaker_match,
        speaker_match_confidence=speaker_match_confidence,
        sha256=sha256,
        md5=md5,
        call_metadata=safe_call_metadata,
        claimed_caller=claimed_caller,
        context=context,
        provided_by=provided_by,
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report


# update report file paths
def update_report_paths(
    db: Session, report_id: uuid.UUID, json_path: str, pdf_path: str
):
    db_report = db.query(models.Report).filter(models.Report.id == report_id).first()
    if db_report:
        db_report.report_json_path = json_path
        db_report.report_pdf_path = pdf_path
        db.commit()


# get report by id
def get_report(db: Session, report_id: str) -> Optional[models.Report]:
    try:
        uuid_id = uuid.UUID(report_id)
        return db.query(models.Report).filter(models.Report.id == uuid_id).first()
    except ValueError:
        return None


# get all reports
def get_reports(db: Session, skip: int = 0, limit: int = 100) -> List[models.Report]:
    return db.query(models.Report).offset(skip).limit(limit).all()


# get report by code
def get_report_by_code(db: Session, report_code: str) -> Optional[models.Report]:
    return (
        db.query(models.Report).filter(models.Report.report_code == report_code).first()
    )
