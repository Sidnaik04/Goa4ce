from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid


class UserCredentials(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class InmateBase(BaseModel):
    id: uuid.UUID
    name: str
    inmate_code: str
    created_at: datetime

    class Config:
        orm_mode = True


class InmateCreate(BaseModel):
    name: str
    inmate_code: str


class InmateResponse(InmateBase):
    voiceprint_id: uuid.UUID


class VoiceprintBase(BaseModel):
    id: uuid.UUID
    inmate_id: uuid.UUID
    sample_audio_path: str
    created_at: datetime

    class Config:
        orm_mode = True


class ReportSummary(BaseModel):
    id: uuid.UUID
    report_code: str
    inmate_id: Optional[uuid.UUID]
    is_synthetic: bool
    synthetic_confidence: float
    speaker_match: bool
    speaker_match_confidence: float
    created_at: datetime

    class Config:
        orm_mode = True


class AnalysisResponse(BaseModel):
    report_id: uuid.UUID
    report_code: str
    is_synthetic: bool
    synthetic_confidence: float
    speaker_match: bool
    speaker_match_confidence: float
    matched_inmate_code: Optional[str]
    json_report_url: str
    pdf_report_url: str


class AudioMetadata(BaseModel):
    duration: float
    sampling_rate: int
    file_format: str
    file_size: int
    creation_date: datetime
