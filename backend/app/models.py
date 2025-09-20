from sqlalchemy import Column, String, Boolean, Float, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.database import Base


class Inmate(Base):
    __tablename__ = "inmates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, nullable=False)
    inmate_code = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())

    # relationship
    voiceprints = relationship("Voiceprint", back_populates="inmate")
    reports = relationship("Report", back_populates="inmate")


class Voiceprint(Base):
    __tablename__ = "voiceprints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    inmate_id = Column(UUID(as_uuid=True), ForeignKey("inmates.id"), nullable=False)
    embedding = Column(JSON, nullable=False)
    sample_audio_path = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    inmate = relationship("Inmate", back_populates="voiceprints")


class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    report_code = Column(String, unique=True, nullable=False, index=True)
    inmate_id = Column(UUID(as_uuid=True), ForeignKey("inmates.id"), nullable=True)

    audio_path = Column(Text, nullable=False)
    spectrogram_path = Column(Text, nullable=True)
    report_json_path = Column(Text, nullable=True)
    report_pdf_path = Column(Text, nullable=True)

    is_synthetic = Column(Boolean, nullable=False)
    synthetic_confidence = Column(Float, nullable=False)
    speaker_match = Column(Boolean, nullable=False, default=False)
    speaker_match_confidence = Column(Float, nullable=False, default=0.0)

    sha256 = Column(String, nullable=False)
    md5 = Column(String, nullable=True)
    call_metadata = Column(JSON, nullable=True)

    claimed_caller = Column(String, nullable=True)
    context = Column(Text, nullable=True)
    provided_by = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    inmate = relationship("Inmate", back_populates="reports")
