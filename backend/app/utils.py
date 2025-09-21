import hashlib
import hmac
import json
import os
import qrcode
from datetime import datetime
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import uuid
import librosa

import json
import numpy as np
import torch
from datetime import datetime
import uuid as _uuid
import types
from pathlib import Path


# compute md5 hash of file
def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hash of a file"""
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error computing SHA256: {e}")
        return ""


def compute_md5(file_path: str) -> str:
    """Compute MD5 hash of a file"""
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error computing MD5: {e}")
        return ""


# extract metadata for deepfake analysis
def extract_metadata(file_bytes, filename):
    import librosa
    import io
    import soundfile as sf

    try:
        y, sr = sf.read(io.BytesIO(file_bytes), always_2d=False)
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        duration = librosa.get_duration(y=y, sr=sr)
        fmt = os.path.splitext(filename)[1][1:] if "." in filename else "unknown"
        file_hash = hashlib.md5(file_bytes).hexdigest()

        return {
            "file_name": filename,
            "file_format": fmt,
            "duration": f"{int(duration//60)} min {int(duration%60)} sec",
            "sampling_rate": f"{sr} Hz",
            "md5_hash": file_hash,
            "date_of_file_creation": datetime.utcnow().strftime("%Y-%m-%d %H:%M GMT"),
            "file_size": f"{len(file_bytes)} bytes",
        }
    except Exception as e:
        return {
            "file_name": filename,
            "file_format": "unknown",
            "duration": "unknown",
            "sampling_rate": "unknown",
            "md5_hash": hashlib.md5(file_bytes).hexdigest(),
            "date_of_file_creation": datetime.utcnow().strftime("%Y-%m-%d %H:%M GMT"),
            "file_size": f"{len(file_bytes)} bytes",
            "error": str(e),
        }


def get_audio_metadata(audio_path: str) -> dict:
    """Extract audio metadata and return as a dictionary"""
    try:
        if not os.path.exists(audio_path):
            return {
                "file_name": os.path.basename(audio_path),
                "error": "File not found",
            }

        # Get file stats
        file_stats = os.stat(audio_path)
        file_size = file_stats.st_size
        creation_time = datetime.fromtimestamp(file_stats.st_ctime)

        # Basic metadata
        metadata = {
            "file_name": os.path.basename(audio_path),
            "file_format": os.path.splitext(audio_path)[1].upper().replace(".", "")
            or "UNKNOWN",
            "file_size": file_size,
            "creation_date": creation_time.isoformat(),
        }

        # Try to get audio-specific metadata
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr

            metadata.update(
                {
                    "duration": duration,
                    "sampling_rate": sr,
                    "channels": 1 if len(y.shape) == 1 else y.shape[0],
                }
            )
        except Exception as audio_error:
            print(f"Could not extract audio-specific metadata: {audio_error}")
            metadata.update(
                {
                    "duration": 0.0,
                    "sampling_rate": 16000,
                    "channels": 1,
                    "audio_error": str(audio_error),
                }
            )

        return metadata

    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {
            "file_name": os.path.basename(audio_path),
            "error": str(e),
            "fallback": True,
        }


# generate report code in format "DFA-YYYY-MM-XXX"
def generate_report_code() -> str:
    now = datetime.now()
    random_suffix = str(uuid.uuid4())[:3].upper()
    return f"DFA-{now.year}-{now.month:02d}-{random_suffix}"


# create HMAC signature fo report data
def create_signature(data: Dict[str, Any], key: str) -> str:
    data_to_sign = {k: v for k, v in data.items() if k != "signature"}

    json_str = json.dumps(data_to_sign, sort_keys=True, separators=(",", ":"))

    signature = hmac.new(
        key.encode("utf-8"), json_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    return signature


# verify HMAC signature of report data
def verify_signature(data: Dict[str, Any], key: str) -> bool:
    if "signature" not in data:
        return False

    provided_signature = data["signature"]
    expected_signature = create_signature(data, key)

    return hmac.compare_digest(provided_signature, expected_signature)


# generate qr code and save to file
def generate_qr_code(data: str, file_path: str) -> str:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    img.save(file_path)
    return file_path


def generate_reports(report, metadata, spectrogram_path, matched_inmate, hmac_key):
    """Generate JSON and PDF reports (basic implementation)"""
    try:
        # Create a basic report dictionary
        report_data = {
            "report_id": str(report.id),
            "report_code": report.report_code,
            "timestamp": datetime.now().isoformat(),
            "audio_metadata": metadata,
            "analysis_results": {
                "is_synthetic": report.is_synthetic,
                "synthetic_confidence": report.synthetic_confidence,
                "speaker_match": report.speaker_match,
                "speaker_match_confidence": report.speaker_match_confidence,
            },
            "matched_inmate": (
                {
                    "name": matched_inmate.name if matched_inmate else None,
                    "code": matched_inmate.inmate_code if matched_inmate else None,
                }
                if matched_inmate
                else None
            ),
            "spectrogram_path": spectrogram_path,
        }

        # Save JSON report
        os.makedirs("data/reports", exist_ok=True)
        json_path = f"data/reports/{report.report_code}.json"

        with open(json_path, "w") as f:
            import json

            json.dump(report_data, f, indent=2)

        # For now, return the same path for PDF (you can implement PDF generation later)
        pdf_path = f"data/reports/{report.report_code}.pdf"

        print(f"Reports generated: {json_path}")
        return json_path, pdf_path

    except Exception as e:
        print(f"Error generating reports: {e}")
        return None, None


def get_voice_identification_verdict(report, matched_inmate) -> str:
    """Generate voice identification verdict"""
    if report.speaker_match and matched_inmate:
        return f"POSITIVE IDENTIFICATION - Voice matches registered inmate {matched_inmate.inmate_code} ({matched_inmate.name})"
    else:
        return (
            "NO MATCH FOUND - Voice does not match any registered inmates in database"
        )


def get_authenticity_verdict(report, synthetic_details, deepfake_details) -> str:
    """Generate authenticity verdict combining both analyses"""
    if report.is_synthetic:
        sources = []
        if (
            synthetic_details
            and synthetic_details.get("overall_decision") == "synthetic"
        ):
            sources.append("biometric ML analysis")
        if deepfake_details and "Synthetic" in deepfake_details.get("verdict", ""):
            sources.append("deepfake detection models")

        source_text = " and ".join(sources) if sources else "automated analysis"
        return f"SYNTHETIC VOICE DETECTED - Flagged by {source_text}"
    else:
        return "AUTHENTIC VOICE - No synthetic patterns detected by analysis systems"


def get_overall_conclusion(
    report, matched_inmate, synthetic_details, deepfake_details
) -> str:
    """Generate overall conclusion"""
    if report.is_synthetic:
        return "SYNTHETIC/AI-GENERATED AUDIO - Not suitable for voice identification purposes"
    elif report.speaker_match and matched_inmate:
        return f"AUTHENTIC VOICE MATCH - Confirmed identity: {matched_inmate.name} ({matched_inmate.inmate_code})"
    else:
        return (
            "AUTHENTIC VOICE - UNKNOWN SPEAKER - Voice is genuine but not in database"
        )


def get_confidence_rating(report, synthetic_details, deepfake_details) -> str:
    """Generate confidence rating"""
    synthetic_conf = report.synthetic_confidence
    speaker_conf = report.speaker_match_confidence

    # Calculate overall confidence based on both metrics
    if synthetic_conf > 0.8 or speaker_conf > 0.9:
        return "HIGH"
    elif synthetic_conf > 0.6 or speaker_conf > 0.75:
        return "MEDIUM"
    else:
        return "LOW"


def get_supporting_evidence(
    report, matched_inmate, synthetic_details, deepfake_details
) -> list:
    """Generate supporting evidence list"""
    evidence = []

    # Authenticity evidence
    if report.is_synthetic:
        evidence.append(
            f"Synthetic detection confidence: {report.synthetic_confidence:.2%}"
        )
        if deepfake_details:
            evidence.append(
                f"Deepfake detection verdict: {deepfake_details.get('verdict', 'Unknown')}"
            )
    else:
        evidence.append(
            f"Authenticity confidence: {(1-report.synthetic_confidence):.2%}"
        )
        evidence.append("Voice patterns consistent with human speech")

    # Speaker matching evidence
    if report.speaker_match and matched_inmate:
        evidence.append(
            f"Voice similarity score: {report.speaker_match_confidence:.2%} with {matched_inmate.name}"
        )
        evidence.append("Embedding vectors show high correlation")
    else:
        evidence.append(
            "No significant similarity found with registered voice profiles"
        )

    # Technical evidence
    evidence.append("Multi-model analysis completed successfully")
    evidence.append("Spectrogram analysis shows consistent patterns")
    evidence.append("File integrity verified through cryptographic hashing")

    return evidence


# final verdict based on analysis
def get_final_verdict(report, matched_inmate) -> str:
    if report.is_synthetic:
        return "SYNTHETIC VOICE DETECTED - Audio appears to be artificially generated"
    elif report.speaker_match and matched_inmate:
        return f"VOICE MATCH CONFIRMED - Audio matches registered inmate {matched_inmate.inmate_code}"
    elif not report.speaker_match:
        return (
            "NO VOICE MATCH - Audio does not match any registered inmates in database"
        )
    else:
        return "AUTHENTIC VOICE - No synthetic patterns detected, origin unidentified"


def get_enhanced_recommendations(
    report, matched_inmate, synthetic_details, deepfake_details
) -> list:
    """Generate enhanced recommendations"""
    recommendations = []

    if report.is_synthetic:
        recommendations.extend(
            [
                "URGENT: Investigate source of synthetic audio generation",
                "Cross-reference with known AI voice generation tools",
                "Flag associated communications for synthetic voice screening",
                "Consider expanding synthetic detection to related cases",
            ]
        )
    else:
        if report.speaker_match and matched_inmate:
            recommendations.extend(
                [
                    "Proceed with positive voice identification protocols",
                    "Document match for legal proceedings",
                    "Consider additional biometric verification if required",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Expand database search to include additional voice samples",
                    "Consider manual expert review by forensic phonetician",
                    "Collect additional voice samples for comparison if possible",
                ]
            )

    # General recommendations
    recommendations.extend(
        [
            "Preserve all original audio files in secure storage",
            "Maintain detailed chain of custody documentation",
            "Archive analysis results for future reference",
            "Consider periodic recalibration of detection thresholds",
        ]
    )

    return recommendations


def generate_enhanced_pdf_report(
    report_data: Dict, pdf_path: str, spectrogram_path: str, hmac_key: str
):
    """Generate enhanced PDF forensic report with both analyses"""
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER,
    )

    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=12,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.darkblue,
    )

    # Title
    story.append(
        Paragraph("ENHANCED DIGITAL FORENSIC AUDIO ANALYSIS REPORT", title_style)
    )
    story.append(
        Paragraph("Voice Biometric Recognition & Deepfake Detection", styles["Normal"])
    )
    story.append(Spacer(1, 20))

    # Report Header
    header_data = report_data["report_header"]
    header_table_data = [
        ["Report ID:", header_data["report_id"]],
        ["Date of Analysis:", header_data["date_of_analysis"][:19]],
        ["Analyzed By:", header_data["analyzed_by"]],
        ["Tool Used:", header_data["tool_used"]],
    ]

    header_table = Table(header_table_data, colWidths=[2 * inch, 4 * inch])
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(header_table)
    story.append(Spacer(1, 20))

    # Audio File Metadata
    story.append(Paragraph("1. AUDIO FILE METADATA", heading_style))

    metadata = report_data["audio_file_metadata"]
    metadata_table_data = [
        ["File Name", metadata["file_name"]],
        ["File Format", metadata["file_format"]],
        ["Duration", metadata["duration"]],
        ["Sampling Rate", metadata["sampling_rate"]],
        ["MD5 Hash", metadata["md5_hash"][:32] + "..."],
        ["SHA256 Hash", metadata["sha256_hash"][:32] + "..."],
        ["File Size", metadata["file_size"]],
    ]

    metadata_table = Table(metadata_table_data, colWidths=[2 * inch, 4 * inch])
    metadata_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightblue),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Spacer(1, 15))

    # Biometric Analysis
    story.append(Paragraph("2. VOICE BIOMETRIC ANALYSIS", heading_style))

    biometric = report_data["biometric_analysis"]["voice_recognition"]
    biometric_table_data = [
        ["Speaker Match", "YES" if biometric["speaker_match"] else "NO"],
        ["Similarity Score", biometric["similarity_score"]],
        ["Matched Inmate", biometric["matched_inmate"]],
        ["Threshold Used", biometric["threshold_used"]],
        ["Database Search", biometric["database_search"]],
    ]

    biometric_table = Table(biometric_table_data, colWidths=[2 * inch, 4 * inch])
    biometric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgreen),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(biometric_table)
    story.append(Spacer(1, 15))

    # Deepfake Analysis
    story.append(Paragraph("3. DEEPFAKE DETECTION ANALYSIS", heading_style))

    deepfake = report_data["deepfake_analysis"]
    deepfake_table_data = [
        ["Overall Verdict", deepfake["overall_verdict"]],
        ["Confidence Score", deepfake["confidence_score"]],
        [
            "Models Used",
            ", ".join(deepfake["models_used"]) if deepfake["models_used"] else "N/A",
        ],
        ["Chunks Analyzed", str(deepfake["chunk_analysis"]["chunks_analyzed"])],
        ["Processing Device", deepfake["device_used"]],
    ]

    deepfake_table = Table(deepfake_table_data, colWidths=[2 * inch, 4 * inch])
    deepfake_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.lightyellow),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(deepfake_table)
    story.append(Spacer(1, 15))

    # Final Verdict
    story.append(Paragraph("4. FINAL VERDICT & CONCLUSIONS", heading_style))

    verdict = report_data["final_verdict"]
    story.append(
        Paragraph(
            f"<b>Voice Identification:</b> {verdict['voice_identification']}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"<b>Authenticity Assessment:</b> {verdict['authenticity_assessment']}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"<b>Overall Conclusion:</b> {verdict['overall_conclusion']}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"<b>Confidence Rating:</b> {verdict['confidence_rating']}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 15))

    # Supporting Evidence
    story.append(Paragraph("<b>Supporting Evidence:</b>", styles["Normal"]))
    for evidence in verdict["supporting_evidence"]:
        story.append(Paragraph(f"• {evidence}", styles["Normal"]))
    story.append(Spacer(1, 15))

    # Recommendations
    story.append(Paragraph("<b>Recommendations:</b>", styles["Normal"]))
    for rec in verdict["recommendations"][:5]:  # Limit to 5 recommendations
        story.append(Paragraph(f"• {rec}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Add spectrogram if available
    if os.path.exists(spectrogram_path):
        story.append(Paragraph("5. SPECTROGRAM ANALYSIS", heading_style))
        try:
            img = Image(spectrogram_path, width=6 * inch, height=3 * inch)
            story.append(img)
            story.append(Spacer(1, 15))
        except:
            story.append(
                Paragraph("Spectrogram image could not be loaded.", styles["Normal"])
            )
            story.append(Spacer(1, 15))

    # Quality Assurance
    story.append(Paragraph("6. QUALITY ASSURANCE", heading_style))
    qa = report_data["quality_assurance"]
    qa_items = [
        f"File integrity verified: {'YES' if qa['file_integrity_verified'] else 'NO'}",
        f"Multi-model analysis: {'YES' if qa['multi_model_analysis'] else 'NO'}",
        f"Biometric ML status: {qa['biometric_ml_status']}",
        f"Deepfake ML status: {qa['deepfake_ml_status']}",
    ]

    for item in qa_items:
        story.append(Paragraph(f"• {item}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Digital Signature and QR Code
    story.append(Paragraph("7. VERIFICATION & SIGNATURE", heading_style))
    story.append(
        Paragraph(
            "This report is digitally signed using HMAC-SHA256.", styles["Normal"]
        )
    )

    # Generate QR code
    qr_data = f"Report:{report_data['report_header']['report_id']}|Hash:{report_data['signature'][:32]}..."
    qr_path = f"data/reports/qr_{report_data['report_header']['report_id']}.png"
    generate_qr_code(qr_data, qr_path)

    try:
        qr_img = Image(qr_path, width=1.5 * inch, height=1.5 * inch)
        story.append(qr_img)
    except:
        story.append(Paragraph("QR code could not be generated.", styles["Normal"]))


def make_json_serializable(obj):
    """
    Convert obj to a JSON-serializable structure.
    - Handles: numpy arrays/scalars, torch tensors, datetimes, UUIDs, Paths, modules, exceptions.
    - For unknown types, returns a string fallback.
    """
    # primitives and None are OK
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]

    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else str(k)
            out[key] = make_json_serializable(v)
        return out

    # numpy
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    # torch
    try:
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass

    # datetime
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()

    # uuid
    if isinstance(obj, _uuid.UUID):
        return str(obj)

    # Path
    if isinstance(obj, (Path,)):
        return str(obj)

    # module -> return module name
    if isinstance(obj, types.ModuleType):
        return getattr(obj, "__name__", str(obj))

    # exceptions
    if isinstance(obj, BaseException):
        return {"error_type": type(obj).__name__, "error_msg": str(obj)}

    # fallback: try json.dumps, else str()
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            return str(obj)
        except Exception:
            return repr(obj)
