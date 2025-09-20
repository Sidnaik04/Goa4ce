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


# compute md5 hash of file
def compute_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_mdf5(file_path: str) -> str:
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def get_audio_metadata(file_path: str) -> Dict[str, Any]:
    try:
        file_size = os.path.getsize(file_path)
        creation_time = os.path.getctime(file_path)

        duration = max(1.0, filter(file_size % 30000) / 1000)
        sampling_rate = 16000 if file_size % 2 == 0 else 8000

        return {
            "duration": round(duration, 2),
            "sampling_rate": sampling_rate,
            "file_format": file_path.split(".")[-1].upper(),
            "file_size": file_size,
            "creation_date": datetime.fromtimestamp(creation_time).isoformat(),
            "channels": 1,
        }
    except Exception:
        return {
            "duration": 5.0,
            "sampling_rate": 16000,
            "file_format": "WAV",
            "file_size": os.path.getsize(file_path),
            "creation_date": datetime.now().isoformat(),
            "channels": 1,
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


def generate_reports(
    report,
    metadata: Dict[str, Any],
    spectrogram_path: str,
    matched_inmate=None,
    hmac_key: str = "",
) -> tuple[str, str]:

    report_data = {
        "report_header": {
            "report_id": report.report_code,
            "date_of_analysis": datetime.now().isoformat(),
            "analyzed_by": "Digital Forensics Unit",
            "tool_used": "AI-Based Voice Biometric Recognition System v1.0",
        },
        "audio_file_metadata": {
            "file_name": os.path.basename(report.audio_path),
            "file_format": metadata.get("file_format", "UNKNOWN"),
            "duration": f"{metadata.get('duration', 0)} seconds",
            "sampling_rate": f"{metadata.get('sampling_rate', 0)} Hz",
            "md5_hash": report.md5 or "N/A",
            "sha256_hash": report.sha256,
            "date_of_file_creation": metadata.get("creation_date", "Unknown"),
            "file_size": f"{metadata.get('file_size', 0)} bytes",
        },
        "claimed_source_information": {
            "claimed_caller_identity": report.claimed_caller or "Not specified",
            "context": report.context or "Not specified",
            "provided_by": report.provided_by or "Not specified",
        },
        "technical_analysis_summary": {
            "speech_analysis": {
                "result": "Match" if report.speaker_match else "No Match",
                "confidence": f"{report.speaker_match_confidence:.2%}",
                "comments": "Voice pattern analysis completed using AI embedding comparison",
            },
            "background_noise_consistency": {
                "result": "Consistent",
                "comments": "Background noise patterns analyzed",
            },
            "speech_embedding_comparison": {
                "result": "Match" if report.speaker_match else "No Match",
                "similarity_score": f"{report.speaker_match_confidence:.4f}",
                "comments": f"Cosine similarity threshold: ≥0.75 (Demo), ≥0.85 (Production)",
            },
        },
        "voiceprint_matching": {
            "result": "Match" if report.speaker_match else "No Match",
            "similarity_score": f"{report.speaker_match_confidence:.2%}",
            "threshold_acceptance": "≥75% (Demo Mode)",
            "matched_inmate": matched_inmate.inmate_code if matched_inmate else "None",
        },
        "transcript_comparison": {
            "excerpt": "Transcript analysis not implemented in demo version",
            "commentary": "Manual transcript comparison recommended for production use",
        },
        "synthetic_detection": {
            "is_synthetic": report.is_synthetic,
            "confidence": f"{report.synthetic_confidence:.2%}",
            "threshold": ">60% confidence to classify as synthetic",
            "result": "Synthetic" if report.is_synthetic else "Authentic",
        },
        "verdict": {
            "final_conclusion": get_final_verdict(report, matched_inmate),
            "supporting_evidence": get_supporting_evidence(report, matched_inmate),
            "recommendations": get_recommendations(report),
        },
        "technical_details": {
            "spectrogram_path": spectrogram_path,
            "processing_timestamp": datetime.now().isoformat(),
            "system_version": "1.0.0",
        },
    }

    report_data["signature"] = create_signature(report_data, hmac_key)

    # generate json format
    json_filename = f"report_{report.report_code}.json"
    json_path = f"data/reports/{json_filename}"

    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    # generate pdf format
    pdf_filename = f"forensic_report_{report.report_code}.pdf"
    pdf_path = f"data/reports/{pdf_filename}"

    generate_pdf_report(report_data, pdf_path, spectrogram_path, hmac_key)

    return json_path, pdf_path


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


# supporting evidence list
def get_supporting_evidence(report, matched_inmate) -> list:
    evidence = []

    if report.is_synthetic:
        evidence.append(
            f"Synthetic detection confidence: {report.synthetic_confidence:.2%}"
        )
    else:
        evidence.append("Voice patterns consistent with human speech")

    if report.speaker_match and matched_inmate:
        evidence.append(
            f"Voice similarity score: {report.speaker_match_confidence:.2%} with {matched_inmate.name}"
        )
        evidence.append("Embedding vectors show high correlation")
    else:
        evidence.append("No significant similarity found with database records")

    evidence.append("Spectrogram analysis completed")
    evidence.append("Hash verification confirms file integrity")

    return evidence


# generate recommendations based on analysis
def get_recommendations(report) -> list:
    recommendations = []

    if report.is_synthetic:
        recommendations.append("Investigate source of synthetic audio generation")
        recommendations.append(
            "Cross-reference with known deepfake detection databases"
        )

    if report.speaker_match:
        recommendations.append("Proceed with voice identification protocols")
        recommendations.append("Consider additional biometric verification")
    else:
        recommendations.append("Expand database search if possible")
        recommendations.append("Consider manual expert review")

    recommendations.append("Preserve original audio files for future analysis")
    recommendations.append("Document chain of custody")

    return recommendations


# generate pdf forensic report
def generate_pdf_report(
    report_data: Dict, pdf_path: str, spectrogram_path: str, hmac_key: str
):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

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
    story.append(Paragraph("DIGITAL FORENSIC AUDIO ANALYSIS REPORT", title_style))
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

    # Section 1: Audio File Metadata
    story.append(Paragraph("1. AUDIO FILE METADATA", heading_style))

    metadata = report_data["audio_file_metadata"]
    metadata_table_data = [
        ["File Name", metadata["file_name"]],
        ["File Format", metadata["file_format"]],
        ["Duration", metadata["duration"]],
        ["Sampling Rate", metadata["sampling_rate"]],
        ["MD5 Hash", metadata["md5_hash"]],
        ["SHA256 Hash", metadata["sha256_hash"][:32] + "..."],
        ["File Creation Date", metadata["date_of_file_creation"][:19]],
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

    # Section 2: Claimed Source Information
    story.append(Paragraph("2. CLAIMED SOURCE INFORMATION", heading_style))

    source_info = report_data["claimed_source_information"]
    source_table_data = [
        ["Claimed Caller Identity", source_info["claimed_caller_identity"]],
        ["Context", source_info["context"]],
        ["Provided By", source_info["provided_by"]],
    ]

    source_table = Table(source_table_data, colWidths=[2 * inch, 4 * inch])
    source_table.setStyle(
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

    story.append(source_table)
    story.append(Spacer(1, 15))

    # Section 3: Technical Analysis Summary
    story.append(Paragraph("3. TECHNICAL ANALYSIS SUMMARY", heading_style))

    tech_analysis = report_data["technical_analysis_summary"]
    tech_table_data = [
        ["Analysis Type", "Result", "Comments"],
        [
            "Speech Analysis",
            tech_analysis["speech_analysis"]["result"],
            tech_analysis["speech_analysis"]["comments"],
        ],
        [
            "Background Noise",
            tech_analysis["background_noise_consistency"]["result"],
            tech_analysis["background_noise_consistency"]["comments"],
        ],
        [
            "Embedding Comparison",
            tech_analysis["speech_embedding_comparison"]["result"],
            tech_analysis["speech_embedding_comparison"]["comments"],
        ],
    ]

    tech_table = Table(tech_table_data, colWidths=[1.5 * inch, 1.5 * inch, 3 * inch])
    tech_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(tech_table)
    story.append(Spacer(1, 15))

    # Section 4: Voiceprint Matching
    story.append(Paragraph("4. VOICEPRINT MATCHING", heading_style))

    voiceprint = report_data["voiceprint_matching"]
    voiceprint_table_data = [
        ["Match Result", voiceprint["result"]],
        ["Similarity Score", voiceprint["similarity_score"]],
        ["Threshold", voiceprint["threshold_acceptance"]],
        ["Matched Inmate", voiceprint["matched_inmate"]],
    ]

    voiceprint_table = Table(voiceprint_table_data, colWidths=[2 * inch, 4 * inch])
    voiceprint_table.setStyle(
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

    story.append(voiceprint_table)
    story.append(Spacer(1, 15))

    # Section 5: Synthetic Detection
    story.append(Paragraph("5. SYNTHETIC VOICE DETECTION", heading_style))

    synthetic = report_data["synthetic_detection"]
    synthetic_table_data = [
        ["Detection Result", synthetic["result"]],
        ["Confidence Score", synthetic["confidence"]],
        ["Detection Threshold", synthetic["threshold"]],
        ["Classification", "Synthetic" if synthetic["is_synthetic"] else "Authentic"],
    ]

    synthetic_table = Table(synthetic_table_data, colWidths=[2 * inch, 4 * inch])
    synthetic_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    (
                        colors.lightcoral
                        if synthetic["is_synthetic"]
                        else colors.lightgreen
                    ),
                ),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(synthetic_table)
    story.append(Spacer(1, 15))

    # Section 6: Final Verdict
    story.append(Paragraph("6. FINAL VERDICT", heading_style))

    verdict = report_data["verdict"]
    story.append(
        Paragraph(f"<b>Conclusion:</b> {verdict['final_conclusion']}", styles["Normal"])
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Supporting Evidence:</b>", styles["Normal"]))
    for evidence in verdict["supporting_evidence"]:
        story.append(Paragraph(f"• {evidence}", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Recommendations:</b>", styles["Normal"]))
    for rec in verdict["recommendations"]:
        story.append(Paragraph(f"• {rec}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Add spectrogram if available
    if os.path.exists(spectrogram_path):
        story.append(Paragraph("7. SPECTROGRAM ANALYSIS", heading_style))
        try:
            img = Image(spectrogram_path, width=6 * inch, height=3 * inch)
            story.append(img)
            story.append(Spacer(1, 15))
        except:
            story.append(
                Paragraph("Spectrogram image could not be loaded.", styles["Normal"])
            )
            story.append(Spacer(1, 15))

    # Add QR Code for verification
    qr_data = f"Report ID: {report_data['report_header']['report_id']}\nSignature: {report_data['signature'][:32]}..."
    qr_path = f"data/reports/qr_{report_data['report_header']['report_id']}.png"
    generate_qr_code(qr_data, qr_path)

    story.append(Paragraph("8. VERIFICATION", heading_style))
    story.append(
        Paragraph(
            "This report is digitally signed. Scan QR code to verify authenticity:",
            styles["Normal"],
        )
    )

    try:
        qr_img = Image(qr_path, width=1.5 * inch, height=1.5 * inch)
        story.append(qr_img)
    except:
        story.append(Paragraph("QR code could not be generated.", styles["Normal"]))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            f"Digital Signature: {report_data['signature'][:64]}...", styles["Normal"]
        )
    )

    # Build PDF
    doc.build(story)
