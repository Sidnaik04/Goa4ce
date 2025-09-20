import os
import json
import datetime
import hashlib
from fpdf import FPDF
import qrcode
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------- Directories ---------------- #
REPORT_DIR = "reports"
PDF_DIR = os.path.join(REPORT_DIR, "pdf")
JSON_DIR = os.path.join(REPORT_DIR, "json")
SPEC_DIR = os.path.join(REPORT_DIR, "spectrograms")
QR_DIR = os.path.join(REPORT_DIR, "signatures")

for d in [REPORT_DIR, PDF_DIR, JSON_DIR, SPEC_DIR, QR_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- Generate Report ---------------- #
def generate_report(audio_path, results):
    file_name = os.path.basename(audio_path)
    report_id = results["report_id"]

    # --- JSON Report ---
    json_report = {
        "report_id": report_id,
        "file_name": file_name,
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "chunk_index": c["chunk_index"],
                "prediction": c["prediction"],
                "genuine_score": c["genuine_score"],
                "deepfake_score": c["deepfake_score"]
            } for c in results["chunk_results"]  # remove raw audio
        ],
        "final_verdict": results.get("final_verdict", ""),
        "md5_hash": hashlib.md5(open(audio_path, "rb").read()).hexdigest()
    }

    json_path = os.path.join(JSON_DIR, f"{report_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=4, ensure_ascii=False)

    # --- PDF Report ---
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Deepfake Audio Analysis Report", ln=True, align="C")
    pdf.ln(5)

    # Metadata
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Report ID: {report_id}", ln=True)
    pdf.cell(0, 8, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0, 8, f"Analyzed By: Automated AI Tool", ln=True)
    pdf.cell(0, 8, f"Audio File: {file_name}", ln=True)
    pdf.cell(0, 8, f"MD5 Hash: {json_report['md5_hash']}", ln=True)
    pdf.ln(5)

    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(20, 8, "Chunk", border=1)
    pdf.cell(50, 8, "Prediction", border=1)
    pdf.cell(40, 8, "Genuine Score", border=1)
    pdf.cell(40, 8, "Deepfake Score", border=1)
    pdf.ln()

    # Table Rows + Heatmaps
    pdf.set_font("Arial", '', 12)
    for idx, chunk in enumerate(results['chunk_results']):
        pdf.cell(20, 8, str(chunk["chunk_index"]), border=1)
        pdf.cell(50, 8, chunk["prediction"], border=1)
        pdf.cell(40, 8, f"{chunk['genuine_score']:.2f}", border=1)
        pdf.cell(40, 8, f"{chunk['deepfake_score']:.2f}", border=1)
        pdf.ln()

        # Generate heatmap if audio available
        y = chunk.get("audio_chunk")
        if y is not None:
            S = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(6, 2))
            sns.heatmap(S_db, cmap="ocean")
            heatmap_path = os.path.join(SPEC_DIR, f"{report_id}_chunk{chunk['chunk_index']}.png")
            plt.savefig(heatmap_path, bbox_inches='tight')
            plt.close()
            pdf.image(heatmap_path, x=10, w=180)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Final Verdict: {results.get('final_verdict','')}", ln=True, align="C")

    # QR Code
    qr = qrcode.QRCode(box_size=3, border=2)
    qr.add_data(report_id)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")
    qr_path = os.path.join(QR_DIR, f"{report_id}.png")
    img_qr.save(qr_path)
    pdf.image(qr_path, x=80, w=50)

    # Save PDF
    pdf_path = os.path.join(PDF_DIR, f"{report_id}.pdf")
    pdf.output(pdf_path)

    return {
        "json_report": json_path,
        "pdf_report": pdf_path
    }
