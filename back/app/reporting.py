def generate_report(audio_path, results):
    import hashlib, datetime, os, json, qrcode
    from fpdf import FPDF

    # Define the directory for JSON reports
    JSON_DIR = "json_reports"  # Change this to your desired path
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)

    file_name = os.path.basename(audio_path)
    report_id = results["report_id"]

    # JSON Report (no chunk results)
    json_report = {
        "report_id": report_id,
        "file_name": file_name,
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "final_verdict": results.get("final_verdict", ""),
        "md5_hash": hashlib.md5(open(audio_path, "rb").read()).hexdigest()
    }

    json_path = os.path.join(JSON_DIR, f"{report_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=4, ensure_ascii=False)

    # PDF Report (no chunk tables or spectrograms)
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

    # Final Verdict
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Final Verdict: {results.get('final_verdict','')}", ln=True, align="C")

    # QR Code
    QR_DIR = "qr_codes"  # Define the directory for QR codes
    if not os.path.exists(QR_DIR):
        os.makedirs(QR_DIR)
    qr = qrcode.QRCode(box_size=3, border=2)
    qr.add_data(report_id)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white")
    qr_path = os.path.join(QR_DIR, f"{report_id}.png")
    img_qr.save(qr_path)
    pdf.image(qr_path, x=80, w=50)

    # Define the directory for PDF reports
    PDF_DIR = "pdf_reports"  # Change this to your desired path
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)

    # Save PDF
    pdf_path = os.path.join(PDF_DIR, f"{report_id}.pdf")
    pdf.output(pdf_path)

    return {
        "json_report": json_path,
        "pdf_report": pdf_path
    }
