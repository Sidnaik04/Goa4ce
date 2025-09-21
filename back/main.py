from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import router
from app.routes.predict import router as detector_router

app = FastAPI(
    title="Deepfake Audio Detector",
    description="AI-powered deepfake detection for audio files",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(detector_router, prefix="/api/v1", tags=["detection"])

# Home page UI
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Deepfake Audio Detector</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f8f9fa; padding: 20px; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 20px; }
            .btn { background: #3498db; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; width: 100%; margin-top: 10px; }
            .btn:hover { background: #2980b9; }
            #results { margin-top: 20px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Deepfake Audio Detector</h1>
            <input type="file" id="audioFile" accept="audio/*">
            <button class="btn" onclick="analyzeAudio()">🔍 Analyze Audio</button>
            <div id="results"></div>
        </div>

        <script>
            async function analyzeAudio() {
                const fileInput = document.getElementById('audioFile');
                const resultsDiv = document.getElementById('results');
                if (!fileInput.files.length) { 
                    alert("Please select an audio file!"); 
                    return; 
                }

                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append("file", file);

                resultsDiv.innerHTML = "<p>Analyzing... ⏳</p>";

                try {
                    const response = await fetch("/api/v1/predict/", { method: "POST", body: formData });
                    const data = await response.json();

                    if (data.error) {
                        resultsDiv.innerHTML = `<p style="color:red;">${data.error}</p>`;
                        return;
                    }

                    // Display only final verdict and PDF link
                    let html = `<h3>Final Verdict: ${data.report_data.final_verdict}</h3>`;
                    html += `<p><a href="${data.report_files.pdf_report}" target="_blank">📄 Download PDF Report</a></p>`;
                    resultsDiv.innerHTML = html;

                } catch (err) {
                    resultsDiv.innerHTML = "<p style='color:red;'>Error analyzing audio</p>";
                    console.error(err);
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    uvicorn.r
