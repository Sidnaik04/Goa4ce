from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Import your router
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

# Include the router
app.include_router(detector_router, prefix="/api/v1", tags=["detection"])

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
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #f8f9fa;
                min-height: 100vh;
                padding: 20px;
                line-height: 1.6;
            }
            .container {
                max-width: 500px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
            }
            h1 {
                text-align: center;
                color: #2c3e50;
                margin-bottom: 10px;
                font-size: 2.2em;
                font-weight: 300;
            }
            .subtitle {
                text-align: center;
                color: #7f8c8d;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            .upload-box {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                padding: 40px 20px;
                text-align: center;
                margin: 20px 0;
                transition: all 0.3s ease;
                background: #f8f9fa;
            }
            .upload-box:hover {
                border-color: #3498db;
                background: #ebf3fd;
            }
            .upload-icon {
                font-size: 3em;
                color: #95a5a6;
                display: block;
                margin-bottom: 15px;
            }
            .upload-text {
                color: #2c3e50;
                font-size: 1.2em;
                margin-bottom: 5px;
                font-weight: 500;
            }
            .upload-subtext {
                color: #7f8c8d;
                font-size: 0.9em;
            }
            .form-group {
                margin: 20px 0;
            }
            .form-group label {
                display: block;
                color: #2c3e50;
                margin-bottom: 5px;
                font-weight: 500;
            }
            .form-group input[type="file"] {
                width: 100%;
                padding: 10px;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                background: white;
                font-size: 16px;
            }
            .form-group input[type="file"]:focus {
                outline: none;
                border-color: #3498db;
            }
            .btn {
                background: #3498db;
                color: white;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 10px;
                transition: background 0.3s;
            }
            .btn:hover {
                background: #2980b9;
            }
            .info-box {
                background: #ecf0f1;
                padding: 15px;
                border-radius: 6px;
                margin: 20px 0;
                border-left: 4px solid #3498db;
            }
            .info-box h3 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .info-box ul {
                list-style: none;
                padding-left: 0;
                text-align: left;
            }
            .info-box li {
                padding: 3px 0;
                color: #34495e;
                position: relative;
                padding-left: 20px;
            }
            .info-box li:before {
                content: "→";
                color: #3498db;
                font-weight: bold;
                position: absolute;
                left: 0;
            }
            .footer {
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #ecf0f1;
                color: #95a5a6;
                font-size: 14px;
            }
            .footer a {
                color: #3498db;
                text-decoration: none;
            }
            .footer a:hover {
                text-decoration: underline;
            }
            @media (max-width: 480px) {
                .container {
                    margin: 10px;
                    padding: 20px;
                }
                h1 {
                    font-size: 1.8em;
                }
                .upload-box {
                    padding: 30px 15px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎤 Deepfake Detector</h1>
            <p class="subtitle">AI-powered audio analysis for deepfake detection</p>
            
            <form action="/api/v1/predict/" method="post" enctype="multipart/form-data">
                <div class="upload-box">
                    <div class="upload-icon">📁</div>
                    <div class="upload-text">Upload Audio File</div>
                    <div class="upload-subtext">MP3, WAV, M4A • Maximum 30 seconds</div>
                </div>
                
                <div class="form-group">
                    <label for="file">Select Audio File:</label>
                    <input type="file" id="file" name="file" accept="audio/*" required>
                </div>
                
                <button type="submit" class="btn">🔍 Analyze for Deepfake</button>
            </form>
            
            <div class="info-box">
                <h3>📊 Detection Results</h3>
                <ul>
                    <li><strong>🟢 Genuine</strong>: High confidence in real human voice</li>
                    <li><strong>🟡 Suspicious</strong>: Possible AI generation detected</li>
                    <li><strong>🔴 Deepfake</strong>: Confirmed AI-generated audio</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>
                    <a href="/docs" target="_blank">📖 API Documentation</a> | 
                    <a href="/api/v1/health" target="_blank">🔧 System Status</a>
                </p>
                <p>Deepfake Audio Detector v1.0</p>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    # Create uploads directory
    os.makedirs("uploads", exist_ok=True)
    
    print("🚀 Deepfake Detector starting...")
    print("📡 Web UI: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)