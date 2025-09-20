# Voice Biometric Recognition & Synthetic Voice Detection System

**Status**: Backend Complete ✅ | Frontend TODO | ML Models TODO

A complete FastAPI backend for law enforcement voice biometric authentication and synthetic voice detection.

## 🚀 Quick Start (Backend)

```bash
# Clone and run
git clone <repo>
cd voice-biometric-backend
docker-compose up --build

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

**Default Login**: `username: officer`, `password: police123`

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   ML Models     │
│   (TODO)        │◄──►│   Backend       │◄──►│   (TODO)        │
│                 │    │   (COMPLETE)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                       ┌─────────────────┐
                       │   PostgreSQL    │
                       │   Database      │
                       └─────────────────┘
```

## 📡 API Endpoints (COMPLETE)

| Method | Endpoint | Purpose | Auth Required |
|--------|----------|---------|---------------|
| POST | `/auth/login` | Get JWT token | ❌ |
| POST | `/inmates` | Register inmate + voice | ✅ |
| GET | `/inmates` | List all inmates | ✅ |
| POST | `/upload` | Analyze audio file | ✅ |
| GET | `/reports` | List analysis reports | ✅ |
| GET | `/reports/{id}/pdf` | Download forensic report | ✅ |
| GET | `/reports/{id}/verify` | Verify report signature | ✅ |

## 🔧 For ML Engineer

### Current ML Implementation (Stubs)
Located in `app/ml_stubs.py` - **REPLACE WITH REAL MODELS**:

```python
# 🔄 REPLACE THESE FUNCTIONS:

def get_embedding(audio_path: str) -> list[float]:
    # TODO: Use Resemblyzer or similar
    # Return 256-dimensional voice embedding
    
def detect_synthetic(audio_path: str) -> tuple[bool, float]:
    # TODO: Use AASIST, WaveFake detector
    # Return (is_synthetic, confidence_score)
    
def compare_embeddings(a: list[float], b: list[float]) -> float:
    # TODO: Optimize similarity calculation
    # Return cosine similarity (0.0-1.0)
```

### Recommended Models
- **Voice Embeddings**: Resemblyzer, SpeechBrain, pyannote.audio
- **Synthetic Detection**: AASIST, RawNet2, WaveFake
- **Languages**: Hindi, English (Marathi, Kannada later)

### Integration Points
```python
# app/ml_stubs.py - Replace these functions
# Thresholds: app/crud.py line 45 (0.75 demo, 0.85 production)
# Audio processing: app/utils.py get_audio_metadata()
```

### Model Requirements
- **Input**: 16kHz WAV files, 2-30 seconds
- **Output**: 256-dim embeddings, 0-1 confidence scores
- **Performance**: <2 seconds processing time
- **Memory**: <2GB RAM per request

## 🎨 For Frontend Engineer

### Authentication Flow
```javascript
// 1. Login
POST /auth/login → { access_token, token_type }

// 2. Use token in headers
Authorization: Bearer {access_token}
```

### Core User Flows

#### 1. Register New Inmate
```javascript
// Form: name, inmate_code, audio_file
POST /inmates (multipart/form-data)
→ { id, name, inmate_code, voiceprint_id }
```

#### 2. Analyze Audio
```javascript
// Form: audio_file, optional inmate_code, context
POST /upload (multipart/form-data) 
→ { 
    report_id, 
    is_synthetic, 
    speaker_match, 
    confidence_scores,
    pdf_report_url 
}
```

#### 3. View Reports
```javascript
GET /reports → [{ report_code, is_synthetic, speaker_match, created_at }]
GET /reports/{id} → { full_json_report }
GET /reports/{id}/pdf → PDF file download
```

### UI Components Needed
1. **Login Page** - Simple form
2. **Dashboard** - Recent reports, statistics
3. **Inmate Registration** - Form + audio upload
4. **Audio Analysis** - Drag-drop audio + results
5. **Reports List** - Table with filters
6. **Report Viewer** - JSON/PDF display

### Sample API Responses
```javascript
// Analysis Result
{
  "report_id": "uuid",
  "report_code": "DFA-2024-03-ABC", 
  "is_synthetic": false,
  "synthetic_confidence": 0.15,
  "speaker_match": true,
  "speaker_match_confidence": 0.87,
  "matched_inmate_code": "INM001",
  "pdf_report_url": "/reports/uuid/pdf"
}
```

## 🗄️ Database Schema

```sql
-- Inmates table
inmates { id, name, inmate_code, created_at }

-- Voice embeddings  
voiceprints { id, inmate_id, embedding[], sample_audio_path, created_at }

-- Analysis reports
reports { 
  id, report_code, audio_path, 
  is_synthetic, synthetic_confidence,
  speaker_match, speaker_match_confidence,
  sha256, report_pdf_path, created_at
}
```

## 🧪 Testing

### Test Data Available
```bash
# Run to create sample audio files
python create_test_audio.py

# Run automated demo
python demo_script.py
```

### Postman Collection
Import `postman_collection.json` for complete API testing.

## 📁 Project Structure
```
voice-biometric-backend/
├── app/
│   ├── main.py          # FastAPI routes (COMPLETE)
│   ├── models.py        # Database models (COMPLETE)
│   ├── ml_stubs.py      # 🔄 ML interface (REPLACE)
│   ├── utils.py         # PDF generation (COMPLETE)
│   └── crud.py          # DB operations (COMPLETE)
├── data/                # File storage
├── docker-compose.yml   # Full stack setup
└── requirements.txt     # Dependencies
```

## 🎯 TODO for Team

### ML Engineer Priority
1. **Replace `ml_stubs.py` functions** with real models
2. **Add language detection** (Hindi/English)
3. **Optimize embedding similarity** thresholds
4. **Add model caching** for performance

### Frontend Engineer Priority  
1. **Build React/Vue dashboard** 
2. **Implement file upload UI** with drag-drop
3. **Create report viewing interface**
4. **Add real-time analysis feedback**

### Deployment
- Backend runs on port 8000
- Database on port 5432
- Frontend should run on port 3000
- All CORS configured for localhost

## 🔐 Security Notes
- JWT tokens expire in 24 hours
- All reports digitally signed (HMAC)
- File uploads validated and hashed
- QR codes for report verification

## 📞 Integration Examples

### Frontend API Calls
```javascript
// Login
const login = async () => {
  const response = await fetch('/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'officer', password: 'police123' })
  });
  const { access_token } = await response.json();
  localStorage.setItem('token', access_token);
};

// Upload audio
const analyzeAudio = async (audioFile, inmateCode) => {
  const formData = new FormData();
  formData.append('audio', audioFile);
  if (inmateCode) formData.append('inmate_code', inmateCode);
  
  const response = await fetch('/upload', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: formData
  });
  return response.json();
};
```

### ML Model Integration
```python
# In app/ml_stubs.py - replace with real implementation
from resemblyzer import VoiceEncoder

encoder = VoiceEncoder()

def get_embedding(audio_path: str) -> list[float]:
    wav = preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(wav)
    return embedding.tolist()
```

---

**Backend is 100% functional. Focus on ML models and frontend UI to complete the system.**