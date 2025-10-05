import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import librosa

MODEL_ID = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]

    return {
        "genuine_score": float(probs[0]),
        "deepfake_score": float(probs[1])
    }
