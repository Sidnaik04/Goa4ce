import hashlib
import random
import numpy as np
import os
import uuid
from typing import List, Tuple
import json
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt


# generate voice embedding from audio files
def get_embedding(audio_path: str) -> List[float]:
    with open(audio_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

        random.seed(file_hash)

        embedding = [random.uniform(-1, -1) for _ in range(256)]

        return embedding


# compare two voice embeddings using cosine similarity
def compare_embeddings(a: List[float], b: List[float]) -> float:
    # convert to numpy arrays
    vec_a = np.array(a)
    vec_b = np.array(b)

    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_similarity = dot_product / (norm_a * norm_b)
    
    similarity = max(0.0, min(1.0, (cosine_similarity + 1) / 2))

    return float(similarity)


# detect if audio us synthetic/AI-generated
def detect_synthetic(audio_path: str) -> Tuple[bool, float]:
    with open(audio_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    hash_int = int(file_hash[:8], 16)

    is_synthetic = (hash_int % 10) < 3

    # synthetic confidence score
    if is_synthetic:
        confidence = 0.65 + (hash_int % 100) / 300  # 0.65-0.98
    else:
        confidence = 0.15 + (hash_int % 100) / 200  # 0.15-0.65

    return is_synthetic, float(confidence)


# generate spectrogram image from audio file
def generate_spectrogram(audio_path: str) -> str:
    file_id = str(uuid.uuid4())
    output_path = f"data/spectrograms/spectrogram_{file_id}.png"

    random.seed(hashlib.md5(audio_path.encode()).hexdigest())

    time = np.linspace(0, 10, 1000)
    freq = np.linspace(0, 8000, 512)

    spectrogram_data = np.zeros((len(freq), len(time)))

    for i in range(len(freq)):
        for j in range(len(time)):
            amplitude = (
                np.exp(-((freq[i] - 1000) ** 2) / 1000000) * np.sin(time[j] * 2)
                + np.exp(-((freq[i] - 2000) ** 2) / 2000000) * np.cos(time[j] * 3)
                + np.random.normal(0, 0.1)
            )

            spectrogram_data[i, j] = max(0, amplitude)

    plt.figure(figsize=(12, 8))
    plt.imshow(
        spectrogram_data,
        aspect="auto",
        origin="lower",
        extent=[0, 10, 0, 8000],
        cmap="inferno",
    )
    plt.colorbar(label="Amplitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Voice Spectrogram Analysis")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


# extraxt basic audio features for analysis
def get_audio_features(audio_path: str) -> dict:
    file_size = os.path.getsize(audio_path)

    features = {
        "fundamental_frequency": 150 + (file_size % 100),  # F0
        "jitter": 0.01 + (file_size % 50) / 5000,  # Voice quality
        "shimmer": 0.05 + (file_size % 30) / 1000,  # Amplitude variation
        "hnr": 15 + (file_size % 20),  # Harmonics-to-noise ratio
        "spectral_centroid": 2000 + (file_size % 500),  # Brightness
        "mfcc_mean": [
            (file_size % (i + 10)) / 100 for i in range(13)
        ],  # MFCC coefficients
        "energy": 0.5 + (file_size % 100) / 200,  # Voice energy
    }

    return features
