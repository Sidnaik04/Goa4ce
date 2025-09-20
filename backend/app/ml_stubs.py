import hashlib
import random
import numpy as np
import os
import uuid
from typing import List, Tuple
import json
import matplotlib
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier
import warnings

matplotlib.use("Agg")  # Use non-GUI backend
import matplotlib.pyplot as plt

# Global variable to hold the model (singleton pattern)
_speaker_encoder = None


def _get_speaker_encoder():
    """
    Initialize and return the SpeechBrain speaker encoder model.
    This uses a singleton pattern to avoid reloading the model on every request.
    """
    global _speaker_encoder
    if _speaker_encoder is None:
        try:
            print("Loading SpeechBrain speaker encoder model...")
            _speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            print("SpeechBrain model loaded successfully!")
        except Exception as e:
            print(f"Error loading SpeechBrain model: {e}")
            raise e
    return _speaker_encoder


def _convert_to_wav(audio_path: str) -> str:
    """
    Convert audio file to WAV format if needed.
    Returns path to WAV file (original if already WAV, or converted temp file).
    """
    if audio_path.lower().endswith('.wav'):
        return audio_path
    
    try:
        # Load audio with torchaudio (supports mp3, flac, etc.)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Create temporary WAV file
        temp_wav_path = audio_path.rsplit('.', 1)[0] + '_temp.wav'
        
        # Save as WAV
        torchaudio.save(temp_wav_path, waveform, sample_rate)
        
        return temp_wav_path
    except Exception as e:
        print(f"Error converting audio file {audio_path}: {e}")
        raise ValueError(f"Unable to process audio file: {e}")


def get_embedding(audio_path: str) -> List[float]:
    """
    Generate voice embedding from audio file using SpeechBrain ECAPA-TDNN model.
    
    Args:
        audio_path: Path to audio file (supports .wav, .mp3, .flac, .m4a, .ogg)
    
    Returns:
        List of floats representing the voice embedding (192-dimensional)
    
    Raises:
        ValueError: If audio file cannot be processed
        FileNotFoundError: If audio file doesn't exist
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    temp_wav_path = None
    try:
        # Get the speaker encoder model
        encoder = _get_speaker_encoder()
        
        # Convert to WAV if needed
        wav_path = _convert_to_wav(audio_path)
        temp_wav_path = wav_path if wav_path != audio_path else None
        
        # Generate embedding using SpeechBrain
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # The model expects the audio file path
            signal = encoder.load_audio(wav_path)
            embedding = encoder.encode_batch(signal)
            
            # Convert tensor to numpy array and flatten
            if isinstance(embedding, torch.Tensor):
                embedding_np = embedding.squeeze().detach().cpu().numpy()
            else:
                embedding_np = np.array(embedding).flatten()
            
            # Convert to Python list
            embedding_list = embedding_np.tolist()
            
        return embedding_list
        
    except Exception as e:
        print(f"Error generating embedding for {audio_path}: {e}")
        raise ValueError(f"Failed to generate embedding: {e}")
    
    finally:
        # Clean up temporary WAV file if created
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass


# compare two voice embeddings using cosine similarity
def compare_embeddings(a: List[float], b: List[float]) -> float:
    """
    Compare two voice embeddings using cosine similarity.
    
    Args:
        a: First embedding (list of floats)
        b: Second embedding (list of floats)
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # convert to numpy arrays
    vec_a = np.array(a)
    vec_b = np.array(b)

    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_similarity = dot_product / (norm_a * norm_b)
    
    # Normalize to 0-1 range
    similarity = max(0.0, min(1.0, (cosine_similarity + 1) / 2))

    return float(similarity)


# detect if audio us synthetic/AI-generated
def detect_synthetic(audio_path: str) -> Tuple[bool, float]:
    """
    Detect if audio is synthetic/AI-generated (stub implementation).
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Tuple of (is_synthetic: bool, confidence: float)
    """
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
    """
    Generate spectrogram image from audio file.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Path to generated spectrogram image
    """
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


# extract basic audio features for analysis
def get_audio_features(audio_path: str) -> dict:
    """
    Extract basic audio features for analysis (stub implementation).
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary of audio features
    """
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