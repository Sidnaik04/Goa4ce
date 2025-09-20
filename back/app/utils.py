import hashlib, os
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_md5(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def generate_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=16000)
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path
