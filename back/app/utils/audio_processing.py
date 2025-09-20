import librosa

def load_audio(file_bytes, target_sr=16000):
    import io
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(file_bytes))
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # Convert to mono
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y
