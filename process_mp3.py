import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa.display

AUDIO_DIR="C:\\Users\The Factory\Documents\GPU-Accelerated-Notebooks\MAIS-hacks\crema-d-mirror\AudioMP3\\"
OUTPUT_DIR = "spectrograms"
N_FFT = 2048
HOP_LENGTH = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")]

for file in files:
    try: 
        parts = file.split("_")
        if len(parts) < 4:
            print(f"file {file} name has wrong format.")
            continue
        emotion = parts[2].upper()

        emotion_folder = os.path.join(OUTPUT_DIR, emotion)
        os.makedirs(emotion_folder, exist_ok=True)

        filepath = os.path.join(AUDIO_DIR, file)
        y, sr = librosa.load(filepath, sr=None)

        stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
        spec = np.abs(stft)

        spec = (spec - spec.min()) / (spec.max() - spec.min())

        base_name = os.path.splitext(file)[0]
        save_path = os.path.join(emotion_folder, base_name)

        if os.path.exists(save_path):
            print(f"skipping {file} - already exists")
            continue

        np.save(f"{save_path}.npy", spec)

        print(f"processed {file} --> {emotion}/")

    except Exception as e:
        print(f"Error processing {file}: {e}")
print("done generating spectrograms")
