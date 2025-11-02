import os
import numpy as np
import librosa
from tqdm import tqdm
import re

# Paths
DATASET_PATH = "ALL"
OUTPUT_PATH = "spectrograms4"

# Emotion code → folder mapping
code_map = {
    "a": "ANG",
    "d": "DIS",
    "f": "FEA",
    "h": "HAP",
    "n": "NEU",
    "su": "SUR",
    "sa": "SAD"
}

# Create output folders
os.makedirs(OUTPUT_PATH, exist_ok=True)
for label in code_map.values():
    os.makedirs(os.path.join(OUTPUT_PATH, label), exist_ok=True)

# Function to create spectrogram
def create_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        return S_db
    except Exception as e:
        print(f"⚠️ Error processing {file_path}: {e}")
        return None

print("🎧 Generating spectrograms...")

# Process all WAV files
for file_name in tqdm(os.listdir(DATASET_PATH)):
    if not file_name.lower().endswith(".wav"):
        continue

    # Extract emotion code after underscore
    match = re.search(r'_(sa|su|[adfhn])\d*\.wav$', file_name, re.IGNORECASE)
    if not match:
        print(f"⚠️ Skipping {file_name} — cannot find emotion code")
        continue

    code = match.group(1).lower()
    emotion_label = code_map.get(code)
    if emotion_label is None:
        print(f"⚠️ Skipping {file_name} — unknown code: {code}")
        continue

    # Create spectrogram
    file_path = os.path.join(DATASET_PATH, file_name)
    spectrogram = create_spectrogram(file_path)
    if spectrogram is None:
        continue

    # Save as .npy
    save_path = os.path.join(OUTPUT_PATH, emotion_label, file_name.replace(".wav", ".npy"))
    np.save(save_path, spectrogram)

print("\n✅ Spectrograms successfully saved in 'spectrograms4/'")
