import os
import numpy as np
import librosa
from tqdm import tqdm

# Path to dataset (update this)
DATASET_PATH = "dataverse_files"
OUTPUT_PATH = "spectrograms3"

# Mapping from emotion text in filename → folder abbreviation
emotion_map = {
    "neutral": "NEU",
    "disgust": "DIS",
    "angry": "ANG",
    "sad": "SAD",
    "happy": "HAP",
    "fear": "FEA",
    "ps": "PS"
}

# Create output directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
for abbrev in emotion_map.values():
    os.makedirs(os.path.join(OUTPUT_PATH, abbrev), exist_ok=True)

# Function to create a spectrogram
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

# Iterate through all files in the dataset
for file_name in tqdm(os.listdir(DATASET_PATH)):
    if not file_name.lower().endswith(".wav"):
        continue

    # Extract emotion part of filename
    parts = file_name.split("_")
    emotion_raw = parts[-1].replace(".wav", "").lower().strip()

    # Map to folder abbreviation
    emotion_abbr = emotion_map.get(emotion_raw)
    if emotion_abbr is None:
        print(f"⚠️ Skipping {file_name} — unknown emotion: {emotion_raw}")
        continue
    file_path = os.path.join(DATASET_PATH, file_name)
    spectrogram = create_spectrogram(file_path)
    if spectrogram is None:
        continue

    # Save as NumPy array
    save_path = os.path.join(OUTPUT_PATH, emotion_abbr, file_name.replace(".wav", ".npy"))
    np.save(save_path, spectrogram)

print("\n✅ Spectrograms successfully saved in 'spectrograms3/'")