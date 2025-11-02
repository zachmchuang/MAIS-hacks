import os
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

AUDIO_DIR = "C:\\Users\The Factory\Documents\GPU-Accelerated-Notebooks\MAIS-hacks\Audio_Speech_Actors_01-24"
OUTPUT_DIR = "spectrograms2"
N_FFT = 2048
HOP_LENGTH = 512

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for emotion in emotion_map.values():
    os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

def create_spectrogram(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        spec = librosa.amplitude_to_db(stft, ref=np.max)
        return spec        
    except Exception as e:
        print(f"error processing {audio_path}: {e}")
        return None

print("processing audio files...")

for root, dirs, files in os.walk(AUDIO_DIR):
    for file in tqdm(files):
        if file.endswith(".wav"):
            parts = file.split("-")
            if len(parts) < 7:
                print("skipping {file}. file name is irregular")
                continue
            emotion_id = parts[2]
            emotion = emotion_map.get(emotion_id)
            if emotion is None:
                print(f"emotion {emotion} is \"None\"")
                continue
            file_path = os.path.join(root, file)
            spectrogram = create_spectrogram(file_path)
            if spectrogram is None:
                print(f"error creating spectrogram for {file_path}")
                continue
            output_file = os.path.join(OUTPUT_DIR, emotion, file.replace(".wav", ".npy"))
            np.save(output_file, spectrogram)

print(f"successfully saved spectrograms to {OUTPUT_DIR}")