import os
import numpy as np

ROOT_DIR = "spectrograms3"  # The root folder where your emotion subfolders are

for emotion in os.listdir(ROOT_DIR):
    emotion_path = os.path.join(ROOT_DIR, emotion)
    if not os.path.isdir(emotion_path):
        continue

    print(f"\n Emotion: {emotion}")
    for file in os.listdir(emotion_path):
        if file.endswith(".npy"):
            path = os.path.join(emotion_path, file)
            arr = np.load(path)
            print(f"{file}: {arr.shape}")
