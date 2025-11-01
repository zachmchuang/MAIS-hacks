import librosa
import numpy as np
import matplotlib.pyplot as plt

y, sr = librosa.load("C:\MAIS-hacks\dataset1\crema-d-mirror\AudioMP3\\1001_DFA_ANG_XX.mp3", sr = 16000)

D = librosa.stft(y, n_fft=1024, hop_length=512)

S, phase = librosa.magphase(D)

S_db = librosa.amplitude_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram (STFT)')
plt.tight_layout()
plt.show()