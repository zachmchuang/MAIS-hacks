import streamlit as st
from st_audiorec import st_audiorec
import soundfile as sf
import numpy as np
import io
import os
from datetime import datetime
import torch
import torch.nn as nn
import librosa
from pathlib import Path

# -------- PAGE SETUP --------
st.set_page_config(page_title="Audio Recorder", page_icon="🎙️")
st.title("🎙️ Audio Recorder & Emotion Detector")

# -------- MODEL DEFINITION --------
class EmotionDetector(nn.Module):
    def __init__(self, layers=4, channels_in=1, channels_out=2, kernel_size=4, num_classes=6, n_fft=2048):
        super().__init__()

        self.convs = nn.ModuleList()

        self.layers = layers
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.channels_mult = channels_out

        for i in range(layers):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(self.channels_in, self.channels_out, self.kernel_size, padding="same"),
                    nn.BatchNorm2d(self.channels_out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            self.channels_in = self.channels_out
            self.channels_out = self.channels_mult * self.channels_in

        self.freq_pool = nn.AdaptiveAvgPool2d((None, 256))
        self.fc1 = nn.Linear(262144, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        for conv in self.convs:
            x = conv(x)
        x = self.freq_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        return x

# -------- MODEL CONFIGURATION --------
MODEL_PATH = "emotion2_checkpoint.pt"
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- MODEL FUNCTIONS --------
@st.cache_resource
def load_model(path):
    """Load the emotion detection model (cached)"""
    try:
        model = torch.load(path, map_location=DEVICE, weights_only=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def process_audio(file_path):
    """Convert audio file to spectrogram tensor"""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    spec_db = librosa.amplitude_to_db(stft, ref=np.max)
    spec_db = (spec_db - spec_db.mean()) / (spec_db.std() + 1e-6)
    spec_tensor = torch.tensor(spec_db).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    return spec_tensor

def classify(model, spec_tensor):
    """Classify emotion from spectrogram"""
    with torch.no_grad():
        outputs = model(spec_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(outputs, dim=1).item()
    return CLASSES[predicted], probs[0].cpu().numpy()

# -------- CREATE OUTPUT FOLDER --------
os.makedirs("recordings", exist_ok=True)

# -------- TABS FOR RECORDING AND TESTING --------
tab1, tab2 = st.tabs(["🎤 Record Audio", "🧪 Test Model"])

# -------- TAB 1: RECORD AUDIO --------
with tab1:
    st.write("Click the mic button below to record a short audio clip.")
    st.write("Your recordings will be saved automatically in the **recordings/** folder.")

    # Record audio
    audio_data = st_audiorec()

    # Save file
    if audio_data is not None:
        # Convert recorded bytes into NumPy audio array
        wav_bytes = io.BytesIO(audio_data)
        data, samplerate = sf.read(wav_bytes, dtype='float32')

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/recording_{timestamp}.wav"

        # Save the audio file
        sf.write(filename, data, samplerate)

        st.success(f"✅ Audio saved as `{filename}`")

# -------- TAB 2: TEST MODEL --------
with tab2:
    st.write("Select a recording from the **recordings/** folder to detect its emotion.")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: `{MODEL_PATH}`")
        st.info("Please ensure the model checkpoint file is in the same directory as this script.")
    else:
        # Load model
        model = load_model(MODEL_PATH)
        
        if model is not None:
            # Get list of recordings
            recordings_dir = Path("recordings")
            recording_files = sorted(list(recordings_dir.glob("*.wav")), reverse=True)
            
            if len(recording_files) == 0:
                st.warning("⚠️ No recordings found. Record some audio first!")
            else:
                # Dropdown to select recording
                selected_file = st.selectbox(
                    "Choose a recording:",
                    recording_files,
                    format_func=lambda x: x.name
                )
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Button to run prediction
                    if st.button("🔍 Detect Emotion", type="primary"):
                        with st.spinner("Processing audio..."):
                            try:
                                # Process and classify
                                spec_tensor = process_audio(str(selected_file))
                                prediction, probabilities = classify(model, spec_tensor)
                                
                                # Display results
                                st.success(f"### Predicted Emotion: **{prediction.upper()}**")
                                
                                # Show confidence scores
                                st.write("#### Confidence Scores:")
                                for class_name, prob in zip(CLASSES, probabilities):
                                    st.progress(float(prob), text=f"{class_name}: {prob*100:.1f}%")
                                    
                            except Exception as e:
                                st.error(f"Error processing audio: {e}")
                
                with col2:
                    # Play selected audio
                    st.write("#### Preview:")
                    st.audio(str(selected_file))