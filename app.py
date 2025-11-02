import streamlit as st
from st_audiorec import st_audiorec
import soundfile as sf
import numpy as np
import io
import os
from datetime import datetime

# -------- PAGE SETUP --------
st.set_page_config(page_title="Audio Recorder", page_icon="🎙️")
st.title("🎙️ Audio Recorder Web App")

st.write("Click the mic button below to record a short audio clip.")
st.write("Your recordings will be saved automatically in the **recordings/** folder.")

# -------- CREATE OUTPUT FOLDER --------
os.makedirs("recordings", exist_ok=True)

# -------- RECORD AUDIO --------
audio_data = st_audiorec()

# -------- SAVE FILE --------
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
