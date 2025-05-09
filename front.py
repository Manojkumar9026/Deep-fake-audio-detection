import streamlit as st
import requests
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Set page title and layout
st.set_page_config(page_title="Deep Fake Voice Detection", page_icon="🔊", layout="wide")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stAudio {
            background-color: white;
            padding: 5px;
            border-radius: 5px;
        }
        .stButton button {
            background-color: #007BFF !important;
            color: white !important;
            font-size: 16px !important;
            padding: 8px 20px !important;
            border-radius: 10px !important;
            display: flex;
            margin: 0 auto;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown("<h2 style='text-align: center; color: #007BFF;'>🔊 Deep Fake Voice Detection</h2>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("📂 Upload an audio file", type=["wav", "mp3", "flac", "ogg"])

API_URL = "http://127.0.0.1:8000/predict"  # FastAPI backend URL

if uploaded_file:
    st.markdown("### 🎵 Audio Preview")
    st.audio(uploaded_file, format='audio/wav')

    # Convert uploaded file to numpy array for analysis
    y, sr = librosa.load(uploaded_file, sr=None)

    # Display waveform & spectrogram side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Waveform")
        fig, ax = plt.subplots(figsize=(4, 2))  # Smaller size
        librosa.display.waveshow(y, sr=sr, ax=ax, color='blue')
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Amplitude", fontsize=8)
        ax.set_title("Waveform", fontsize=10)
        ax.tick_params(axis='both', labelsize=6)
        st.pyplot(fig)

    with col2:
        st.markdown("### 🎭 Spectrogram")
        fig, ax = plt.subplots(figsize=(4, 2))  # Smaller size
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='coolwarm')
        ax.set_title("Spectrogram", fontsize=10)
        ax.tick_params(axis='both', labelsize=6)
        st.pyplot(fig)

    # Centered Analyze button
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    analyze = st.button("🚀 Analyze Audio")
    st.markdown("</div>", unsafe_allow_html=True)

    if analyze:
        with st.spinner("Processing... 🔄"):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']

            # Display prediction
            st.success(f"✅ Prediction: {prediction}")

        else:
            st.error("❌ Error processing the file.")
