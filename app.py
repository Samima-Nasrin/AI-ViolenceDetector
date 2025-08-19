import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import av
import io
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ---------- MODEL DEFINITIONS ------------
class CNNEncoder(nn.Module):
    def __init__(self, out_features=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64*14*14, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, cnn_out=256, lstm_hidden=128, num_classes=2):
        super().__init__()
        self.cnn = CNNEncoder(out_features=cnn_out)
        self.lstm = nn.LSTM(input_size=cnn_out, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, -1)
        out, (h_n, c_n) = self.lstm(features)
        last = h_n[-1]
        out = self.fc(last)
        return out

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN_LSTM().to(device)
model.load_state_dict(torch.load("rwf2000_final_model.pth", map_location=device))
model.eval()

# ---------- UTILITY FUNCTIONS ----------
def avi_bytes_to_frames(avi_bytes, max_frames=32, every_n=5):
    container = av.open(io.BytesIO(avi_bytes))
    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i % every_n == 0:
            frames.append(np.array(frame.to_image()))
        if len(frames) >= max_frames:
            break
    if len(frames) == 0:
        return np.zeros((1, 112, 112, 3), dtype=np.uint8)
    return np.stack(frames)

def resize_frames(frames, size=(112,112)):
    resized = [cv2.resize(f, size) for f in frames]
    return np.stack(resized)

def predict_video_bytes(avi_bytes):
    frames = avi_bytes_to_frames(avi_bytes)
    frames = resize_frames(frames)
    frames_tensor = torch.from_numpy(frames).permute(0,3,1,2).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(frames_tensor), dim=1).item()
    return "Violence" if pred==1 else "Non-Violence"

# ---------- STREAMLIT UI ----------

st.set_page_config(
    page_title="Violence Detection App",  # Title in the browser tab
    page_icon="⚠",  # You can use an emoji or a local image file
    layout="centered"  # or "wide"
)

st.title("Violence Detection App")
option = st.radio("Choose mode:", ["Upload Video", "Webcam Real-Time"])

if option == "Upload Video":
    uploaded_file = st.file_uploader("Upload an AVI or MP4 video", type=["avi","mp4"])
    if uploaded_file:
        st.video(uploaded_file)
        bytes_data = uploaded_file.read()
        label = predict_video_bytes(bytes_data)
        st.markdown(f"### Prediction: **{label}**")

elif option == "Webcam Real-Time":
    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.frames_buffer = []

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            # Collect last 32 frames
            self.frames_buffer.append(cv2.resize(img, (112,112)))
            if len(self.frames_buffer) > 32:
                self.frames_buffer.pop(0)
            if len(self.frames_buffer) == 32:
                frames_tensor = torch.from_numpy(np.stack(self.frames_buffer)).permute(0,3,1,2).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = torch.argmax(model(frames_tensor), dim=1).item()
                label = "Violence" if pred==1 else "Non-Violence"
                color = (0,0,255) if pred==1 else (0,255,0)
                cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            return img

    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)