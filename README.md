# Violence Detection in Videos

This project is a **video-based violence detection system** built using PyTorch. It uses the **RWF-2000 dataset** to classify short video clips as either `Violence` or `Non-Violence`. The system can process both uploaded videos and real-time webcam streams using a Streamlit web app.

---

## Features

- **Video Classification**: Detects violence in short video clips.
- **Real-Time Detection**: Uses webcam input to detect violence in real-time.
- **Lightweight Model**: CNN + LSTM architecture for sequence modeling.
- **Simple Deployment**: Easily run the app locally or on Streamlit.

---

## Dataset

The model is trained on the **[RWF-2000 Dataset](https://huggingface.co/datasets/DanJoshua/RWF-2000)**:

- Total videos: 2,000  
  - Violence: 1,000  
  - Non-Violence: 1,000  
- Format: `.avi`  
- Duration: 5–10 seconds per video  
- Resolution: 640x360 pixels  

**Dataset structure:**

RWF-2000/
├── train/
│ ├── Fight/
│ └── NonFight/
└── val/
  ├── Fight/
  └── NonFight/



