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

```
RWF-2000/
├── train/
│ ├── Fight/
│ ├── NonFight/
├── val/
│ ├── Fight/
│ └── NonFight/

```

## Theory / Approach

This project leverages **deep learning for video-based violence detection**, combining **Convolutional Neural Networks (CNNs)** with **Long Short-Term Memory (LSTM) networks** to handle both spatial and temporal information in videos.

### Key Components

1. **CNN (Convolutional Neural Network)**  
   - Extracts **spatial features** from individual video frames.  
   - Uses multiple convolutional layers followed by max-pooling to detect edges, motion patterns, and objects indicative of violent actions.

2. **LSTM (Long Short-Term Memory)**  
   - Processes sequences of frame-level features from the CNN.  
   - Captures **temporal dependencies**, allowing the model to understand motion and progression of events over time.

3. **Loss Function & Optimization**  
   - Uses **Cross-Entropy Loss** for binary classification (Violence vs Non-Violence).  
   - Optimized with **Adam optimizer** for efficient convergence.

4. **Preprocessing**  
   - Videos are converted into a fixed number of frames.  
   - Frames are resized and normalized to ensure consistent input dimensions for the CNN.

### Why This Approach?

- **Spatial + Temporal Modeling**: Violence in videos is not just about objects in a single frame but also about motion patterns and sequences of actions.  
- **Lightweight and Efficient**: CNN + LSTM is simpler and faster than 3D CNNs or Transformers, making it suitable for real-time detection.  
- **Flexibility**: Can handle both uploaded videos and real-time webcam input.

### Optional Improvements

- Using **3D CNNs** or **Vision Transformers (ViT)** for better spatio-temporal modeling.  
- Adding **data augmentation** to increase robustness to camera angle, lighting, and motion variations.  
- Using **attention mechanisms** in LSTM to focus on critical frames indicating violence.
