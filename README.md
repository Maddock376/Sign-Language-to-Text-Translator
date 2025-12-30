## Sign-Language-to-Text-Translator
Real-time sign language to text translation using MediaPipe keypoints and an CNN model.

The application leverages MediaPipe for hand keypoint extraction and a TensorFlow LSTM model for temporal gesture classification.

#Features

 - Real-time sign language recognition from live video input
 - Hand keypoint extraction using MediaPipe
 - Sequence-based gesture classification using an LSTM neural network
 - Modular design for easy dataset expansion and retraining
 - Live on-screen text output

#Technologies Used

 - Python
 - OpenCV – video capture and frame processing
 - MediaPipe – hand landmark detection and tracking
 - TensorFlow / Keras – LSTM-based deep learning model
 - NumPy – data handling and preprocessing

#System Architecture

1. Video Capture
Live video is captured from a webcam using OpenCV.
2. Keypoint Extraction
MediaPipe detects and tracks hand landmarks in each frame.
3. Sequence Formation
Extracted keypoints are stored as fixed-length sequences.
4. Gesture Classification
An LSTM model processes the sequences to predict the corresponding sign.
5. Text Output
The predicted sign is displayed as text in real time.

#Dataset

 - Gesture data is stored as NumPy (.npy) files.
 - Each gesture consists of multiple sequences.
 - Each sequence contains a fixed number of frames with extracted hand keypoints.
 - The dataset was manually created to ensure consistency and accuracy.
Note: The system is currently trained on a limited number of gesture classes for performance and training efficiency.

#Model Details

Architecture: Multi-layer LSTM with fully connected dense layers
Input: Sequences of hand keypoints
Output: Softmax classification over predefined sign classes
Framework: TensorFlow (Keras API)

#Installation
```
git clone https://github.com/your-username/sign-language-to-text-translator.git
cd sign-language-to-text-translator
pip install -r requirements.txt 
```
#Usage

Ensure a webcam is connected.
Run the application:
```
python app.py
```
Perform supported sign language gestures in front of the camera.
The predicted text will be displayed on screen in real time.

#Limitations

Supports a limited number of gesture classes
Performance depends on lighting conditions and camera quality

#Author

Joshua Christopher
Computer Engineering Graduate | Software & AI Enthusiast
