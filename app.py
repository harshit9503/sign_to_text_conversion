import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import os
import gdown
import mediapipe as mp
from collections import deque

# Define the Bottleneck and ResNet50 from original code
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=26):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion),
            )
        layers = [block(self.in_channels, channels, stride, downsample)]
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Load model
@st.cache_resource
def load_model():
    model_path = "resnet50_sign_language_mediapipe.pth"
    st.write(f"Checking for model file at: {os.path.abspath(model_path)}")
    
    # Download the model from Google Drive if it doesn't exist
    if not os.path.exists(model_path):
        st.write("Model file not found. Downloading from Google Drive...")
        try:
            url = "https://drive.google.com/uc?id=1Ygodl58JHyN8obNu0seK-t6VZ6-5ydu5"
            gdown.download(url, model_path, quiet=False)
            st.write("Download completed. Verifying file...")
        except Exception as e:
            st.error(f"Failed to download model file: {str(e)}")
            raise FileNotFoundError(f"Could not download model file: {str(e)}")
    
    # Verify the file exists after download
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {os.path.abspath(model_path)} after download attempt.")
        raise FileNotFoundError(f"Model file not found at {os.path.abspath(model_path)}")
    
    st.write("Loading model...")
    model = ResNet50(num_classes=26)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
classes = [chr(i) for i in range(65, 91)]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Predict function
def predict_sign(landmark_image, model, classes, transform):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Convert numpy image to PIL Image
    image = Image.fromarray(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        # Get top 3 predictions for debugging
        top_probs, top_indices = torch.topk(probabilities, 3)
        top_predictions = [(classes[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]
        # Top prediction
        top_prob, top_idx = torch.max(probabilities, 0)
        return classes[top_idx.item()], top_prob.item(), top_predictions

# Streamlit title and UI
st.title("🤟 Sign to Text - ASL Recognizer")
st.markdown("Show a letter in ASL to your webcam, and this app will predict it!")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

class SignRecognizer(VideoTransformerBase):
    def __init__(self):
        self.last_sign = ""
        self.frame_counter = 0
        # Smoothing parameters
        self.smoothing_window = 5  # Number of frames to average over
        self.prediction_buffer = deque(maxlen=self.smoothing_window)  # Store recent predictions
        self.confidence_buffer = deque(maxlen=self.smoothing_window)  # Store recent confidences
        self.confidence_threshold = 0.7  # Only accept predictions with confidence > 0.7
        self.last_predicted_sign = "None"
        self.last_confidence = 0.0

    def smooth_prediction(self, predicted_sign, confidence, top_predictions):
        # Add the current prediction and confidence to the buffers
        self.prediction_buffer.append(predicted_sign)
        self.confidence_buffer.append(confidence)

        # If the buffer isn't full yet, return the current prediction
        if len(self.prediction_buffer) < self.smoothing_window:
            return predicted_sign, confidence

        # Find the most common prediction in the buffer
        prediction_counts = {}
        for sign in self.prediction_buffer:
            prediction_counts[sign] = prediction_counts.get(sign, 0) + 1
        most_common_sign = max(prediction_counts, key=prediction_counts.get)

        # Calculate the average confidence for the most common sign
        avg_confidence = np.mean([conf for sign, conf in zip(self.prediction_buffer, self.confidence_buffer) if sign == most_common_sign])

        # Only update if the average confidence is above the threshold
        if avg_confidence >= self.confidence_threshold:
            return most_common_sign, avg_confidence
        else:
            return self.last_predicted_sign, self.last_confidence

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror effect

        # Process frame with MediaPipe
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Create a white background for landmarks
        h, w, _ = img.shape
        landmark_image = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the white background
                mp_drawing.draw_landmarks(
                    landmark_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),  # Red dots
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green lines
                )

                # Predict the sign
                predicted_sign, confidence, top_predictions = predict_sign(landmark_image, model, classes, transform)

                # Smooth the prediction
                smoothed_sign, smoothed_confidence = self.smooth_prediction(predicted_sign, confidence, top_predictions)

                # Update the last prediction if the confidence threshold is met
                if smoothed_confidence >= self.confidence_threshold:
                    self.last_predicted_sign = smoothed_sign
                    self.last_confidence = smoothed_confidence
                    self.last_sign = f"{self.last_predicted_sign} ({self.last_confidence:.2f})"
                else:
                    self.last_sign = "Confidence too low"

                # Debug: Display top predictions
                st.write(f"Top Predictions: {top_predictions}")
        else:
            self.last_sign = "No hand detected"

        # Display the prediction on the webcam feed
        cv2.putText(img, f"Prediction: {self.last_sign}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

# Run webcam stream with enhanced STUN and TURN server configuration
webrtc_streamer(
    key="sign-demo",
    video_processor_factory=SignRecognizer,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["turn:turn.anyfirewall.com:443?transport=tcp"], "username": "webrtc", "credential": "webrtc"}
        ]
    }
)

# Cleanup on app shutdown (optional, Streamlit handles most cleanup)
def cleanup():
    hands.close()

import atexit
atexit.register(cleanup)
