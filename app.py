import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import os
import gdown  # Add this import

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
    # Download the model from Google Drive if it doesn't exist
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1Ygodl58JHyN8obNu0seK-t6VZ6-5ydu5"
        gdown.download(url, model_path, quiet=False)
    
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
def predict_sign(image_np):
    image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        top_prob, top_idx = torch.max(probs, 1)
        return classes[top_idx.item()], top_prob.item()

# Streamlit title and UI
st.title("🤟 Sign to Text - ASL Recognizer")
st.markdown("Show a letter in ASL to your webcam, and this app will predict it!")

class SignRecognizer(VideoTransformerBase):
    def __init__(self):
        self.last_sign = ""
        self.frame_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror effect

        # Get a square from center
        h, w, _ = img.shape
        side = min(h, w)
        center_crop = img[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]

        try:
            predicted_sign, confidence = predict_sign(center_crop)
            if confidence > 0.7:
                self.last_sign = f"{predicted_sign} ({confidence:.2f})"
        except Exception as e:
            self.last_sign = f"Error: {str(e)}"

        cv2.putText(img, f"Prediction: {self.last_sign}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

# Run webcam stream
webrtc_streamer(key="sign-demo", video_processor_factory=SignRecognizer)
