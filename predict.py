import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import PlantDiseaseCNN

# Load model
model = PlantDiseaseCNN()
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=torch.device("cpu")))
model.eval()

# Class names
classes = ["Diseased", "Healthy"]

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Select image
image_path = "dataset/test/Healthy/HDnwQRcNJKDmHHAT8GmBz4Wqa8DQE5BpgWS9IBhOyAC8whCaLWPO5rsI7fG17VDMnLV0RKeKAR-0-Ndfwy3MJNIamcjzc3GkczHmOaU8RlqwIkZ48i3V565yoVKtkSOHAfQ5p-GvcWqTi_sgpDpRBgjOREF3f36fKcCDRJTP.jpg"

# Open image
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    outputs = model(image)

    probabilities = F.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probabilities, 1)

print("Prediction :", classes[predicted.item()])
print("Confidence :", round(confidence.item() * 100, 2), "%")