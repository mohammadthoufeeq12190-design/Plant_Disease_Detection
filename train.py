import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import PlantDiseaseCNN

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load Training Dataset
train_dataset = datasets.ImageFolder(
    "dataset/train",
    transform=transform
)

# Load Validation Dataset
valid_dataset = datasets.ImageFolder(
    "dataset/valid",
    transform=transform
)

# Data Loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = PlantDiseaseCNN().to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10

for epoch in range(epochs):

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss:.4f}")

# Save Model
torch.save(model.state_dict(), "plant_disease_model.pth")

print("Model Saved Successfully!")