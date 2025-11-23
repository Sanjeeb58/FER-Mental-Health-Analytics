# utilities/train_model.py

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")

print("ADDING TO PYTHON PATH:", MODEL_PATH)
sys.path.insert(0, MODEL_PATH)


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os

# Custom models from the repo
from custom_cnn import CustomCNN
from swin import CustomSwinTransformer
from vit import CustomViT


# ----------------------------
# DEVICE SETUP (FORCE CPU)
# ----------------------------
device = torch.device("cpu")
print(f"Using device: {device}")


# ----------------------------
# TRAINING LOOP
# ----------------------------
def train(model, train_loader, criterion, optimizer):
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

    return running_loss / len(train_loader)


# ----------------------------
# VALIDATION LOOP
# ----------------------------
def validate(model, val_loader, criterion):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return running_loss / len(val_loader), accuracy


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def main(model_name, epochs, batch_size):

    # ----------------------------
    # DATA TRANSFORMS
    # ----------------------------
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_path = "FER2013_processed/train"
    val_path = "FER2013_processed/val"
    test_path = "FER2013_processed/test"

    if not os.path.exists(train_path):
        raise FileNotFoundError("Dataset not found! Run preprocess_data.py first.")

    # ----------------------------
    # LOAD DATASETS
    # ----------------------------
    train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ----------------------------
    # SELECT MODEL
    # ----------------------------
    if model_name == "custom_cnn":
        model = CustomCNN(num_classes=7)

    elif model_name == "swin_transformer":
        model = CustomSwinTransformer(pretrained=True, num_classes=7)

    elif model_name == "vit":
        model = CustomViT(pretrained=True, num_classes=7)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # ----------------------------
    # LOSS & OPTIMIZER
    # ----------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # ----------------------------
    # TRAINING
    # ----------------------------
    print(f"\nTraining {model_name} for {epochs} epochs on CPU...\n")

    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.2f}%\n")

    # ----------------------------
    # SAVE MODEL
    # ----------------------------
    save_path = f"Models/{model_name}_cpu.pth"
    os.makedirs("Models", exist_ok=True)
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")


# ----------------------------
# ARGUMENT PARSER
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["custom_cnn", "swin_transformer", "vit"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    main(args.model, args.epochs, args.batch_size)
