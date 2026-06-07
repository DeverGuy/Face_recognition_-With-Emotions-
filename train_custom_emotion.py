import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import sys

# Windows multithreading fix for torch dataloader
import multiprocessing
if sys.platform.startswith('win'):
    multiprocessing.freeze_support()

DATA_DIR = "For_Emotions"
BATCH_SIZE = 16
NUM_EPOCHS = 30 # Increased epochs for full convergence
MODEL_SAVE_PATH = "custom_emotion_model.pth"

def main():
    print("==================================================")
    print("       DEEP NEURAL NETWORK TRAINING PIPELINE      ")
    print("==================================================")
    
    if not os.path.exists(DATA_DIR):
        print("[ERROR] Dataset directory not found. Run scraper first.")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    # Data augmentation and normalization for training
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("[INFO] Loading internet scraped dataset into memory...")
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
        class_names = full_dataset.classes
        print(f"[INFO] Found {len(full_dataset)} images belonging to {len(class_names)} classes: {class_names}")
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}. Dataset may be empty.")
        return

    if len(full_dataset) < 10:
        print("[ERROR] Not enough data to train. Waiting for scraper...")
        return

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("[INFO] Building ResNet-18 Transfer Learning Architecture...")
    # Use weights parameter instead of pretrained
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze lower layers but unfreeze layer4 for deep feature fine-tuning
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        
    # Replace final fully connected layer to match our custom classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Use a smaller learning rate since we are unfreezing layer4
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0003)

    print(f"[INFO] Commencing Training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}")

    print("[INFO] Training Complete! Saving custom model weights...")
    torch.save({
        'state_dict': model.state_dict(),
        'classes': class_names
    }, MODEL_SAVE_PATH)
    
    print(f"[SUCCESS] Custom Internet Model successfully saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
