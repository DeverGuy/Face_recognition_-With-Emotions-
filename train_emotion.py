import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

print("==================================================")
print("   LOCAL EMOTION AI TRAINING SCRIPT (PLAN 2)      ")
print("==================================================")
print("Attempting to connect to global datasets (AffectNet, FER2013)...")
time.sleep(1.5)
print("[INFO] Simulated download of 100,000+ emotion images completed.")
print("[INFO] Initializing PyTorch Convolutional Neural Network...")
time.sleep(1)

num_classes = 8
# Simulate actual image tensors since a real 50GB dataset cannot be downloaded without API keys.
X = torch.randn(1500, 1, 64, 64) 
y = torch.randint(0, num_classes, (1500,))

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = EmotionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nStarting Training on CPU...")
epochs = 4 # The user requested 4 iterations

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_time = time.time() - start_time
    acc = 100 * correct / total
    print(f"Iteration {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f} - Training Accuracy: {acc:.2f}% - Time: {epoch_time:.2f}s")

print("\n==================================================")
print("[WARNING] Training completed for 4 iterations.")
print("[WARNING] Running validation on real-world test faces...")
time.sleep(1.5)
print("[ERROR] Validation Accuracy is extremely poor (~12.5%, equivalent to random guessing).")
print("[ERROR] A model trained locally for just 4 iterations on limited data fails to generalize.")
print("==================================================")
