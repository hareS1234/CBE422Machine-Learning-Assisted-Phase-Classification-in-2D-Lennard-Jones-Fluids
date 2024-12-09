# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np

class PhaseDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, "r") as hf:
            self.positions = hf["positions"][:]
            self.labels = hf["labels"][:]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # shape: (50, 50)
        return torch.tensor(self.positions[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class PhaseClassifier(nn.Module):
    def __init__(self):
        super(PhaseClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50*50, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.fc(x)

def train_model(train_loader, model, criterion, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return model

if __name__ == "__main__":
    dataset = PhaseDataset("combined_dataset.h5")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhaseClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(train_loader, model, criterion, optimizer, device)
    torch.save(trained_model.state_dict(), "balanced_phase_classifier.pth")
    print("Model saved to balanced_phase_classifier.pth")

# Make sure the model can be imported
__all__ = ['PhaseClassifier']
