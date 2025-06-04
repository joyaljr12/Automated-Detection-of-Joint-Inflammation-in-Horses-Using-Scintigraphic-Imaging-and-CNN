import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from Model import FTUCNN
from Dataset import create_dataloaders

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set dataset path
dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\FTU & Non FTU classification\FTU_NONFTU_Dataset"
ftu_dir = os.path.join(dataset_path, "FTU")
nonftu_dir = os.path.join(dataset_path, "NonFTU")

# Load training data
train_loader,_ = create_dataloaders(ftu_dir, nonftu_dir, batch_size=64) 

# Initialize model
model = FTUCNN().to(device)

# === Compute class weights from train_loader ===
all_labels = []
for _, labels in train_loader:
    all_labels.extend(labels.tolist())

num_ftu = sum(1 for l in all_labels if l == 1)
num_nonftu = sum(1 for l in all_labels if l == 0)

# Penalize the majority class less
class_weights = torch.tensor([1.0 * num_ftu / num_nonftu, 1.0]).to(device)
loss_function = nn.CrossEntropyLoss(weight=class_weights)

# Define optimizer (AdamW optimizer with learning rate 0.001 and weight decay for regularization)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

# Training loop
def train_model(epochs=10):
    model.train()
    start = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_num, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)

            # Zero the gradients to prevent accumulation
            optimizer.zero_grad()

            # Backward pass
            loss.backward()

            # Optimization step: update model parameters
            optimizer.step()

            # Print loss for every 10 batches
            if (batch_num + 1) % 25 == 0:
                print(f'Batch:{batch_num+1}, Epoch :{epoch+1}, Loss: {loss.item():0.2f}')

            running_loss += loss.item() # Accumulate total loss for the epoch
        
        # Compute and print average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Avg loss: {epoch_loss:.4f}")

    end = time.time() # End timing the training process
    print(f'Training completed in {end - start:.2f} seconds')

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(project_root, "Models")
    os.makedirs(model_dir, exist_ok=True)

    # Save trained model
    model_path = os.path.join(model_dir, "model_FTU_nonftu.pth")
    torch.save(model.state_dict(), model_path)
    print("Model saved successfully!")

# Run training if script is executed directly
if __name__ == "__main__":
    train_model()