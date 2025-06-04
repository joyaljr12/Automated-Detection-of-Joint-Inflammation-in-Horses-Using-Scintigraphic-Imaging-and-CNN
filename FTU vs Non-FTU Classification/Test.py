import os
import torch
import torch.nn as nn
import time
from Dataset import create_dataloaders
from Model import FTUCNN

# Set dataset path
dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\FTU & Non FTU classification\FTU_NONFTU_Dataset"
ftu_dir = os.path.join(dataset_path, "FTU")
nonftu_dir = os.path.join(dataset_path, "NonFTU")

# Load test data
_, test_loader = create_dataloaders(ftu_dir, nonftu_dir, batch_size=64)

# Set device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained model
model = FTUCNN().to(device)  # Ensure model is initialized correctly
model.load_state_dict(torch.load(r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\FTU & Non FTU classification\Models\model_FTU_nonftu.pth"))


# Validation
def test_model():
    """Function to evaluate the model on test data"""
    start = time.time()
    total = 0
    correct = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # No gradients needed during testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Get predicted class

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    end = time.time()
    
    print(f'Validation Accuracy: {accuracy:.2f}%')
    print(f'Execution Time: {end - start:.2f} seconds')

    return all_labels, all_predictions

# Run the test function
if __name__ == "__main__":
    all_labels, all_predictions = test_model()

