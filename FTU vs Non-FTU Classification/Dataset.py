import os
import torch
import pydicom
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt

# === SET SEED FOR REPRODUCIBILITY ===
def set_seed(seed=42):
    """
    Ensures consistent and reproducible results across runs.
    Sets seeds for random, numpy, and PyTorch, and configures cudnn for determinism.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    

# === IMAGE TRANSFORMATIONS ===
# Applied transformation differently for FTU and non-FTU images
ftu_transform = transforms.Compose([
    transforms.Resize((224, 224)),                # Resize to fixed size
    transforms.ToTensor(),                        # Convert to tensor         
    transforms.ColorJitter(contrast=0.1),         # Adjust contrast
    transforms.RandomRotation(10),                # Random rotation
    transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize to [-1, 1]
])

nonftu_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === CUSTOM DATASET CLASS ===
class DicomDataset(Dataset):
    """
    Custom PyTorch Dataset to load DICOM images with FTU/non-FTU distinction.
    Applies appropriate transformations and handles image errors gracefully.
    """
    def __init__(self, image_paths, labels, ftu_transform=None, nonftu_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.ftu_transform = ftu_transform
        self.nonftu_transform = nonftu_transform

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        path = self.image_paths[index]
        label = self.labels[index]

        try:
            dcm = pydicom.dcmread(path, force=True)
            if not hasattr(dcm, 'pixel_array'):
                raise RuntimeError("No pixel data")

            arr = dcm.pixel_array.astype(np.float32)

            if arr.ndim == 2:
                # Grayscale 2D
                ptp = np.ptp(arr)
                arr = (255 * (arr - arr.min()) / ptp).astype(np.uint8) if ptp != 0 else np.zeros_like(arr)
                img = Image.fromarray(arr).convert("RGB")

            elif arr.ndim == 3:
                if arr.shape[0] > 1:
                    # 3D stack: take middle slice
                    slice_2d = arr[arr.shape[0] // 2]
                    ptp = np.ptp(slice_2d)
                    slice_2d = (255 * (slice_2d - slice_2d.min()) / ptp).astype(np.uint8) if ptp != 0 else np.zeros_like(slice_2d)
                    img = Image.fromarray(slice_2d).convert("RGB")
                elif arr.shape[2] == 3:
                    # Already RGB
                    img = Image.fromarray(arr.astype(np.uint8))
                else:
                    raise RuntimeError(f"Unsupported shape: {arr.shape}")
            else:
                raise RuntimeError(f"Unsupported shape: {arr.shape}")

            if label == 1 and self.ftu_transform:
                img = self.ftu_transform(img)
            elif label == 0 and self.nonftu_transform:
                img = self.nonftu_transform(img)

            return img, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"❌ Skipped file: {os.path.basename(path)} — {e}")
            return self.__getitem__((index + 1) % len(self.image_paths))


# === LOAD IMAGE PATHS GROUPED BY CASE ID ===
def load_paths_by_case(ftu_dir, nonftu_dir):
    """
    Loads DICOM file paths and labels, and groups them by case ID.
    Returns a dictionary mapping case ID to (path, label) tuples.
    """
    case_dict = {}

    for file in os.listdir(ftu_dir):
        if file.endswith(".dcm"):
            case = file.split("_")[1]  # Extract case ID
            case_dict.setdefault(case, []).append((os.path.join(ftu_dir, file), 1))

    for file in os.listdir(nonftu_dir):
        if file.endswith(".dcm"):
            case = file.split("_")[1]
            case_dict.setdefault(case, []).append((os.path.join(nonftu_dir, file), 0))

    return case_dict

# === CREATE TRAIN AND TEST DATALOADERS ===
def create_dataloaders(ftu_dir, nonftu_dir, batch_size=64):
    """
    Splits data case-wise into train/test sets to avoid data leakage.
    Returns DataLoaders for training and testing.
    """
    set_seed(42)  # Ensure reproducibility

    # Group images by case
    case_dict = load_paths_by_case(ftu_dir, nonftu_dir)
    cases = list(case_dict.keys())

    # Case-wise train/validation split
    train_cases, val_cases = train_test_split(cases, test_size=0.2, random_state=42)

    # Flatten case_dict into image path lists
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for case in train_cases:
        for path, label in case_dict[case]:
            train_paths.append(path)
            train_labels.append(label)

    for case in val_cases:
        for path, label in case_dict[case]:
            val_paths.append(path)
            val_labels.append(label)

    # Initialize dataset and DataLoaders
    train_dataset = DicomDataset(train_paths, train_labels, ftu_transform=ftu_transform, nonftu_transform=nonftu_transform)
    test_dataset = DicomDataset(val_paths, val_labels, ftu_transform=ftu_transform, nonftu_transform=nonftu_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id))

    print(f"✅ Case-wise split: {len(train_dataset)} training images, {len(test_dataset)} validation images")

    return train_loader, test_loader

# === STANDALONE TEST EXECUTION ===
if __name__ == "__main__":
    # Set base directory to your dataset
    dataset_path = r"D:\Master Thesis\Automated Detection of Joint Inflammation in Horses Using Scintigraphic Imaging and CNNs\FTU & Non FTU classification\FTU_NONFTU_Dataset"
    ftu_dir = os.path.join(dataset_path, "FTU")
    nonftu_dir = os.path.join(dataset_path, "NonFTU")

    # Run DataLoader creation as a test
    train_loader, test_loader = create_dataloaders(ftu_dir, nonftu_dir, batch_size=64)
