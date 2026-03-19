import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler
import PIL
from utils.logger import get_logger

logger = get_logger("DatasetAnalyzer")

class DatasetAnalyzer:
    def __init__(self, data_path, dataset_type="auto"):
        self.data_path = data_path
        self.dataset_type = dataset_type  # 'csv', 'image_folder', 'mnist', 'cifar10', 'auto'
        self.metadata = {
            "type": None,
            "input_shape": None,
            "num_classes": None,
            "task": None,
            "num_samples": 0
        }
        self.dataset = None
        self.train_loader = None
        self.val_loader = None

    def analyze_and_load(self, batch_size=32):
        logger.info(f"Analyzing dataset: {self.data_path}")

        if self.dataset_type == "auto":
            self.dataset_type = self._infer_dataset_type()

        if self.dataset_type == "csv":
            self._load_csv()
        elif self.dataset_type == "image_folder":
            self._load_image_folder()
        elif self.dataset_type.lower() == "mnist":
            self._load_torchvision("mnist")
        elif self.dataset_type.lower() == "cifar10":
            self._load_torchvision("cifar10")
        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}")

        self._create_dataloaders(batch_size)
        logger.info(f"Dataset analyzed. Metadata: {self.metadata}")
        return self.metadata, self.train_loader, self.val_loader

    def _infer_dataset_type(self):
        if str(self.data_path).lower() in ["mnist", "cifar10"]:
            return str(self.data_path).lower()
        if os.path.isfile(self.data_path) and self.data_path.endswith(".csv"):
            return "csv"
        if os.path.isdir(self.data_path):
            return "image_folder"
        raise ValueError("Could not auto-detect dataset type from path.")

    def _load_csv(self):
        try:
            df = pd.read_csv(self.data_path)
            # Assume last column is target for simplicity in MVP
            target_col = df.columns[-1]
            features = df.drop(columns=[target_col])
            target = df[target_col]

            # Fill NaNs
            features = features.fillna(features.mean(numeric_only=True)).fillna(0)

            # Convert non-numeric
            features = pd.get_dummies(features).astype(float)

            # Scale features to prevent exploding gradients
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            features = pd.DataFrame(features_scaled, columns=features.columns)

            # Task inference
            if target.dtype == 'object' or len(target.unique()) < 20:
                self.metadata["task"] = "classification"
                target = target.astype('category').cat.codes
                self.metadata["num_classes"] = len(target.unique())
                y_tensor = torch.tensor(target.values, dtype=torch.long)
            else:
                self.metadata["task"] = "regression"
                self.metadata["num_classes"] = 1
                y_tensor = torch.tensor(target.values, dtype=torch.float32)

            x_tensor = torch.tensor(features.values, dtype=torch.float32)
            self.dataset = TensorDataset(x_tensor, y_tensor)

            self.metadata["type"] = "tabular"
            self.metadata["input_shape"] = (features.shape[1],)
            self.metadata["num_samples"] = len(df)

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise e

    def _load_image_folder(self):
        # Apply heavy data augmentation to combat overfitting
        transform = transforms.Compose([
            transforms.Resize((32, 32)), # Resize for compatibility with deeper CIFAR-like CNN templates
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset = datasets.ImageFolder(root=self.data_path, transform=transform)
        self.metadata["type"] = "image"
        self.metadata["input_shape"] = (3, 64, 64)
        self.metadata["num_classes"] = len(self.dataset.classes)
        self.metadata["task"] = "classification"
        self.metadata["num_samples"] = len(self.dataset)

    def _load_torchvision(self, name):
        if name == "mnist":
            transform = transforms.Compose([
                transforms.RandomRotation(10), # Slight rotation for robustness
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4), # Standard CIFAR augmentation
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        root = "./data"
        os.makedirs(root, exist_ok=True)

        if name == "mnist":
            self.dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
            self.metadata["input_shape"] = (1, 28, 28)
            self.metadata["num_classes"] = 10
        elif name == "cifar10":
            self.dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
            self.metadata["input_shape"] = (3, 32, 32)
            self.metadata["num_classes"] = 10

        self.metadata["type"] = "image"
        self.metadata["task"] = "classification"
        self.metadata["num_samples"] = len(self.dataset)

    def _create_dataloaders(self, batch_size):
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        # Mandatory Train/Validation Split
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])

        # Data Handling Rules: Shuffle training data, DO NOT shuffle validation data
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True if train_size > batch_size else False)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
