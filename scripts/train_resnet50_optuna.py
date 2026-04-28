import glob
import os
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import ResNet34_Weights, ResNet50_Weights, resnet34, resnet50
from tqdm import tqdm

from resnet_triplet.augmentation import create_advanced_v2_transforms
from resnet_triplet.utils import save_triplet_images


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(42)


class TripletDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        anchor_path = self.image_paths[idx]
        anchor_img = Image.open(anchor_path).convert("RGB")

        positive_img = anchor_img.copy()

        negative_idx = random.randint(0, len(self.image_paths) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.image_paths) - 1)

        negative_path = self.image_paths[negative_idx]
        negative_img = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor_img)
            positive = self.transform(positive_img)
            negative = self.transform(negative_img)
        else:
            base_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            anchor = base_transform(anchor_img)
            positive = base_transform(positive_img)
            negative = base_transform(negative_img)

        return anchor, positive, negative


def create_model(embedding_dim=512):
    print("Resnet50 model")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=embedding_dim),
        nn.BatchNorm1d(embedding_dim),
    )

    nn.init.xavier_uniform_(model.fc[0].weight)
    nn.init.zeros_(model.fc[0].bias)

    return model


def create_transforms(is_training=True):
    if is_training:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return train_transform
    else:
        val_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return val_transform


def create_v2transforms(is_training=True):
    if is_training:
        return create_advanced_v2_transforms()
    else:
        return create_advanced_v2_transforms()


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for anchor, positive, negative in val_loader:

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_val_loss += loss.item()
            num_batches += 1

    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    return avg_val_loss


def train_model(
    data_dir,
    epochs=10,
    batch_size=16,
    lr=1e-4,
    embedding_dim=256,
    margin=1.0,
    weight_decay=1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = TripletDataset(data_dir, transform=None)

    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    train_indices, val_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_transform = create_v2transforms(is_training=True)
    val_transform = create_v2transforms(is_training=False)

    train_dataset = TripletDataset(data_dir, transform=train_transform)
    train_dataset.image_paths = [
        full_dataset.image_paths[i] for i in train_indices.indices
    ]

    val_dataset = TripletDataset(data_dir, transform=val_transform)
    val_dataset.image_paths = [full_dataset.image_paths[i] for i in val_indices.indices]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = create_model(embedding_dim)
    model = model.to(device)

    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for anchor, positive, negative in pbar:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

            pbar.set_postfix(
                {"Train Loss": f"{total_train_loss/num_train_batches:.4f}"}
            )

        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss = validate_model(model, val_loader, criterion, device)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        print(
            f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    return best_val_loss


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    embedding_dim = trial.suggest_categorical("embedding_dim", [256, 512])
    margin = trial.suggest_float("margin", 0.5, 2.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    best_val_loss = train_model(
        data_dir="150r",
        epochs=20,
        batch_size=batch_size,
        lr=lr,
        embedding_dim=embedding_dim,
        margin=margin,
        weight_decay=weight_decay,
    )

    return best_val_loss


if __name__ == "__main__":
    DATA_DIR = "150r"

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found!")
        exit(1)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\nOptimization completed.")
    print("Best parameters:", study.best_params)
    print(f"Best validation loss: {study.best_value:.4f}")
