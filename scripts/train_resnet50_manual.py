import glob
import os
import random

import numpy as np
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
    def __init__(self, data_dir, transform=None, num_negatives=3, seed=42):
        """
        Args:
            data_dir: путь к директории с изображениями
            transform: трансформации для аугментации
            num_negatives: количество negative примеров для каждой пары (anchor, positive)
            seed: seed для воспроизводимости negative примеров
        """
        self.data_dir = data_dir
        self.transform = transform
        self.num_negatives = num_negatives
        self.seed = seed

        self.image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        self._generate_negatives()

        print(f"Found {len(self.image_paths)} images")
        print(f"Dataset size with {num_negatives} negatives per anchor: {len(self)}")

    def _generate_negatives(self):
        """Предварительно генерирует все negative индексы для воспроизводимости"""
        rng = random.Random(self.seed)
        self.negative_indices = []

        for anchor_idx in range(len(self.image_paths)):
            for _ in range(self.num_negatives):
                negative_idx = rng.randint(0, len(self.image_paths) - 1)
                while negative_idx == anchor_idx:
                    negative_idx = rng.randint(0, len(self.image_paths) - 1)
                self.negative_indices.append(negative_idx)

    def __len__(self):
        return len(self.image_paths) * self.num_negatives

    def __getitem__(self, idx):
        anchor_idx = idx // self.num_negatives

        anchor_path = self.image_paths[anchor_idx]
        anchor_img = Image.open(anchor_path).convert("RGB")

        positive_img = anchor_img.copy()

        negative_idx = self.negative_indices[idx]
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


def create_model(embedding_dim=256):

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, embedding_dim),
        nn.BatchNorm1d(embedding_dim),
    )

    nn.init.xavier_uniform_(model.fc[0].weight)
    nn.init.zeros_(model.fc[0].bias)

    return model


def create_transforms(is_training=True):
    """Создает трансформации для обучения и валидации"""
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


def create_v2transforms():
    return create_advanced_v2_transforms()


def validate_model(model, val_loader, criterion, device):
    """Функция валидации модели"""
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
    data_dir, epochs=50, batch_size=16, lr=1e-4, embedding_dim=256, val_split=0.2
):
    """Основная функция обучения с валидацией"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = TripletDataset(data_dir, transform=None, num_negatives=3, seed=42)

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    train_transform = create_v2transforms()
    val_transform = create_v2transforms()

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = create_model(embedding_dim)
    model = model.to(device)

    criterion = nn.TripletMarginLoss(margin=PARAMS["margin"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=PARAMS["weight_decay"]
    )

    print(f"Starting training for {epochs} epochs...")
    train_losses = []
    val_losses = []
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

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Model saved: model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "final_model_03.pth")
    print("Training completed. Final model saved as 'final_model_03.pth'")

    return model, train_losses, val_losses


def extract_embeddings(model, image_path, transform):
    """Извлекает embedding для одного изображения"""
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        embedding = model(image_tensor)

    return embedding.cpu().numpy()


if __name__ == "__main__":
    DATA_DIR = "150r-filtered"

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found!")
        exit(1)

    PARAMS = {
        "lr": 7.070841175369515e-05,
        "batch_size": 64,
        "embedding_dim": 512,
        "margin": 0.8081267628711091,
        "weight_decay": 1.0429648755277248e-06,
    }
    model, train_losses, val_losses = train_model(
        data_dir=DATA_DIR,
        epochs=40,
        batch_size=PARAMS["batch_size"],
        lr=PARAMS["lr"],
        embedding_dim=PARAMS["embedding_dim"],
        val_split=0.2,
    )

    print("Training finished!")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
