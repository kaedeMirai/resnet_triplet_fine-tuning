import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet34_Weights, resnet34
from torchvision.transforms import v2
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


def create_model(embendding_dim=256):
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=embendding_dim),
        nn.BatchNorm1d(embendding_dim),
    )

    nn.init.xavier_uniform_(model.fc[0].weight)
    nn.init.zeros_(model.fc[0].bias)

    return model


def create_transforms():
    """Создает простые аугментации"""
    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform


def create_v2transforms():
    return create_advanced_v2_transforms()


def train_model(data_dir, epochs=50, batch_size=16, lr=1e-4, embedding_dim=256):
    """Основная функция обучения"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = create_v2transforms()
    dataset = TripletDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = create_model(embedding_dim)
    model = model.to(device)

    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

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

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"Loss": f"{total_loss/num_batches:.4f}"})

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Model saved: model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "final_model_03.pth")
    print("Training completed. Final model saved as 'final_model_03.pth'")

    return model


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
    DATA_DIR = "150r"

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found!")
        exit(1)

    model = train_model(
        data_dir=DATA_DIR, epochs=40, batch_size=64, lr=2e-4, embedding_dim=512
    )

    print("Training finished!")
