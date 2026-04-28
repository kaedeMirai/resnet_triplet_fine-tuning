import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from oml.models import ResnetExtractor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


def create_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class OMLWithProjection(nn.Module):
    """ВАРИАНТ 1: OML + проекционный слой 2048 -> меньший размер"""

    def __init__(self, projection_dim=512, freeze_backbone=False):
        super().__init__()

        self.backbone = ResnetExtractor.from_pretrained("resnet50_moco_v2")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone заморожен - обучаем только проекционный слой")
        else:
            print("Backbone разморожен - обучаем всю модель")

        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, projection_dim),
            nn.BatchNorm1d(projection_dim),
        )

        for layer in self.projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):

        features = self.backbone(x)

        projected = self.projection(features)

        projected = F.normalize(projected, p=2, dim=1)

        return projected


class OMLFreeze(nn.Module):
    """ВАРИАНТ 2: Замороженная OML + только проекционный слой"""

    def __init__(self, projection_dim=256):
        super().__init__()

        self.backbone = ResnetExtractor.from_pretrained("resnet50_moco_v2")
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(2048, projection_dim), nn.BatchNorm1d(projection_dim)
        )

        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)

        print(f"OML заморожена, обучается только проекция 2048->{projection_dim}")

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)

        projected = self.projection(features)
        projected = F.normalize(projected, p=2, dim=1)
        return projected


def train_oml_model(
    data_dir,
    model_type="projection",
    projection_dim=512,
    freeze_backbone=False,
    epochs=20,
    batch_size=32,
    lr=1e-4,
):
    """Обучение OML модели"""

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_type == "projection":
        model = OMLWithProjection(
            projection_dim=projection_dim, freeze_backbone=freeze_backbone
        )
    else:
        model = OMLFreeze(projection_dim=projection_dim)

    model = model.to(device)

    transform = create_transforms()
    dataset = TripletDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    criterion = nn.TripletMarginLoss(margin=0.5)

    if not freeze_backbone and model_type == "projection":
        optimizer = torch.optim.Adam(
            [
                {
                    "params": model.backbone.parameters(),
                    "lr": lr * 0.1,
                },
                {"params": model.projection.parameters(), "lr": lr},
            ],
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for anchor, positive, negative in pbar:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

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

        if (epoch + 1) % 5 == 0:
            save_name = f"oml_{model_type}_{projection_dim}d_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"Model saved: {save_name}")

    final_name = f"oml_{model_type}_{projection_dim}d_final.pth"
    torch.save(model.state_dict(), final_name)
    print(f"Training completed. Final model saved as '{final_name}'")

    return model


if __name__ == "__main__":
    DATA_DIR = "150r"

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found!")
        exit(1)

    print("Variant 1: OML with projection head, unfrozen backbone")
    model1 = train_oml_model(
        data_dir=DATA_DIR,
        model_type="projection",
        projection_dim=512,
        freeze_backbone=False,
        epochs=10,
        batch_size=16,
        lr=1e-4,
    )

    print("\nVariant 2: frozen OML backbone with projection head")
    model2 = train_oml_model(
        data_dir=DATA_DIR,
        model_type="freeze",
        projection_dim=256,
        epochs=15,
        batch_size=32,
        lr=1e-3,
    )

    print("Training finished!")
