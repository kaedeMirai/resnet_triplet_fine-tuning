import glob
import os
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from oml.models import ResnetExtractor
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
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


def create_v2transforms():
    return create_advanced_v2_transforms()


def create_oml_model():
    model = ResnetExtractor.from_pretrained("resnet50_moco_v2")
    return model


def create_transforms():
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


def compute_recall_at_k(model, dataloader, device, k=1):
    model.eval()
    anchors, positives = [], []
    with torch.no_grad():
        for anchor, positive, _ in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)

            a_emb = model(anchor)
            p_emb = model(positive)

            a_emb = F.normalize(a_emb, p=2, dim=1)
            p_emb = F.normalize(p_emb, p=2, dim=1)

            anchors.append(a_emb.cpu())
            positives.append(p_emb.cpu())

    anchors = torch.cat(anchors, dim=0)
    positives = torch.cat(positives, dim=0)

    sims = torch.matmul(anchors, positives.t())
    correct = 0
    n = sims.size(0)

    for i in range(n):
        topk = torch.topk(sims[i], k=k).indices
        if i in topk:
            correct += 1

    return correct / n


def train_oml_model(
    data_dir, epochs=10, batch_size=16, lr=1e-4, margin=0.3, weight_decay=1e-4
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = create_v2transforms()
    full_dataset = TripletDataset(data_dir, transform=transform)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = create_oml_model().to(device)
    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for anchor, positive, negative in pbar:

            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )

            optimizer.zero_grad()
            a_emb, p_emb, n_emb = model(anchor), model(positive), model(negative)
            a_emb, p_emb, n_emb = (
                F.normalize(a_emb, p=2, dim=1),
                F.normalize(p_emb, p=2, dim=1),
                F.normalize(n_emb, p=2, dim=1),
            )
            loss = criterion(a_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{total_loss/(pbar.n+1):.4f}"})

    recall1 = compute_recall_at_k(model, val_loader, device, k=1)
    print(f"Validation Recall@1: {recall1:.4f}")
    return recall1


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    margin = trial.suggest_uniform("margin", 0.1, 0.6)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    recall1 = train_oml_model(
        data_dir="150r",
        epochs=5,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        weight_decay=weight_decay,
    )
    return recall1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Best parameters:", study.best_params)
    print("Best Recall@1:", study.best_value)
