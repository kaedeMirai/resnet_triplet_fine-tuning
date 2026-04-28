import glob
import logging
import os
import random

import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.trial import TrialState
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
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


def create_model(embedding_dim=256, dropout_rate=0.0):
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features

    layers = [nn.Linear(in_features=in_features, out_features=embedding_dim)]

    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))

    layers.append(nn.BatchNorm1d(embedding_dim))

    model.fc = nn.Sequential(*layers)

    nn.init.xavier_uniform_(model.fc[0].weight)
    nn.init.zeros_(model.fc[0].bias)

    return model


def create_transforms(
    is_training=True, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0
):
    """Создает трансформации для обучения и валидации с настраиваемыми параметрами"""
    if is_training:
        train_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                ),
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


def train_model_with_params(
    data_dir,
    trial,
    epochs=20,
    val_split=0.2,
):
    """Основная функция обучения с параметрами от Optuna"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    margin = trial.suggest_float("margin", 0.5, 2.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

    brightness = trial.suggest_float("brightness", 0.1, 0.4)
    contrast = trial.suggest_float("contrast", 0.1, 0.4)
    saturation = trial.suggest_float("saturation", 0.1, 0.4)
    hue = trial.suggest_float("hue", 0.0, 0.1)

    full_dataset = TripletDataset(data_dir, transform=None)

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_transform = create_transforms(
        is_training=True,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    val_transform = create_transforms(is_training=False)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    model = create_model(embedding_dim, dropout_rate)
    model = model.to(device)

    criterion = nn.TripletMarginLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):

        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for anchor, positive, negative in train_loader:
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

        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss = validate_model(model, val_loader, criterion, device)

        trial.report(avg_val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(
            f"Trial {trial.number}, Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    return best_val_loss


def objective(trial):
    """Целевая функция для оптимизации"""
    try:
        DATA_DIR = "150r"

        if not os.path.exists(DATA_DIR):
            raise ValueError(f"Directory {DATA_DIR} not found!")

        best_val_loss = train_model_with_params(DATA_DIR, trial)
        return best_val_loss

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float("inf")


def run_optimization(n_trials=100, study_name="triplet_optimization"):
    """Запускает оптимизацию гиперпараметров"""

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())

    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1,
        ),
    )

    print(f"Starting optimization with {n_trials} trials...")
    print(f"Study will be saved to: {storage_name}")

    study.optimize(objective, n_trials=n_trials, timeout=None)

    print("\nOptimization completed!")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Best params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    joblib.dump(study, f"{study_name}_study.pkl")
    print(f"\nStudy saved to: {study_name}_study.pkl")

    return study


def train_best_model(study, data_dir, epochs=50):
    """Обучает финальную модель с лучшими параметрами"""
    print("Training final model with best parameters...")

    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = TripletDataset(data_dir, transform=None)

    total_size = len(full_dataset)
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_transform = create_transforms(
        is_training=True,
        brightness=best_params.get("brightness", 0.2),
        contrast=best_params.get("contrast", 0.2),
        saturation=best_params.get("saturation", 0.2),
        hue=best_params.get("hue", 0.0),
    )
    val_transform = create_transforms(is_training=False)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=best_params["batch_size"], shuffle=False, num_workers=2
    )

    model = create_model(
        best_params["embedding_dim"], best_params.get("dropout_rate", 0.0)
    )
    model = model.to(device)

    criterion = nn.TripletMarginLoss(margin=best_params["margin"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )

    print(f"Training final model for {epochs} epochs...")

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
            torch.save(model.state_dict(), "best_optimized_model.pth")
            print(f"New best model saved with val loss: {best_val_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"optimized_model_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "final_optimized_model.pth")
    print("Final optimized model saved!")

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
    print("Stage 1: hyperparameter optimization")

    study = run_optimization(n_trials=50, study_name="triplet_optimization")

    print("\nStage 2: final model training")

    DATA_DIR = "150r"
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found!")
        exit(1)

    final_model, train_losses, val_losses = train_best_model(study, DATA_DIR, epochs=20)

    print("\nOptimization and training completed.")
    print(f"Best validation loss: {study.best_trial.value:.4f}")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
