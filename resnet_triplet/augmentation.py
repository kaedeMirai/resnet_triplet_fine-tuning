import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torchvision.transforms import v2


class SeasonTransform(torch.nn.Module):
    """Имитация смены сезона"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = v2.functional.to_pil_image(img)

        season = random.choice(["autumn", "winter", "summer", "spring"])

        if season == "autumn":
            img_np = np.array(img).astype(np.float32)
            img_np[:, :, 0] *= 0.8
            img_np[:, :, 1] *= 0.9
            img_np[:, :, 2] *= 0.6
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            return Image.fromarray(img_np)

        elif season == "winter":
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(0.6)
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(1.2)

        elif season == "summer":
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.4)
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(1.2)

        else:
            img_np = np.array(img).astype(np.float32)
            img_np[:, :, 1] *= 1.2
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            return Image.fromarray(img_np)


class WeatherTransform(torch.nn.Module):
    """Имитация погодных условий"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = v2.functional.to_pil_image(img)

        weather = random.choice(["cloudy", "foggy", "rainy"])

        if weather == "cloudy":
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(0.7)
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(0.9)

        elif weather == "foggy":
            img_np = np.array(img).astype(np.float32)
            fog = np.ones_like(img_np) * 255
            alpha = random.uniform(0.2, 0.4)
            img_np = img_np * (1 - alpha) + fog * alpha
            return Image.fromarray(img_np.astype(np.uint8))

        else:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(0.7)
            return img.filter(ImageFilter.GaussianBlur(radius=1))


class TimeOfDayTransform(torch.nn.Module):
    """Имитация времени суток"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = v2.functional.to_pil_image(img)

        time = random.choice(["dawn", "noon", "dusk", "overcast"])

        if time == "noon":
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(1.1)

        elif time == "overcast":
            enhancer = ImageEnhance.Color(img)
            return enhancer.enhance(0.5)

        img_np = np.array(img).astype(np.float32)

        if time == "dawn":
            img_np[:, :, 0] *= 1.2
            img_np[:, :, 2] *= 0.8

        elif time == "dusk":
            img_np[:, :, 0] *= 1.1
            img_np[:, :, 1] *= 0.9
            img_np[:, :, 2] *= 0.7
            img_np *= 0.8

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


class ShadowTransform(torch.nn.Module):
    """Добавляет тени от облаков"""

    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = v2.functional.to_pil_image(img)

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        num_shadows = random.randint(1, 3)

        for _ in range(num_shadows):
            x = random.randint(0, w)
            y = random.randint(0, h)
            radius = random.randint(20, 80)

            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - x) ** 2 + (Y - y) ** 2)

            shadow_mask = np.clip(1 - (dist / radius), 0, 1)
            shadow_intensity = random.uniform(0.2, 0.4)

            for c in range(3):
                img_np[:, :, c] = img_np[:, :, c] * (1 - shadow_mask * shadow_intensity)

        return Image.fromarray(img_np.astype(np.uint8))


class NoiseTransform(torch.nn.Module):
    """Добавляет гауссовский шум"""

    def __init__(self, p=0.3, sigma_range=(5, 10)):
        super().__init__()
        self.p = p
        self.sigma_range = sigma_range

    def forward(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = v2.functional.to_pil_image(img)

        img_np = np.array(img).astype(np.float32)

        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0, sigma, img_np.shape)
        img_np = img_np + noise

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


class HazeTransform(torch.nn.Module):
    """Имитация дымки/тумана"""

    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, img):
        if random.random() > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = v2.functional.to_pil_image(img)

        img_np = np.array(img).astype(np.float32)
        h, w = img_np.shape[:2]

        direction = random.choice(["horizontal", "vertical", "diagonal"])

        if direction == "horizontal":
            gradient = np.linspace(0, 1, w).reshape(1, w, 1)
            gradient = np.repeat(gradient, h, axis=0)
        elif direction == "vertical":
            gradient = np.linspace(0, 1, h).reshape(h, 1, 1)
            gradient = np.repeat(gradient, w, axis=1)
        else:
            gradient_h = np.linspace(0, 1, h).reshape(h, 1, 1)
            gradient_w = np.linspace(0, 1, w).reshape(1, w, 1)
            gradient = (gradient_h + gradient_w) / 2

        haze_intensity = random.uniform(0.15, 0.3)
        haze_color = np.array([200, 200, 220]).reshape(1, 1, 3)

        img_np = (
            img_np * (1 - gradient * haze_intensity)
            + haze_color * gradient * haze_intensity
        )

        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return Image.fromarray(img_np)


def create_advanced_v2_transforms():
    """Создает продвинутые v2 трансформации с кастомными аугментациями"""

    transform = v2.Compose(
        [
            v2.Resize(256),
            v2.RandomCrop(224),
            v2.RandomApply(
                [
                    SeasonTransform(p=0.8),
                    WeatherTransform(p=0.6),
                ],
                p=0.4,
            ),
            v2.RandomApply(
                [
                    TimeOfDayTransform(p=0.7),
                ],
                p=0.3,
            ),
            v2.RandomApply(
                [
                    ShadowTransform(p=0.8),
                    HazeTransform(p=0.6),
                ],
                p=0.2,
            ),
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                    ),
                    v2.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.5)),
                    v2.RandomRotation(degrees=(-8, 8)),
                ],
                p=0.5,
            ),
            NoiseTransform(p=0.2),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


def create_base_v2_transforms():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


def create_light_v2_transforms():
    """Создает облегченную версию с основными аугментациями"""

    transform = v2.Compose(
        [
            v2.Resize(256),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [
                    SeasonTransform(p=0.8),
                    WeatherTransform(p=0.6),
                ],
                p=0.4,
            ),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform


def create_simple_v2_transforms():
    """Простые аугментации без кастомных трансформаций"""

    transform = v2.Compose(
        [
            v2.Resize(256),
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                    ),
                    v2.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 2.0)),
                    v2.RandomRotation(degrees=(-15, 15)),
                ],
                p=0.6,
            ),
            v2.RandomApply(
                [
                    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                ],
                p=0.4,
            ),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform
