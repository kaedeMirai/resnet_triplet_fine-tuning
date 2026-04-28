import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CRITERION = {
    "forest_canny": 500,
    "forest": 0.65,
    "plain": 0.6,
    "water": 0.55,
    "entropy_threshold": 5.5,
    "std_threshold": 25,
    "keypoints_min": 50,
}


def forest_canny_filter(image):
    """Детектирует монотонные изображения через поиск границ"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    white_pixels = np.sum(edges > 0)
    return white_pixels < CRITERION["forest_canny"]


def lacks_distinct_features(image):
    """Проверяет наличие различимых объектов"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (512, 512))

    sift = cv2.SIFT_create()
    keypoints = sift.detect(resized, None)

    return len(keypoints) < 50


def has_repetitive_pattern(image):
    """Детектирует повторяющиеся паттерны через FFT"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))

    f_transform = np.fft.fft2(resized)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    center = magnitude.shape[0] // 2
    magnitude[center - 5 : center + 5, center - 5 : center + 5] = 0

    max_magnitude = np.max(magnitude)
    mean_magnitude = np.mean(magnitude)

    return max_magnitude / (mean_magnitude + 1e-7) > 50


def has_low_brightness_variance(image):
    """Детектирует изображения с низкой вариативностью яркости"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    return std < 25


def is_forest(image_hsv):
    """Определяет лес по зеленым и коричневым оттенкам"""
    forest_mask = cv2.inRange(
        image_hsv, np.array([30, 35, 15]), np.array([178, 255, 255])
    )
    brown_mask = cv2.inRange(image_hsv, np.array([5, 0, 0]), np.array([30, 70, 70]))
    mask = cv2.bitwise_or(forest_mask, brown_mask)

    ratio_green = cv2.countNonZero(mask) / (image_hsv.size / 3)
    return ratio_green > CRITERION["forest"]


def is_plain(image_hsv):
    """Определяет сплошные поля"""
    plain_mask = cv2.inRange(image_hsv, np.array([5, 50, 50]), np.array([40, 255, 255]))
    ratio_plain = cv2.countNonZero(plain_mask) / (image_hsv.size / 3)
    return ratio_plain > CRITERION["plain"]


def is_water(image_hsv):
    """Определяет воду по синим оттенкам"""
    water_mask = cv2.inRange(
        image_hsv, np.array([90, 50, 50]), np.array([130, 255, 255])
    )
    ratio_water = cv2.countNonZero(water_mask) / (image_hsv.size / 3)
    return ratio_water > CRITERION["water"]


def has_low_texture_diversity(image):
    """Детектирует повторяющиеся текстуры через энтропию"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))

    block_size = 32
    entropies = []

    for i in range(0, 256 - block_size, block_size):
        for j in range(0, 256 - block_size, block_size):
            block = resized[i : i + block_size, j : j + block_size]
            hist = cv2.calcHist([block], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            entropies.append(entropy)

    mean_entropy = np.mean(entropies)
    return mean_entropy < 5.5


def is_featureless(image):
    """Расширенная проверка монотонности"""
    image_resized = cv2.resize(image, (256, 256))
    image_hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

    if (
        forest_canny_filter(image_resized)
        or is_water(image_hsv)
        or is_forest(image_hsv)
        or is_plain(image_hsv)
    ):
        return True

    if has_low_brightness_variance(image_resized):
        return True

    if has_low_texture_diversity(image_resized):
        return True

    if lacks_distinct_features(image_resized):
        return True

    return False


def filter_dataset(source_dir, target_dir):
    """
    Фильтрует датасет, удаляя монотонные изображения

    Args:
        source_dir: путь к исходному датасету
        target_dir: путь к отфильтрованному датасету
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    target_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    image_files = []
    for ext in image_extensions:
        image_files.extend(source_path.glob(f"**/*{ext}"))
        image_files.extend(source_path.glob(f"**/*{ext.upper()}"))

    print(f"Найдено изображений: {len(image_files)}")

    stats = {"total": len(image_files), "kept": 0, "filtered": 0, "errors": 0}

    for img_path in tqdm(image_files, desc="Фильтрация изображений"):
        try:
            image = cv2.imread(str(img_path))

            if image is None:
                print(f"Ошибка чтения: {img_path}")
                stats["errors"] += 1
                continue

            if is_featureless(image):
                stats["filtered"] += 1
            else:
                relative_path = img_path.relative_to(source_path)
                target_file = target_path / relative_path

                target_file.parent.mkdir(parents=True, exist_ok=True)

                shutil.copy2(img_path, target_file)
                stats["kept"] += 1

        except Exception as e:
            print(f"Ошибка обработки {img_path}: {e}")
            stats["errors"] += 1

    print("\nСтатистика фильтрации")
    print(f"Всего изображений:      {stats['total']}")
    print(
        f"Сохранено:              {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)"
    )
    print(
        f"Отфильтровано:          {stats['filtered']} ({stats['filtered']/stats['total']*100:.1f}%)"
    )
    print(f"Ошибок:                 {stats['errors']}")


if __name__ == "__main__":
    SOURCE_DIR = "/home/user/papka/main_services/learning_lln/renet34/150r-big"
    TARGET_DIR = "150r-filtered-big-03"

    print(f"Исходный датасет: {SOURCE_DIR}")
    print(f"Целевой датасет: {TARGET_DIR}")
    print()

    filter_dataset(SOURCE_DIR, TARGET_DIR)

    print(f"\nГотово. Отфильтрованный датасет сохранен в: {TARGET_DIR}")
