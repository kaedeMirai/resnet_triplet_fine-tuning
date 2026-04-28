# ResNet Triplet Fine-Tuning

Проект с экспериментами по дообучению ResNet-моделей на триплетах изображений. `anchor` и `positive` берутся из одного изображения с разными аугментациями, `negative` выбирается случайно из другого изображения. Основная цель - получить embedding-модель для поиска похожих тайлов/локаций.

## Основные файлы

| Путь | Назначение |
| --- | --- |
| `scripts/train_resnet34_optuna.py` | Главный сценарий для ResNet34: Optuna-подбор гиперпараметров, train/validation split, финальное обучение, сохранение лучшей и финальной модели. Раньше это был `main_val_optuna.py`. |
| `scripts/train_resnet34_basic.py` | Базовый ранний скрипт обучения ResNet34 без валидации и Optuna. Раньше `main.py`. |
| `scripts/train_resnet50_manual.py` | Ручной эксперимент с ResNet50, фиксированными параметрами и валидацией. Раньше `main_val.py`. |
| `scripts/train_resnet50_optuna.py` | Optuna-эксперимент для ResNet50. Раньше `main_val_v4_optuna.py`. |
| `scripts/train_oml_projection.py` | Эксперимент с OML ResNet50 MoCo v2 и projection head. Раньше `main_v1.py`. |
| `scripts/train_oml_finetune.py` | Fine-tuning OML ResNet50 MoCo v2 без projection head. Раньше `main_v2.py`. |
| `scripts/train_oml_optuna.py` | Optuna-эксперимент для OML ResNet50 MoCo v2. Раньше `main_v3_optuna.py`. |
| `resnet_triplet/augmentation.py` | Кастомные аугментации: сезон, погода, время суток, тени, шум, дымка и готовые transform pipeline. |
| `resnet_triplet/utils.py` | Утилита `save_triplet_images` для сохранения визуальных примеров триплетов. |
| `tools/filter_dataset.py` | Подготовка датасета: фильтрация монотонных изображений через OpenCV-признаки. |
| `tools/inspect_oml_models.py` | Вспомогательная проверка доступных моделей `open-metric-learning`. |
| `docs/experiment_notes.txt` | Черновые заметки по старым запускам. |
| `experiments/self_attention_demo.py` | Отдельный учебный пример self-attention, не часть пайплайна обучения ResNet. |

Минимальный набор для чистого GitHub-репозитория: `README.md`, `.gitignore`, `resnet_triplet/`, `scripts/train_resnet34_optuna.py`, `tools/filter_dataset.py`. Остальные скрипты можно оставить как историю экспериментов или удалить, если нужен максимально компактный проект.

## Данные

Скрипты ожидают плоскую директорию с изображениями:

```text
150r/
  image_001.jpg
  image_002.jpg
  ...
```

Поддерживаются расширения: `.jpg`, `.jpeg`, `.png`, `.bmp`.

В `TripletDataset` positive создается как копия anchor, затем к anchor/positive применяются независимые аугментации. Negative выбирается случайно из другого изображения.

## Запуск

Установить зависимости:

```bash
pip install torch torchvision pillow numpy tqdm optuna joblib opencv-python
```

Если используются OML-эксперименты:

```bash
pip install open-metric-learning
```

Основной Optuna-пайплайн:

```bash
python -m scripts.train_resnet34_optuna
```

По умолчанию он ищет данные в `150r/`, запускает 50 trials и затем обучает финальную модель. Результаты Optuna пишутся в `triplet_optimization.db` и `triplet_optimization_study.pkl`, а модели сохраняются как `best_optimized_model.pth`, `optimized_model_epoch_*.pth`, `final_optimized_model.pth`.

## Подготовка датасета

`tools/filter_dataset.py` удаляет изображения без выраженных признаков: однотонные лес/поле/вода, низкая вариативность яркости, низкая энтропия, мало SIFT-точек.

Перед запуском нужно поправить пути внизу файла:

```python
SOURCE_DIR = "/path/to/source"
TARGET_DIR = "150r-filtered"
```

Затем:

```bash
python -m tools.filter_dataset
```
