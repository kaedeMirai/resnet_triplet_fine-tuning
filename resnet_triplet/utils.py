import os

import torch
import torchvision.utils as vutils


def save_triplet_images(
    anchor, positive, negative, epoch, batch_idx, out_dir="triplets", max_samples=None
):
    """
    Сохраняет триплеты изображений из батча

    Args:
        anchor: якорные изображения [batch_size, channels, height, width]
        positive: позитивные изображения [batch_size, channels, height, width]
        negative: негативные изображения [batch_size, channels, height, width]
        epoch: номер эпохи
        batch_idx: номер батча
        out_dir: директория для сохранения
        max_samples: максимальное количество примеров для сохранения (None = все)
    """
    os.makedirs(out_dir, exist_ok=True)

    batch_size = anchor.shape[0]

    if max_samples is not None:
        batch_size = min(batch_size, max_samples)
        anchor = anchor[:batch_size]
        positive = positive[:batch_size]
        negative = negative[:batch_size]

    all_images = []
    for i in range(batch_size):
        all_images.extend([anchor[i], positive[i], negative[i]])
    imgs = torch.stack(all_images, dim=0)
    grid = vutils.make_grid(imgs, nrow=3, normalize=True, scale_each=True, padding=2)

    out_path = os.path.join(out_dir, f"epoch{epoch}_batch{batch_idx}_full.png")
    vutils.save_image(grid, out_path)
    print(f"Saved {batch_size} triplet examples to {out_path}")
