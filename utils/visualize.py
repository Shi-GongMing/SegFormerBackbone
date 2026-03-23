from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


def get_default_palette(num_classes: int):
    rng = np.random.default_rng(42)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def denormalize_image(tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    image = tensor * std + mean
    image = image.clamp(0, 1)
    image = (image * 255).byte().permute(1, 2, 0).cpu().numpy()
    return image


def mask_to_color(mask: np.ndarray, palette: np.ndarray):
    color = palette[mask]
    return color


def save_visual_triplet(
    image_tensor: torch.Tensor,
    pred_mask: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    save_path: str,
    class_names: Optional[Iterable[str]] = None,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = denormalize_image(image_tensor)

    pred = pred_mask.detach().cpu().numpy().astype(np.int64)

    gt_valid = None
    if gt_mask is not None:
        gt = gt_mask.detach().cpu().numpy().astype(np.int64)
        gt_valid = gt.copy()
        gt_valid[gt_valid < 0] = 0
        gt_valid[gt_valid == 255] = 0

        max_label = max(int(pred.max()), int(gt_valid.max()))
    else:
        max_label = int(pred.max())

    palette = get_default_palette(max_label + 1 if max_label >= 0 else 1)

    pred_color = mask_to_color(pred, palette)

    panels = [Image.fromarray(image), Image.fromarray(pred_color)]
    if gt_valid is not None:
        gt_color = mask_to_color(gt_valid, palette)
        panels.append(Image.fromarray(gt_color))

    titles = ["Input", "Prediction", "GroundTruth"][: len(panels)]

    width = sum(p.width for p in panels)
    height = max(p.height for p in panels) + 36
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    offset_x = 0
    for title, panel in zip(titles, panels):
        canvas.paste(panel, (offset_x, 36))
        draw.text((offset_x + 8, 10), title, fill=(0, 0, 0))
        offset_x += panel.width

    canvas.save(str(save_path))
