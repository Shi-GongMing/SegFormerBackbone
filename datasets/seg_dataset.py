from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


def _ensure_tuple(size: Sequence[int]) -> Tuple[int, int]:
    if len(size) != 2:
        raise ValueError("size must be [H, W]")
    return int(size[0]), int(size[1])


class CustomSegDataset(Dataset):
    """
    Custom segmentation dataset.

    Expected folder layout:
        split/images/*.jpg
        split/masks/*.png

    The mask filename stem must match the image filename stem.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        crop_size: Optional[Sequence[int]] = None,
        val_size: Optional[Sequence[int]] = None,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        hflip_prob: float = 0.5,
        training: bool = True,
        normalize: bool = True,
        color_jitter_prob: float = 0.2,
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.crop_size = _ensure_tuple(crop_size) if crop_size is not None else None
        self.val_size = _ensure_tuple(val_size) if val_size is not None else None
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.training = training
        self.normalize = normalize
        self.color_jitter_prob = color_jitter_prob

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.color_jitter = ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02)

        self.samples = self._build_samples()
        if len(self.samples) == 0:
            raise RuntimeError(f"No matched samples found in {self.image_dir} and {self.mask_dir}")

    def _build_samples(self) -> List[Dict[str, str]]:
        image_paths = sorted(self.image_dir.glob(f"*{self.img_suffix}"))
        samples = []
        for img_path in image_paths:
            mask_path = self.mask_dir / f"{img_path.stem}{self.mask_suffix}"
            if mask_path.exists():
                samples.append({"image": str(img_path), "mask": str(mask_path), "name": img_path.stem})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_pil(self, index: int):
        item = self.samples[index]
        image = Image.open(item["image"]).convert("RGB")
        mask = Image.open(item["mask"])
        return image, mask, item["name"]

    def _random_resize(self, image: Image.Image, mask: Image.Image):
        if self.crop_size is None:
            return image, mask

        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_h = max(32, int(self.crop_size[0] * scale))
        target_w = max(32, int(self.crop_size[1] * scale))

        image = TF.resize(image, [target_h, target_w], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [target_h, target_w], interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _pad_if_needed(self, image: Image.Image, mask: Image.Image):
        if self.crop_size is None:
            return image, mask

        crop_h, crop_w = self.crop_size
        pad_h = max(0, crop_h - image.height)
        pad_w = max(0, crop_w - image.width)

        if pad_h > 0 or pad_w > 0:
            image = TF.pad(image, [0, 0, pad_w, pad_h], fill=0)
            mask = TF.pad(mask, [0, 0, pad_w, pad_h], fill=255)
        return image, mask

    def _random_crop(self, image: Image.Image, mask: Image.Image):
        if self.crop_size is None:
            return image, mask

        image, mask = self._pad_if_needed(image, mask)
        i, j, h, w = torch.randint(0, image.height - self.crop_size[0] + 1, (1,)).item(), \
                     torch.randint(0, image.width - self.crop_size[1] + 1, (1,)).item(), \
                     self.crop_size[0], self.crop_size[1]

        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return image, mask

    def _center_or_resize_val(self, image: Image.Image, mask: Image.Image):
        if self.val_size is not None:
            image = TF.resize(image, self.val_size, interpolation=InterpolationMode.BILINEAR)
            mask = TF.resize(mask, self.val_size, interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _maybe_hflip(self, image: Image.Image, mask: Image.Image):
        if random.random() < self.hflip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def _maybe_color_jitter(self, image: Image.Image):
        if random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)
        return image

    def _to_tensor(self, image: Image.Image, mask: Image.Image):
        image = TF.to_tensor(image)
        if self.normalize:
            image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = np.array(mask, dtype=np.int64)
        mask = torch.from_numpy(mask).long()
        return image, mask

    def __getitem__(self, index: int):
        image, mask, name = self._load_pil(index)

        if self.training:
            image, mask = self._random_resize(image, mask)
            image, mask = self._maybe_hflip(image, mask)
            image = self._maybe_color_jitter(image)
            image, mask = self._random_crop(image, mask)
        else:
            image, mask = self._center_or_resize_val(image, mask)

        image, mask = self._to_tensor(image, mask)
        return {
            "image": image,
            "mask": mask,
            "name": name,
        }
