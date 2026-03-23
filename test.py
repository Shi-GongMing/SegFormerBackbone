import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from models.segformer_b0 import build_segformer_b0
from utils.visualize import get_default_palette


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for SegFormer-B0")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--input", type=str, required=True, help="Image file or folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Where prediction masks will be saved")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_image(image: Image.Image):
    original_size = image.size[::-1]  # H, W
    image_tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return image_tensor.unsqueeze(0), original_size


def save_mask(mask: np.ndarray, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(str(save_path))


def save_color_mask(mask: np.ndarray, save_path: Path):
    palette = get_default_palette(int(mask.max()) + 1 if mask.max() >= 0 else 1)
    color = palette[mask]
    Image.fromarray(color).save(str(save_path))


def collect_inputs(input_path: Path):
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in input_path.iterdir() if p.suffix.lower() in exts])


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device_name = cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    model = build_segformer_b0(
        num_classes=cfg["num_classes"],
        in_chans=cfg["model"].get("in_chans", 3),
        decoder_dim=cfg["model"].get("decoder_dim", 256),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_inputs(input_path)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {input_path}")

    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            tensor, original_size = preprocess_image(image)
            tensor = tensor.to(device)

            logits = model(tensor)
            logits = torch.nn.functional.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            save_mask(pred, output_dir / f"{img_path.stem}_mask.png")
            save_color_mask(pred, output_dir / f"{img_path.stem}_color.png")
            print(f"Saved prediction for {img_path.name}")

    print(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
