import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

from models.segformer_b0 import build_segformer_b0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LoveDA test-set prediction masks for submission"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="LoveDA test image root, e.g. data/loveda/test/images or data/loveda/test",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save submission masks")
    parser.add_argument("--device", type=str, default="", help="Optional override device: cuda / cpu / mps")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search images under test_dir and preserve subfolder structure",
    )
    parser.add_argument(
        "--exts",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
        help="Allowed image suffixes",
    )
    parser.add_argument("--save_zip", action="store_true", help="Zip the output_dir after mask generation")
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(cfg_device: str, override_device: str = ""):
    if override_device:
        name = override_device
    else:
        name = cfg_device

    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available, fallback to CPU.")
        return torch.device("cpu")

    if name == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps_ok:
            return torch.device("mps")
        print("MPS not available, fallback to CPU.")
        return torch.device("cpu")

    return torch.device("cpu")


def build_model(cfg, device):
    model = build_segformer_b0(
        num_classes=cfg["num_classes"],
        in_chans=cfg["model"].get("in_chans", 3),
        decoder_dim=cfg["model"].get("decoder_dim", 256),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()


def collect_images(test_dir: Path, recursive: bool, exts: List[str]) -> List[Path]:
    exts = {e.lower() for e in exts}
    if recursive:
        files = [p for p in test_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def preprocess(image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    original_size = image.size[::-1]  # (H, W)
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return tensor.unsqueeze(0), original_size


def pred_to_loveda_label(pred: np.ndarray) -> np.ndarray:
    """
    Convert model predictions back to LoveDA official label IDs.

    Assumed LoveDA training convention:
    - official labels 1..7 -> remapped during training to 0..6
    - no-data label 0 -> ignored during training as 255

    Therefore submission masks should be 1..7.
    """
    pred = pred.astype(np.uint8)
    return pred + 1


def save_mask(mask: np.ndarray, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(str(save_path))


def make_zip(output_dir: Path):
    import shutil
    zip_base = output_dir.parent / output_dir.name
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=str(output_dir))
    print(f"ZIP created: {archive}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = choose_device(cfg.get("device", "cuda"), args.device)

    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint, device)

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")

    image_paths = collect_images(test_dir, recursive=args.recursive, exts=args.exts)
    if len(image_paths) == 0:
        raise RuntimeError(f"No test images found under: {test_dir}")

    print("=" * 80)
    print("LoveDA submission inference")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Test images : {len(image_paths)}")
    print(f"Input root  : {test_dir}")
    print(f"Output dir  : {output_dir}")
    print("=" * 80)

    with torch.no_grad():
        for idx, img_path in enumerate(image_paths, start=1):
            image = Image.open(img_path).convert("RGB")
            tensor, original_size = preprocess(image)
            tensor = tensor.to(device)

            logits = model(tensor)
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)

            loveda_mask = pred_to_loveda_label(pred)

            if args.recursive:
                rel = img_path.relative_to(test_dir)
                save_path = output_dir / rel.with_suffix(".png")
            else:
                save_path = output_dir / f"{img_path.stem}.png"

            save_mask(loveda_mask, save_path)

            if idx % 50 == 0 or idx == len(image_paths):
                print(f"[{idx}/{len(image_paths)}] saved: {save_path}")

    if args.save_zip:
        make_zip(output_dir)

    print("Done. Submission masks generated successfully.")


if __name__ == "__main__":
    main()
