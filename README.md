# SegFormer-B0 Engineering Project

A clean PyTorch engineering project for semantic segmentation based on **SegFormer-B0**.

## Project structure

```text
segformer_b0_engineering_project/
├── configs/
│   └── segformer_b0_custom.yaml
├── datasets/
│   ├── __init__.py
│   └── seg_dataset.py
├── models/
│   ├── __init__.py
│   └── segformer_b0.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── metrics.py
│   └── visualize.py
├── infer.py
├── requirements.txt
├── train.py
└── README.md
```

## Features

- Pure PyTorch implementation of SegFormer-B0
- Clear config-driven training pipeline
- Real-time metric display during training:
  - loss
  - pixel accuracy
  - mIoU
  - mean Dice
  - learning rate
- CSV log + TensorBoard log
- Best checkpoint / last checkpoint saving
- Validation sample visualization export
- Simple inference script for single image or folder

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset format

```text
data/
├── train/
│   ├── images/
│   │   ├── 0001.jpg
│   │   └── 0002.jpg
│   └── masks/
│       ├── 0001.png
│       └── 0002.png
└── val/
    ├── images/
    │   ├── 1001.jpg
    │   └── 1002.jpg
    └── masks/
        ├── 1001.png
        └── 1002.png
```

Mask pixel values should be integer class IDs:
- `0 ... num_classes - 1`
- ignore label usually uses `255`

## Training

Edit `configs/segformer_b0_custom.yaml`, then run:

```bash
python train.py --config configs/segformer_b0_custom.yaml
```

## TensorBoard

```bash
tensorboard --logdir outputs
```

## Inference

```bash
python infer.py \
  --config configs/segformer_b0_custom.yaml \
  --checkpoint outputs/segformer_b0_custom/best_miou.pth \
  --input demo.jpg \
  --output_dir demo_outputs
```

## Notes

- This project is designed for **custom semantic segmentation tasks**
- The training script prioritizes **clarity + tunability**
- Validation metrics are computed each epoch
- Training progress bar shows batch-level metrics during training
