from typing import Dict, List, Optional

import torch


class SegmentationMetrics:
    """
    Accumulates confusion matrix and computes:
    - pixel accuracy
    - mean IoU
    - mean Dice
    - per-class IoU
    """

    def __init__(self, num_classes: int, ignore_index: int = 255, device: Optional[torch.device] = None):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.device = device if device is not None else torch.device("cpu")
        self.confmat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred: [B, H, W]
        target: [B, H, W]
        """
        pred = pred.view(-1)
        target = target.view(-1)

        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]

        if pred.numel() == 0:
            return

        k = (target >= 0) & (target < self.num_classes)
        inds = self.num_classes * target[k].to(torch.int64) + pred[k].to(torch.int64)
        self.confmat += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def reset(self):
        self.confmat.zero_()

    def compute(self) -> Dict[str, object]:
        confmat = self.confmat
        tp = torch.diag(confmat)
        total = confmat.sum()
        gt = confmat.sum(dim=1)
        pred = confmat.sum(dim=0)

        pixel_acc = (tp.sum() / total.clamp(min=1)).item()

        union = gt + pred - tp
        iou = tp / union.clamp(min=1)
        miou = iou.mean().item()

        dice = (2 * tp) / (gt + pred).clamp(min=1)
        mdice = dice.mean().item()

        per_class_iou = iou.tolist()
        per_class_dice = dice.tolist()

        return {
            "pixel_acc": pixel_acc,
            "miou": miou,
            "mdice": mdice,
            "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice,
        }


@torch.no_grad()
def batch_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    pred = torch.argmax(logits, dim=1)
    meter = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index, device=logits.device)
    meter.update(pred, target)
    return meter.compute()
