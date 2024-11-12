# custom_metrics.py

import torch
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS
from typing import List, Sequence, Optional


@METRICS.register_module()
class DiceCoefficient(BaseMetric):
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None):
        """
        Args:
            num_classes (int): Number of classes.
            ignore_index (int, optional): Index to ignore. Defaults to None.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        """Resets the internal evaluation state."""
        self.preds = []
        self.targets = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """
        Processes a batch of data and stores predictions and targets.

        Args:
            data_batch (dict): A batch of data containing model outputs.
            data_samples (Sequence[dict]): A sequence of data samples containing ground truth.
        """
        preds = data_batch['outputs']  # 모델의 출력 (logits)
        targets = data_samples['gt_semantic_seg']  # 실제 마스크

        # 예측값에 시그모이드 적용 후 임계값을 통해 이진화
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).float()

        # ignore_index가 설정된 경우 해당 인덱스 마스크
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds * mask.unsqueeze(1)
            targets = targets * mask

        # 다중 레이블이므로 각 클래스별로 마스크 분리
        for cls in range(self.num_classes):
            self.preds.append(preds[:, cls, :, :].cpu())
            self.targets.append((targets == cls).float().cpu())

    def compute_metrics(self, results: List[dict]) -> dict:
        """
        Computes the Dice coefficient for each class.

        Args:
            results (List[dict]): A list of results containing predictions and targets.

        Returns:
            dict: A dictionary with Dice coefficients for each class and the mean Dice.
        """
        dice_scores = torch.zeros(self.num_classes)
        for cls in range(self.num_classes):
            pred = self.preds[cls]
            target = self.targets[cls]

            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()

            if union == 0:
                dice = 1.0  # 두 집합이 모두 비어있다면 Dice는 1
            else:
                dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores[cls] += dice

        # 평균 Dice 계산
        mean_dice = dice_scores.mean().item()

        # 각 클래스별 Dice
        class_dice = {f'class_{cls}': dice_scores[cls].item() for cls in range(self.num_classes)}
        class_dice['mean_dice'] = mean_dice

        return class_dice

    def evaluate(self, size: int) -> dict:
        """
        Aggregates and computes the final metrics.

        Args:
            size (int): Total number of samples.

        Returns:
            dict: Final evaluation metrics.
        """
        return self.compute_metrics(None)
