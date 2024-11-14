import torch
import pytorch_lightning as pl

from src.models.smp_model import SmpModel
from src.utils.data_utils import load_yaml_config

class SmpModule(pl.LightningModule):
    def __init__(self, train_config_path, model_config_path):
        super().__init__()
        self.train_config = load_yaml_config(train_config_path)
        self.model = SmpModel(model_config_path=model_config_path)

    def forward(self, images, labels=None):
        return self.model(images, labels)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs, loss = self.model(images, labels)

        # log
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs, loss = self.model(images, labels)

        # log
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()  # threshold 0.5 적용
        labels = labels.float()

        dics = self.dice_coef(outputs, labels)

        return dics
    
    def on_validation_epoch_end(self, outputs):
        # validation_step에서 계산된 dice scores를 모아서 평균값 계산
        all_dices = torch.stack(outputs)

        # 클래스별 Dice score 평균 계산
        avg_dices_per_class = all_dices.mean(dim=0)

        # 전체 평균 Dice score 계산
        avg_dice = avg_dices_per_class.mean().item()

        # 각 클래스에 대한 Dice score 기록
        for i, dice in enumerate(avg_dices_per_class):
            self.log(f'class_{i}_dice', dice.item(), on_epoch=True, prog_bar=True)

        # 전체 평균 Dice 기록
        self.log('avg_dice_score', avg_dice, on_epoch=True, prog_bar=True)

        # Return avg dice score for early stopping or logging
        return avg_dice
    
    def configure_optimizers(self):
        # lr_backbone을 float로 변환
        lr_backbone = float(self.train_config["optimizer"]["lr_backbone"])

        # optimizer params 내 모든 값들 중 'lr'을 float로 변환
        optimizer_params = self.train_config["optimizer"]["params"]
        
        # optimizer params 내에서 모든 str 값을 float으로 변환
        for key, value in optimizer_params.items():
            if isinstance(value, str):  # 값이 문자열이면 float로 변환
                optimizer_params[key] = float(value)

        # param_dicts 정의
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": lr_backbone,
            },
        ]

        # Optimizer class
        optimizer_class = getattr(torch.optim, self.train_config["optimizer"]["name"])
        
        # optimizer 생성 시, params가 제대로 float로 설정되어 있는지 확인
        optimizer = optimizer_class(param_dicts, **optimizer_params)

        # Scheduler 설정 확인
        if "scheduler" in self.train_config:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.train_config["scheduler"]["name"])
            scheduler = scheduler_class(optimizer, **self.train_config["scheduler"]["params"])
            return [optimizer], [scheduler]
        else:
            return optimizer

        
    
    @staticmethod
    def dice_coef(outputs, targets, eps=1e-4):
        """
        Dice coefficient 계산
        :param outputs: 모델의 예측 (sigmoid를 거친 후 thresholding된 tensor)
        :param targets: 실제 마스크 (ground truth)
        :param eps: 작은 값으로 나누기 방지
        :return: dice coefficient
        """
        intersection = (outputs * targets).sum(dim=(1, 2, 3))
        union = outputs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return (2. * intersection + eps) / (union + eps)