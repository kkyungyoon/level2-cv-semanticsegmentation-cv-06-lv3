import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import pandas as pd

from src.models.smp_model import SmpModel
from src.utils.data_utils import load_yaml_config
from src.utils.constants import IND2CLASS

class SmpModule(pl.LightningModule):
    def __init__(self, train_config_path, model_config_path):
        super().__init__()
        self.train_config = load_yaml_config(train_config_path)
        self.model = SmpModel(model_config_path=model_config_path)
        self.validation_outputs = []
        self.rles = []
        self.filename_and_class = []

    def forward(self, images, labels=None):
        return self.model(images, labels)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs, loss = self.model(images, labels)

        # log
        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs, loss = self.model(images, labels)

        # log
        self.log('val_loss', loss, prog_bar=True)

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()  # threshold 0.5 적용
        labels = labels.float()

        dices = self.dice_coef(outputs, labels)

        # validation outputs에 추가
        self.validation_outputs.append(dices)
        
        return dices
    
    def test_step(self, batch, batch_idx):
        images, image_names = batch
        outputs = self.model(images)

        outputs = F.interpolate(
            outputs, 
            size=(2048, 2048),
            mode="bilinear"         
            )

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).detach().cpu().numpy()

        for output, image_name in zip(outputs, image_names):
            for c, segm in enumerate(output):
                rle = self.encode_mask_to_rle(segm)
                self.rles.append(rle)
                self.filename_and_class.append(
                    f"{IND2CLASS[c]}_{image_name}"
                )

        return
    
    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            return  # validation_outputs가 비어 있으면 종료

        # 전체 dices를 하나로 합침
        all_dices = torch.cat(self.validation_outputs, dim=0)  

        # 클래스별 Dice score 평균 계산
        dices_per_class = all_dices.mean(dim=0)  

        # 전체 평균 Dice score 계산
        avg_dice = dices_per_class.mean().item()

        # 각 클래스에 대한 Dice score 기록
        for i, dice in enumerate(dices_per_class):
            self.log(f'class_{i}_dice', dice.item(), on_epoch=True, prog_bar=False)

        # 전체 평균 Dice 기록
        self.log('avg_dice_score', avg_dice, on_epoch=True, prog_bar=True)

        # validation outputs 초기화
        self.validation_outputs = []

        # Return avg dice score for early stopping or logging
        return avg_dice
    
    def on_test_epoch_end(self):
        classes, filename = zip(*[x.split("_") for x in self.filename_and_class])

        image_name = [os.path.basename(f) for f in filename]

        df = pd.DataFrame({
            "image_name": image_name,
            "class": classes,
            "rle": self.rles,
        })

        df.to_csv(f"./logs/{self.train_config['logger']['name']}.csv", index=False)

        return df

    
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
    def dice_coef(y_true, y_pred, eps=1e-4):
        y_true_f = y_true.flatten(2)
        y_pred_f = y_pred.flatten(2)
        intersection = torch.sum(y_true_f * y_pred_f, -1)

        return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)
    
    @staticmethod
    def encode_mask_to_rle(mask):
        '''
        mask: numpy array binary mask
        1 - mask
        0 - background
        Returns encoded run length
        '''
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)