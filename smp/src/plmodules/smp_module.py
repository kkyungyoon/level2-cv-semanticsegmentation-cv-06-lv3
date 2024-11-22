import os
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime

import torch.nn as nn

from skimage.color import gray2rgb
from skimage.color import rgb2gray

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

from src.models.smp_model import SmpModel
from src.utils.data_utils import load_yaml_config
from src.utils.constants import IND2CLASS

class SmpModule(pl.LightningModule):
    def __init__(self, train_config_path, model_config_path, use_crf=False):
        super().__init__()
        self.train_config = load_yaml_config(train_config_path)
        self.model_config = load_yaml_config(model_config_path)
        self.model = SmpModel(model_config_path=model_config_path)
        self.mode = self.model_config['interpolate']['mode']
        self.sliding_window = self.model_config.get("sliding_window", False)

        if self.sliding_window:
            self.stride = self.model_config["sliding_window"]["stride"]
            self.patch_size = self.model_config["sliding_window"]["patch_size"]

        self.use_crf = use_crf
        self.validation_outputs = []
        self.rles = []
        self.filename_and_class = []

        self.model = self.replace_batchnorm_with_groupnorm(model=self.model)

    def forward(self, images, labels=None):
        return self.model(images, labels)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs, loss = self.model(images, labels)

        # log
        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):

        if not self.sliding_window:
            images, labels = batch
            outputs, loss = self.model(images, labels)

        else:
            outputs, loss = self._sliding_window_step(batch)

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
        if not self.sliding_window:
            images, image_names = batch
            outputs = self.model(images)

            outputs = F.interpolate(
                outputs, 
                size=(2048, 2048),
                mode=self.mode        
                )
        else:
            outputs = self._sliding_window_step(batch)

        outputs = torch.sigmoid(outputs)

        if self.use_crf:
            outputs = self.crf(mask_img=outputs)

        outputs = (outputs > 0.5).detach().cpu().numpy()

        for output, image_name in zip(outputs, image_names):
            for c, segm in enumerate(output):
                rle = self.encode_mask_to_rle(segm)
                self.rles.append(rle)
                self.filename_and_class.append(
                    f"{IND2CLASS[c]}_{image_name}"
                )

        return
    
    def _sliding_window_step(self, batch, batch_idx):
        images, labels = batch

        # sliding window inference
        stride = self.stride
        patch_size = self.patch_size
        batch_patches = []
        batch_coords = []
        batch_size = self.batch_size

        _, _, H, W = images.shape  # 입력 이미지 크기 (Batch, Channels, Height, Width)
        outputs_full = torch.zeros((images.size(0), self.num_classes, H, W)).to(images.device)

        # 슬라이딩 윈도우로 이미지 나누기
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch = images[:, :, i:i + patch_size, j:j + patch_size]
                batch_patches.append(patch)
                batch_coords.append((i, j))

                # 배치 크기에 도달하면 모델 추론
                if len(batch_patches) == batch_size:
                    batch_patches_tensor = torch.cat(batch_patches, dim=0)
                    batch_outputs = self.model(batch_patches_tensor)  # 모델 예측
                    for k, (x, y) in enumerate(batch_coords):
                        outputs_full[:, :, x:x + patch_size, y:y + patch_size] += batch_outputs[k]
                    batch_patches = []
                    batch_coords = []

        # 남은 패치에 대해 추론
        if batch_patches:
            batch_patches_tensor = torch.cat(batch_patches, dim=0)
            batch_outputs = self.model(batch_patches_tensor)
            for k, (x, y) in enumerate(batch_coords):
                outputs_full[:, :, x:x + patch_size, y:y + patch_size] += batch_outputs[k]

        # 패치 합성 후 평균값으로 정규화
        norm_map = torch.zeros_like(outputs_full)
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                norm_map[:, :, i:i + patch_size, j:j + patch_size] += 1
        outputs_full /= norm_map

        # 레이블이 주어지면 손실 계산
        if labels is not None:
            loss = self.criterion(outputs_full, labels)
            return outputs_full, loss

        return outputs_full

    
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
            self.log(f'{IND2CLASS[i]}_dice', dice.item(), on_epoch=True, prog_bar=False)

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

        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"./logs/{self.train_config['logger']['name']}_{current_time}.csv", index=False)

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

    def replace_batchnorm_with_groupnorm(self, model, default_num_groups=32):
        """
        모델 내 모든 BatchNorm2d를 GroupNorm으로 변환. 
        num_channels에 따라 num_groups를 자동 조정.
        
        Args:
            model (nn.Module): 변환할 모델.
            default_num_groups (int): 기본 그룹 수. 채널 수에 맞게 자동 조정.
        """
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                num_channels = module.num_features
                # num_channels에 따라 num_groups를 자동으로 설정
                num_groups = min(default_num_groups, num_channels)
                if num_channels % num_groups != 0:  # 나누어떨어지지 않으면 그룹 수 조정
                    num_groups = 1
                group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
                setattr(model, name, group_norm)
            else:
                # 하위 모듈이 있을 경우 재귀적으로 교체
                self.replace_batchnorm_with_groupnorm(module)
        return model

    
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
    

    @staticmethod
    def crf(image_size=(2048, 2048), mask_img=None):
        
        # Converting annotated image to RGB if it is Gray scale
        if(len(mask_img.shape)<3):
            mask_img = gray2rgb(mask_img)

        # Converting the annotations RGB color to single 32 bit integer
        annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)
        
        # Convert the 32bit integer color to 0,1, 2, ... labels.
        colors, labels = np.unique(annotated_label, return_inverse=True)

        n_labels = 29
        
        # Setting up the CRF model
        d = dcrf.DenseCRF2D(image_size[1], image_size[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
            
        #Run Inference for 10 steps 
        Q = d.inference(10)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)

        return MAP.reshape((image_size[0], image_size[1]))