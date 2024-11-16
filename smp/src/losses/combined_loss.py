import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class CombinedLoss(nn.Module):
    def __init__(self, losses, weight):
        super(CombinedLoss, self).__init__()
        
        # losses: DiceLoss, BCEWithLogitsLoss 등의 리스트
        # weight: 각 손실 함수에 대한 가중치
        self.losses = nn.ModuleList()
        self.weights = list(weight.values())
        
        for loss in losses:
            loss_name = loss["name"]
            loss_args = loss.get("args", {})
            
            if loss_name == "JaccardLoss":
                self.losses.append(smp.losses.JaccardLoss(**loss_args))
            elif loss_name == "BCEWithLogitsLoss":
                self.losses.append(nn.BCEWithLogitsLoss(**loss_args))
            elif loss_name == "DiceLoss":
                self.losses.append(smp.losses.DiceLoss(**loss_args))
            elif loss_name == "TverskyLoss":
                self.losses.append(smp.losses.TverskyLoss(**loss_args))
            elif loss_name == "FocalLoss":
                self.losses.append(smp.losses.FocalLoss(**loss_args))
            elif loss_name == "LovaszLoss":
                self.losses.append(smp.losses.LovaszLoss(**loss_args))
            elif loss_name == "SoftBCEWithLogitsLoss":
                self.losses.append(smp.losses.SoftBCEWithLogitsLoss(**loss_args))
            else:
                raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def forward(self, outputs, targets):
        total_loss = 0
        for idx, loss_fn in enumerate(self.losses):
            total_loss += self.weights[idx] * loss_fn(outputs, targets)
        return total_loss
