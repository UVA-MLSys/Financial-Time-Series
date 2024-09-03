import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.ditill_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}

class DistillationLoss(nn.Module):
    def __init__(self, distill_loss, logits_loss, task_loss, task_name, feature_w=0.01, logits_w=1.0, task_w=1.0, features='S', pred_len=1):
        super(DistillationLoss, self).__init__()
        
        # logits_w = 0 
        # feature_w= 0 
        self.task_w = task_w
        self.logits_w = logits_w
        self.feature_w = feature_w
        
        # weight 1.0      1.0         0.01
        print('Loss function weights:' , task_w , logits_w , feature_w )
        
        self.feature_loss = loss_dict[distill_loss]
        self.logits_loss = loss_dict[logits_loss]
        self.task_loss = loss_dict[task_loss]
        
        self.task_name = task_name
        self.f_dim = -1 if features == 'MS' else 0
        self.pred_len = pred_len

    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        """
        outputs_time: 隐藏层特征经过残差连接+任务head之后的结果
        intermidiate_feat_time: 大小为num_blk+1, 包含最初的输入特征，最后一个元素是没有经过残差和head的特征。
        """
        outputs_text, outputs_time, intermidiate_feat_time, intermidiate_feat_text = (
            outputs["outputs_text"],
            outputs["outputs_time"],
            outputs["intermidiate_time"],
            outputs["intermidiate_text"],
        )
        
        # 1-----------------中间特征损失
        if intermidiate_feat_time is not None :
            feature_loss = sum(
                [
                    (0.8**idx) * self.feature_loss(feat_time, feat_text)
                    for idx, (feat_time, feat_text) in enumerate(
                        zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])
                    )
                ]
            )
        else:
            feature_loss = 0 
            
        batch_y = batch_y[:, -self.pred_len:, self.f_dim:]
        outputs_time = outputs_time[:, -self.pred_len:, self.f_dim:]
        outputs_text = outputs_text[:, -self.pred_len:, self.f_dim:]

        # 2----------------输出层的教师-学生损失
        logits_loss = self.logits_loss(outputs_time, outputs_text)
            
        # 3----------------任务特定的标签损失
        batch_y = batch_y.to(outputs_time.device)
        
        task_loss = self.task_loss(outputs_time, batch_y)
        total_loss = self.task_w * task_loss + self.logits_w * logits_loss + self.feature_w * feature_loss
        return total_loss
