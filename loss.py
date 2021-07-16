import torch
from torch import nn
import eval.beat_evaluation_toolbox as eval_tool
from utils import y2time


class ChamferLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, y):
        y = torch.cat([yi.view(-1, *yi.shape) for yi in y])
        y = y.repeat(1, 1, y.shape[-1])
        y = y.view(-1, y.shape[-1], y.shape[-1])
        outputs = outputs.repeat(1, 1, outputs.shape[-1])
        outputs = outputs.view(-1, outputs.shape[-1], outputs.shape[-1])
        outputs = outputs.t()
        left_dist = torch.argmax(torch.abs(y - outputs), axis=-1)
        right_dist = torch.argmax(torch.abs(y - outputs), axis=-2)
        dist = left_dist + right_dist
        return dist


class YoloLoss(nn.Module):

    def __init__(self, lambda_loc=1.0, lambda_beat=1.0, lambda_nobeat=1.0):
        super().__init__()
        self.lambda_loc = lambda_loc
        self.lambda_beat = lambda_beat
        self.lambda_nobeat = lambda_nobeat
        self.loss_fn = nn.L1Loss()

    def forward(self, outputs, y):
        y = torch.cat([yi.view(-1, *yi.shape) for yi in y])
        mask_beat = (y[..., 1] != 0).squeeze()
        mask_nobeat = (y[..., 1] == 0).squeeze()
        loss_loc = self.loss_fn(mask_beat * outputs[..., 0], mask_beat * y[..., 0])
        loss_beat_conf = self.loss_fn(mask_beat * outputs[..., 1], mask_beat * y[..., 1])
        loss_nobeat_conf = self.loss_fn(mask_nobeat * outputs[..., 1], mask_nobeat * y[..., 1])
        return self.lambda_loc * loss_loc + self.lambda_beat * loss_beat_conf + self.lambda_nobeat * loss_nobeat_conf
