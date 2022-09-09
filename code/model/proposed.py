from .unet_model import UNet
import torch
import torch.nn as nn


def outer_concatenation(x, y):
    y_expand = y.unsqueeze(1)
    x_expand = x.unsqueeze(2)
    y_repeat = y_expand.repeat(1, y_expand.shape[2], 1, 1)
    x_repeat = x_expand.repeat(1, 1, x_expand.shape[1], 1)
    outer_concat = torch.cat((x_repeat, y_repeat), 3)
    return outer_concat.permute(0, 3, 1, 2).contiguous()


class ProposedModel(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        self.unet = UNet(n_channels=input_ch*2, n_classes=1)

    def forward(self, data1, data2):
        pred_path = self.unet(outer_concatenation(data1, data2))
        return pred_path.squeeze(1)
