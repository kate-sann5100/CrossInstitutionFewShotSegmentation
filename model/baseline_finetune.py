import numpy as np
import torch
from monai.losses import DiceLoss

from monai.networks.nets import BasicUnet
from torch import nn


class Pretrain(nn.Module):

    def __init__(self, args):
        super(Pretrain, self).__init__()

        # initialise backbone
        features = np.array([1, 1, 2, 4, 8, 1]) * args.feature
        basic_unet = BasicUnet(features=features, out_channels=9, dropout=0.2)
        self.conv_0 = basic_unet.conv_0
        self.down_1 = basic_unet.down_1
        self.down_2 = basic_unet.down_2
        self.down_3 = basic_unet.down_3
        self.down_4 = basic_unet.down_4
        self.upcat_4 = basic_unet.upcat_4
        self.upcat_3 = basic_unet.upcat_3
        self.upcat_2 = basic_unet.upcat_2
        self.upcat_1 = basic_unet.upcat_1

        self.seg = basic_unet.final_conv

        # initialise loss
        self.loss_fn = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
        )

    def forward_backbone(self, x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)  # 64
        u3 = self.upcat_3(u4, x2)  # 32
        u2 = self.upcat_2(u3, x1)  # 16
        u1 = self.upcat_1(u2, x0)  # 16

        return u1

    def forward(self, x):
        """
        :param x:
        "t2w": (B, 1, ...)
        "mask": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        """
        f = self.forward_backbone(x["t2w"])
        pred = self.seg(f)  # (B, 9, ...)
        if self.training:
            # (B, 9, W, H, D), (B, 1, W, H, D)
            loss = self.get_loss(pred, x["seg"])
            return {"label": loss}
        else:
            binary = torch.argmax(pred, dim=1, keepdim=True)  # (B, 1, W, H, D)
            return {"seg": binary}

    def get_loss(self, pred, y):
        """
        :param pred: (B, 2, W, H, D) or (B, 9, W, H, D)
        :param y: (B, 1, W, H, D)
        :return:
        """
        loss_fn = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
        )
        # mask: (B, 1, W, H, D)
        loss = loss_fn(pred, y)

        return loss


class FineTune(nn.Module):

    def __init__(self, args):
        super(FineTune, self).__init__()

        # initialise backbone
        features = np.array([1, 1, 2, 4, 8, 1]) * args.feature
        basic_unet = BasicUnet(features=features, out_channels=2, dropout=0.2)
        self.conv_0 = basic_unet.conv_0
        self.down_1 = basic_unet.down_1
        self.down_2 = basic_unet.down_2
        self.down_3 = basic_unet.down_3
        self.down_4 = basic_unet.down_4
        self.upcat_4 = basic_unet.upcat_4
        self.upcat_3 = basic_unet.upcat_3
        self.upcat_2 = basic_unet.upcat_2
        self.upcat_1 = basic_unet.upcat_1

        self.seg = basic_unet.final_conv

        # initialise loss
        self.loss_fn = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
        )

    def forward_backbone(self, x):
        x0 = self.conv_0(x)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)

        u4 = self.upcat_4(x4, x3)  # 64
        u3 = self.upcat_3(u4, x2)  # 32
        u2 = self.upcat_2(u3, x1)  # 16
        u1 = self.upcat_1(u2, x0)  # 16

        return u1

    def forward(self, x):
        """
        :param x:
        "t2w": (B, 1, ...)
        "mask": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        """
        f = self.forward_backbone(x["t2w"])
        pred = self.seg(f)  # (B, 9, ...)
        if self.training:
            # (B, 9, W, H, D), (B, 1, W, H, D)
            loss = self.get_loss(pred, x["mask"])
            return {"label": loss}
        else:
            binary = torch.argmax(pred, dim=1, keepdim=True)  # (B, 1, W, H, D)
            return {"mask": binary}

    def get_loss(self, pred, y):
        """
        :param pred: (B, 2, W, H, D) or (B, 9, W, H, D)
        :param y: (B, 1, W, H, D)
        :return:
        """
        loss_fn = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
        )
        # mask: (B, 1, W, H, D)
        loss = loss_fn(pred, y)

        return loss