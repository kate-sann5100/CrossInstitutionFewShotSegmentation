import torch
from monai.losses import DiceLoss
from monai.networks import one_hot

from torch import nn

from monai.networks.blocks import Warp
from monai.networks.nets import GlobalNet


class RegistrationModel(nn.Module):

    def __init__(self, args):
        super(RegistrationModel, self).__init__()
        self.model = GlobalNet(
            image_size=args.size,
            spatial_dims=3,
            num_channel_initial=32,
            depth=3,
            in_channels=2
        )

        self.label_loss = DiceLoss(
                include_background=True,
                softmax=True,
                squared_pred=True,
            )

    def forward(self, moving, fixed):
        """
        :param moving:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        :param fixed:
        "t2w": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        :return:
        """
        moving_seg, fixed_seg = moving["seg"], fixed["seg"]

        ddf = self.model(
            torch.cat([moving_seg, fixed_seg], 1)  # (B, 2, ...)
        )  # (B, 3, ...)

        if self.training:
            # moving_seg: (B, 1, ...), ddf: (B, 3, ...)
            pred = Warp()(
                one_hot(moving_seg, num_classes=9), ddf
            )  # (B, 9, ...)
            # pred: (B, 9, ...) gt: (B, 1, ...)
            label_loss = self.label_loss(
                pred, one_hot(fixed_seg, num_classes=9)
            )
            loss_dict = {"label": label_loss}
            return loss_dict
        else:
            # moving_seg: (B, 1, ...), ddf: (B, 3, ...)
            seg_binary = Warp(mode="nearest")(moving["seg"], ddf)  # (B, 9, ...)
            mask_binary = Warp(mode="nearest")(moving["mask"], ddf)
            binary = {
                "seg": seg_binary,
                "mask": mask_binary
            }
            return binary