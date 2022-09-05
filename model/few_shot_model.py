import numpy as np
import torch
from monai.losses import MaskedDiceLoss
from monai.networks import one_hot
from monai.networks.blocks import Warp

from monai.networks.nets import BasicUnet, GlobalNet
from torch import nn
import torch.nn.functional as F


class LSNet(nn.Module):

    def __init__(self, args, align_head=False):
        super(LSNet, self).__init__()

        self.args = args
        self.shot = args.shot

        # initialise bin
        self.f_size = args.f_size
        self.stride, self.pool_weight = self.init_bins(args)

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

        # initialise align mechanism
        if args.align:
            self.seg = basic_unet.final_conv
            if align_head:
                self.align_head = GlobalNet(
                    image_size=args.size,
                    spatial_dims=3,
                    num_channel_initial=32,
                    depth=3,
                    in_channels=2
                )
            else:
                self.align_head = None
        else:
            self.seg = None
            self.align_head = None

        # initialise prior head
        if args.con:
            self.con = nn.Sequential(
                nn.Conv3d(3, args.feature, kernel_size=3),
                nn.ReLU(),
                nn.Conv3d(args.feature, 2, kernel_size=3),
            )
        else:
            self.con = None

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

    def init_bins(self, args):
        """
        :param args:
        :return:
            stride: (3)
            pool_weight: (in_ch, 1, k, k, 1)
        """
        kernel_size = int(args.f_size[0] * args.alpha)
        assert kernel_size % 2 == 0, f"feature size {args.f_size} and alpha {args.alpha} does not fit"
        stride = int(kernel_size / 2)
        stride = (stride, stride, 1)

        pool_weight = torch.ones(
            (args.feature, 1, kernel_size, kernel_size, 1)
        )
        return stride, pool_weight

    def get_prototype(self, f, y):
        """
        :param f: (B, C, W, H, D)
        :param y: (B, 1, W, H, D)
        :return:
            p_f: (B, C, W, W, W, 4)
            area: (B, 1, W, W, W, 4)
        """
        p_f = F.conv3d(
            f * y,  # (B, C, W, H, D)
            self.pool_weight.to(f),
            stride=self.stride,
            groups=f.shape[1]
        )  # (B, C, w, h, D)
        area = F.conv3d(
            y,
            self.pool_weight.to(f)[:1],
            stride=self.stride,
        )  # (B, 1, w, h, D)
        p_f = p_f / (area + 1e-7)  # (B, C, w, h, D)

        def tile(x):
            stride = self.stride[0]
            b, ch, w, h, d = x.shape
            x = x.reshape(b, ch, w, 1, h, 1, d)
            x = x.expand(b, ch, w, stride, h, stride, d)
            x = x.reshape(b, ch, w * stride, h * stride, d)  # (B, C, W-S, H-S, d)

            x = F.pad(x, [0, 0, stride, stride, stride, stride])  # (B, C, W+S, H+S, d)
            x = [
                x[:, :, stride * _w: stride * (w + _w + 1), stride * _h: stride * (h + _h + 1), :]
                for _w in range(2) for _h in range(2)
            ]  # 4 x (B, C, W, H, d)

            x = torch.stack(x, dim=-1)  # (B, C, W, H, d, 4)
            return x

        p_f = tile(p_f)  # (B, C, W, H, D, 4)
        return p_f

    @staticmethod
    def get_similarity(q, s):
        """
        :param q: (B, C, W, H, D)
        :param s: (B, C, W, H, D, 4) or (B, C, W, H, D, 4S)
        :return:
            similarity: (B, W, H, D)
        """
        similarity = F.cosine_similarity(
            q.unsqueeze(-1),  # (B, C, H, W, D, 1)
            s,  # (B, C, W, H, D, 4)
            dim=1
        )  # (B, W, H, D, 4)
        similarity = torch.amax(similarity, dim=-1)  # (B, W, H, D)
        return similarity

    def average_prototype_over_shot(self, p, w=None):
        """
        :param p: (BxS, C, W, H, D, 4)
        :param w: (BxS)
        :return: p: (B, C, W, H, D, 4)
        """
        # (BxS, C, W, H, D, 4) -> (B, S, C, W, H, D, 4)
        p = p.reshape(-1, self.shot, *p.shape[1:])
        p_count = torch.sum(p, dim=2, keepdim=True)  # (B, S, 1, W, H, D, 4)
        if w is not None:
            w = w.reshape(-1, self.shot)  # (B, S)
            w = one_hot(
                torch.argmax(w, dim=1, keepdim=True), num_classes=self.shot
            )  # (B, S)
            p = p * w.reshape(-1, self.shot, 1, 1, 1, 1, 1)
            p_count = p_count * w.reshape(-1, self.shot, 1, 1, 1, 1, 1)
        # (B, S, C, W, H, D, 4) -> (B, C, W, H, D, 4)
        p_sum, p_count_sum = torch.sum(p, dim=1), torch.sum(p_count, dim=1)
        p_count_sum[p_count_sum == 0] = -1
        p = p_sum / p_count_sum
        return p

    def forward(self, query, support):
        """
        :param query:
        "t2w": (B, 1, ...)
        "mask": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        :param support:
        "t2w": (B, 1, ...) or (B, S, 1, ...)
        "mask": (B, 1, ...) or (B, S, 1, ...)
        "seg": (B, 1, ...) or (B, S, 1, ...)
        "name": str
        :return:
        """
        # (B, S, 1, ...) -> (BxS, 1, ...)
        if len(support["t2w"].shape) > 5:
            support["t2w"] = support["t2w"].reshape(-1, 1, *self.args.size)  # (BxS, 1, ...)
            support["mask"] = support["mask"].reshape(-1, 1, *self.args.size)  # (BxS, 1, ...)

        q_f = self.forward_backbone(query["t2w"])
        s_f = self.forward_backbone(support["t2w"])

        # segmentation head
        q_pred, s_pred = None, None
        s_mask = support["mask"]
        if self.seg is not None:
            q_pred = self.seg(q_f)  # (B, 9, W, H, D)
            s_pred = self.seg(s_f)  # (BxS, 9, W, H, D)
            if self.align_head is not None:
                if self.shot > 1:
                    # (B, 9, W, H, D) -> (BxS, 9, W, H, D)
                    q_pred = q_pred.repeat(self.shot, 1, 1, 1, 1)  # (BxS, 1, W, H, D)
                s_pred = torch.argmax(s_pred, dim=1, keepdim=True)  # (B, 1, W, H, D) or (BxS, 1, W, H, D)
                q_pred = torch.argmax(q_pred, dim=1, keepdim=True)  # (B, 1, W, H, D) or (BxS, 1, W, H, D)
                s_pred_original = s_pred.clone()
                ddf = self.align_head(
                    torch.cat([s_pred, q_pred], 1).to(torch.float32)  # (B, 2, W, H, D) or (BxS, 2, W, H, D)
                )  # (B, 3, W, H, D) or (BxS, 3, W, H, D)
                s_mask = Warp(mode="nearest")(support["mask"].to(ddf), ddf)  # (B, 1, W, H, D) or (BxS, 1, W, H, D)
                s_f = Warp(mode="nearest")(s_f.to(ddf), ddf)  # (B, C, W, H, D) or (BxS, C, W, H, D)
                s_pred = Warp(mode="nearest")(s_pred.to(ddf), ddf)  # (B, 1, W, H, D) or (BxS, 1, W, H, D)

        if self.shot > 1:
            # (BxS, 1, ...) -> (B, S, 1, ...)
            s_mask = s_mask.reshape(-1, self.shot, *s_mask.shape[1:])
            if self.seg is not None:
                b = q_f.shape[0]
                q_pred = q_pred.reshape(b, self.shot, -1)  # (B, 1, WxHxD)
                s_pred = s_pred.reshape(b, self.shot, -1)  # (B, S, WxHxD)
                w = F.cosine_similarity(q_pred, s_pred, dim=-1)  # (B, S)
                # (B, S, WxHxD) -> (B, S) -> S -> 1
                idx = torch.argmax(w.squeeze())
                s_mask = s_mask[:, idx, ...]  # (B, 1, ...)
                # w = F.cosine_similarity(
                #     q_f.repeat(self.shot, 1, 1, 1, 1).reshape(b*self.shot, -1),  # (BxS, CxWxHxD)
                #     s_f.reshape(b*self.shot, -1)  # (BxS, CxWxHxD)
                # )  # (BxS)
            else:
                s_mask = s_mask[:, 0, ...]  # (B, 1, ...)

        # resize to save computation cost
        q_f = F.interpolate(q_f, size=self.f_size, mode="trilinear", align_corners=True)  # (B, C, W, H, D)
        s_f = F.interpolate(s_f, size=self.f_size, mode="trilinear", align_corners=True)
        s_y = F.interpolate(s_mask.float(), size=self.f_size, mode="trilinear", align_corners=True)

        # get prototypes
        p_fg = self.get_prototype(s_f, s_y)  # (B, C, W, H, D, 4) or (BxS, C, W, H, D, 4)
        p_fg_count = torch.sum(p_fg, dim=1, keepdim=True) > 0  # (B, 1, W, H, D, 4) or (BxS, 1, W, H, D, 4)

        p_bg = self.get_prototype(s_f, torch.ones_like(s_y) - s_y)  # (B, C, W, H, D, 4) or (BxS, C, W, H, D, 4)

        if self.shot > 1:
            p_fg = self.average_prototype_over_shot(p_fg, w)
            p_bg = self.average_prototype_over_shot(p_bg, w)

            # def permute_shot(p):
            #     # (BxS, 1, W, H, D, 4) -> (B, S, 1, W, H, D, 4) -> (B, 1, W, H, D, 4, S)
            #     p = p.reshape(-1, self.shot, *p.shape[1:]).permute(0, 2, 3, 4, 5, 6, 1)
            #     # (B, 1, W, H, D, 4, S) -> (B, 1, W, H, D, 4S)
            #     p = p.reshape(*p.shape[:-2], -1)
            #     return p
            # p_fg = permute_shot(p_fg)
            # p_bg = permute_shot(p_bg)

            # (BxS, 1, W, H, D, 4) -> (B, S, 1, W, H, D, 4) -> (B, 1, W, H, D, 4)
            p_fg_count = p_fg_count.reshape(-1, self.shot, *p_fg_count.shape[1:]).sum(dim=1)

        p_fg_count = torch.sum(
            p_fg_count.to(torch.float), dim=-1
        )  # (B, 1, W, H, D)

        # cosine similarity
        fg_score = self.get_similarity(q_f, p_fg)  # (B, W, H, D)
        bg_score = self.get_similarity(q_f, p_bg)  # (B, W, H, D)
        pred = torch.stack([bg_score, fg_score], dim=1)  # (B, 2, W, H, D)

        # prior head
        if self.con is not None:
            pred = torch.cat([
                pred, F.interpolate(s_mask, size=pred.shape[-3:])
            ], dim=1)
            pred = self.con(pred)

        # resize
        pred = F.interpolate(
            pred, size=query["mask"].shape[-3:], mode="trilinear", align_corners=True
        )  # (B, 2, W, H, D)
        pred[:, 0, ...][pred[:, 1, ...] == 0] = 0
        p_fg_count = F.interpolate(
            p_fg_count, size=query["mask"].shape[-3:], mode="nearest"
        )

        if self.training:
            # mask out inactive grids
            few_shot_loss_mask = p_fg_count > 0  # (B, 1, W, H, D)
            # few shot loss
            few_shot_loss = self.get_loss(pred, query["mask"].long(), few_shot_loss_mask)
            # segmentation loss
            if self.seg is not None:
                seg_loss = self.get_loss(q_pred, query["seg"]) + self.get_loss(s_pred, support["seg"])
            else:
                seg_loss = torch.zeros_like(few_shot_loss)

            # return loss_dict
            loss_dict = {
                "few_shot": few_shot_loss,
                "label": seg_loss,
            }
            return loss_dict
        else:
            binary = {"mask": torch.argmax(pred, dim=1, keepdim=True)}  # (B, 1, W, H, D)
            if self.seg:
                binary["q_seg"] = q_pred  # (B, 9, W, H, D)
                binary["s_seg"] = s_pred  # (B, 9, W, H, D)
                if self.align_head is not None:
                    binary["aligned_s_seg_gt"] = Warp(mode="nearest")(support["seg"].to(ddf), ddf)  # (B, 1, W, H, D)
                    binary["aligned_s_mask"] = s_mask
                    binary["aligned_s_t2w"] = Warp()(support["t2w"].to(ddf), ddf)  # (B, 1, W, H, D)
            return binary

    def get_loss(self, pred, y, mask=None):
        """
        :param pred: (B, 2, W, H, D) or (B, 9, W, H, D)
        :param y: (B, 1, W, H, D)
        :return:
        """
        loss_fn = MaskedDiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
        )
        # mask: (B, 1, W, H, D)
        loss = loss_fn(pred, y, mask=mask) if mask is not None else loss_fn(pred, y)
        return loss


