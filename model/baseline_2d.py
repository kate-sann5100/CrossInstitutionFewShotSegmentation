import torch
from monai.losses import MaskedDiceLoss
from torch import nn
from torch.nn import functional as F

from model.resnet import resnet50


class LSNet2D(nn.Module):

    def __init__(self, args):
        super(LSNet2D, self).__init__()

        self.input_size = args.size

        # initialise bin
        self.stride, self.pool_weight = self.init_bins(args)

        # initialise loss
        self.seg_loss = args.seg_loss

        # initialise backbone
        resnet = resnet50(pretrained=True)

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.ReLU(),
            nn.Dropout3d(p=0.5),
        )

        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu1,
            resnet.conv2, resnet.bn2, resnet.relu2,
            resnet.conv3, resnet.bn3, resnet.relu3,
            resnet.maxpool
        )
        self.layer2, self.layer3, self.layer4, self.layer5 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer5.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def depth_to_batch(self, input):
        """
        :param input: dict including
        "t2w": (B, 1, W, H, D)
        "mask": (B, 1, W, H, D)
        "seg": (B, 1, W, H, D)
        "name": str
        :return: dict
        """
        for k in ["t2w", "mask"]:
            v = input[k]  # (B, 1, W, H, D)
            v = v.permute(0, 4, 1, 2, 3)  # (B, D, 1, W, H)
            v = v.reshape(-1, 1, *self.input_size[:2])  # (BxD, 1, W, H)
            input[k] = v
        return input

    def batch_to_depth(self, input):
        """
        :param input: (BxD, 1, W, H)
        """
        input = input.reshape(-1, 10, 1, *self.input_size[:2])  # (B, D, 1, W, H)
        input = input.permute(0, 2, 3, 4, 1)  # (B, 1, W, H, D)
        return input

    def forward_backbone(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x

    def init_bins(self, args):
        """
        :param args:
        :return:
            stride: (3)
            pool_weight: (in_ch, 1, k, k)
        """
        kernel_size = int(args.size[0] / 8 * args.alpha)
        assert kernel_size % 2 == 0, f"input size {args.size} and alpha {args.alpha} does not fit"
        stride = int(kernel_size / 2)
        stride = (stride, stride)

        pool_weight = torch.ones(
            (2048, 1, kernel_size, kernel_size)
        )
        return stride, pool_weight

    def get_prototype(self, f, y):
        """
        :param f: (B, C, W, H)
        :param y: (B, 1, W, H)
        :return:
            p_f: (B, C, W, H, 4)
            area: (B, 1, W, H, 4)
        """
        p_f = F.conv2d(
            f * y,  # (B, C, W, H)
            self.pool_weight.to(f),
            stride=self.stride,
            groups=f.shape[1]
        )  # (B, C, w, h)
        area = F.conv2d(
            y,
            self.pool_weight.to(f)[:1],
            stride=self.stride,
        )  # (B, 1, w, h)
        p_f = p_f / (area + 1e-7)  # (B, C, w, h)

        def tile(x):
            stride = self.stride[0]
            b, ch, w, h = x.shape
            x = x.reshape(b, ch, w, 1, h, 1)
            x = x.expand(b, ch, w, stride, h, stride)
            x = x.reshape(b, ch, w * stride, h * stride)  # (B, C, W-S, H-S)

            x = F.pad(x, [stride, stride, stride, stride])  # (B, C, W+S, H+S)
            x = [
                x[:, :, stride * _w: stride * (w + _w + 1), stride * _h: stride * (h + _h + 1)]
                for _w in range(2) for _h in range(2)
            ]  # 4 x (B, C, W, H)

            x = torch.stack(x, dim=-1)  # (B, C, W, H, 4)
            return x

        p_f = tile(p_f)  # (B, C, W, H, 4)
        return p_f

    @staticmethod
    def get_similarity(q, s):
        """
        :param q: (B, C, W, H)
        :param s: (B, C, W, H, 4)
        :return:
            similarity: (B, W, H)
        """
        similarity = F.cosine_similarity(
            q.unsqueeze(-1),  # (B, C, H, W, 1)
            s,  # (B, C, W, H, 4)
            dim=1
        )  # (B, W, H, 4)
        similarity = torch.amax(similarity, dim=-1)  # (B, W, H)
        return similarity

    def forward(self, query, support):
        """
        :param query:
        "t2w": (B, 1, ...)
        "mask": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        :param support:
        "t2w": (B, 1, ...)
        "mask": (B, 1, ...)
        "seg": (B, 1, ...)
        "name": str
        :return:
        """
        query, support = self.depth_to_batch(query), self.depth_to_batch(support)

        q_f = self.forward_backbone(query["t2w"])
        s_f = self.forward_backbone(support["t2w"])

        s_mask = support["mask"]

        # resize to save computation cost
        s_y = F.interpolate(s_mask.float(), size=q_f.shape[-2:], mode="bilinear", align_corners=True)

        # get prototypes
        p_fg = self.get_prototype(s_f, s_y)  # (B, C, W, H, 4)
        p_fg_count = torch.sum(p_fg, dim=1, keepdim=True) > 0  # (BxD, 1, W, H, 4)
        p_fg_count = torch.sum(
            p_fg_count.to(torch.float), dim=-1
        )  # (BxD, 1, W, H)

        p_bg = self.get_prototype(s_f, torch.ones_like(s_y) - s_y)  # (BxD, C, W, H, 4)

        # cosine similarity
        fg_score = self.get_similarity(q_f, p_fg)  # (BxD, W, H)
        bg_score = self.get_similarity(q_f, p_bg)  # (BxD, W, H)
        pred = torch.stack([bg_score, fg_score], dim=1)  # (BxD, 2, W, H)

        # resize
        pred = F.interpolate(
            pred, size=query["mask"].shape[-2:], mode="bilinear", align_corners=True
        )  # (BxD, 2, W, H)
        pred[:, 0, ...][pred[:, 1, ...] == 0] = 0
        p_fg_count = F.interpolate(
            p_fg_count, size=query["mask"].shape[-2:], mode="nearest"
        )

        if self.training:
            # mask out inactive grids
            few_shot_loss_mask = p_fg_count > 0  # (BxD, 1, W, H)
            # few shot loss
            few_shot_loss = self.get_seg_loss(pred, query["mask"].long(), few_shot_loss_mask)
            # return loss_dict
            loss_dict = {"few_shot": few_shot_loss}
            return loss_dict
        else:
            binary = torch.argmax(pred, dim=1, keepdim=True)  # (BxD, 1, W, H)
            binary = {"mask": self.batch_to_depth(binary)}  # (B, 1, W, H, D)
            return binary

    def get_seg_loss(self, pred, y, mask=None):
        """
        :param pred: (B, 2, W, H) or (B, 9, W, H)
        :param y: (B, 1, W, H)
        :return:
        """
        if self.seg_loss == "ce":
            if mask is not None:
                y[mask == 0] = 255  # (B, 1, W, H)
            loss = F.cross_entropy(
                input=pred,  # (B, 2, W, H)
                target=y.squeeze(1).to(torch.long),  # (B, W, H)
                ignore_index=255
            )
        elif self.seg_loss == "dice":
            loss_fn = MaskedDiceLoss(
                include_background=True,
                to_onehot_y=True,
                softmax=True,
                squared_pred=True,
            )
            # mask: (B, 1, W, H)
            loss = loss_fn(pred, y, mask=mask) if mask is not None else loss_fn(pred, y)
        else:
            raise ValueError(f"unrecognised loss {self.seg_loss}")
        return loss
