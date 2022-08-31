import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks import one_hot

from dataset.dataset_utils import organ_list
from utils.train_eval_utils import trim_depth


class LossMeter:
    def __init__(self, writer):
        self.writer = writer
        self.sum_dict, self.count = None, None
        self.reset()

    def reset(self):
        self.sum_dict = {
            "total": 0,
            "label": 0,
            "reg": 0,
            "few_shot": 0,
        }
        self.count = 0

    def update(self, loss_dict):
        self.count += 1
        for k, v in loss_dict.items():
            self.sum_dict[k] += v

    def get_average(self, step):
        for k, v in self.sum_dict.items():
            self.writer.add_scalar(
                tag=k,
                scalar_value=v / self.count,
                global_step=step
            )

        self.reset()


class DiceMeter:
    def __init__(self, writer, few_shot=False, test=False):
        self.writer = writer
        self.labels = organ_list
        self.metric_fn = DiceMetric(
            include_background=False,
            reduction="sum_batch",
            ignore_empty=True
        )
        self.few_shot = few_shot
        self.tag = "few_shot_dice" if few_shot else "dice"

        self.test = test
        if test:
            self.result_dict = {
                cls: {}
                for cls in range(1, 9)
            }
        else:
            self.result_dict = None

        self.reset()

    def reset(self):
        self.count, self.sum = torch.zeros(8), torch.zeros(8)

    def one_hot(self, pred_binary, mask, cls):
        if self.few_shot:
            assert cls is not None, f"got cls=None in few-shot DiceMeter"
        if cls is not None:
            cls = cls[0]
            mask = mask * cls
            pred_binary = pred_binary * cls

        mask_one_hot = one_hot(mask, num_classes=9)  # (1, C, H, W, D)
        pred_one_hot = one_hot(pred_binary, num_classes=9)
        return mask_one_hot, pred_one_hot

    def update(self, pred_binary, mask, cls=None, name=None, support_ins=None):
        """
        :param pred_binary: (1, 1, H, W, D)
        :param mask: (1, 1, H, W, D)
        :param cls: (1)
        :param name: query image name, only used at few_shot test
        :param support_ins: str, only used at few_shot test
        """
        mask_one_hot, pred_one_hot = self.one_hot(pred_binary, mask, cls)
        mean_dice = self.metric_fn(y_pred=pred_one_hot, y=mask_one_hot).sum(dim=0)  # (C)
        nan = torch.isnan(mean_dice)
        n_n = (~nan).float()
        mean_dice[nan] = 0
        self.sum += mean_dice.cpu()
        self.count += n_n.cpu()

        if self.test:
            name = name[0]
            if cls is None:
                for cls in range(1, 9):
                    if name not in self.result_dict[cls].keys():
                        self.result_dict[cls][name] = {}
                    self.result_dict[cls][name]["N/A"] = mean_dice[cls - 1].item()
            else:
                cls = cls.item()
                support_ins = support_ins[0].item()
                if name not in self.result_dict[cls].keys():
                    self.result_dict[cls][name] = {}
                self.result_dict[cls][name][support_ins] = mean_dice[cls - 1].item()

    def get_average(self, step):

        self.count[self.count == 0] = -1
        mean = self.sum / self.count
        print(self.tag)
        for k, v in zip(self.labels, mean):
            if self.writer is not None:
                self.writer.add_scalar(tag=f"{self.tag}/{k}", scalar_value=v, global_step=step)
            print(f"{k}: {v}")

        metric = torch.mean(mean[self.count > 0])
        if self.writer is not None:
            self.writer.add_scalar(tag=f"{self.tag}/mean", scalar_value=metric, global_step=step)
            print(f"mean: {metric}")

        print("\n")

        self.reset()

        return metric, self.result_dict


class HausdorffMeter(DiceMeter):
    def __init__(self, writer, few_shot=False, test=False):
        super(HausdorffMeter, self).__init__(writer, few_shot, test)
        self.metric_fn = HausdorffDistanceMetric(
            include_background=False,
            percentile=95,
            reduction="sum_batch"
        )
        self.tag = "hausdorff_few_shot" if few_shot else "hausdorff"
        self.reset()


class SurfaceDistanceMeter(DiceMeter):
    def __init__(self, writer, few_shot=False, test=False):
        super(SurfaceDistanceMeter, self).__init__(writer, few_shot, test)
        self.metric_fn = SurfaceDistanceMetric(
            include_background=False,
            reduction="sum_batch"
        )
        self.tag = "surface_distance_few_shot" if few_shot else "surface_distance"
        self.reset()
