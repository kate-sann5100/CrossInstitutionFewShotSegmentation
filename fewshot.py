import os

import torch
from monai.transforms import Spacingd
from torch.backends import cudnn
from torch.cuda import device_count
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from analyse_result import get_result
from dataset.few_shot_dataset import FewShotDataset
from model.baseline_2d import LSNet2D
from model.few_shot_model import LSNet
from model.registration_model import RegistrationModel
from utils.meter import LossMeter, DiceMeter, HausdorffMeter
from utils.train_eval_utils import get_parser, set_seed, cuda_batch, get_save_dir, save_result_dicts
from utils.visualisation import Visualisation


def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)

    if args.test:
        val_worker(args)
    elif args.test_base_ins:
        val_base_worker(args)
    else:
        train_worker(args)


def train_worker(args):
    align = args.align
    set_seed(args.manual_seed)
    save_dir = get_save_dir(args)
    print(save_dir)

    args.query_ins = args.novel_ins
    train_dataset = FewShotDataset(args=args, mode="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )

    val_dataset = FewShotDataset(args=args, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=1)

    if args.model == "baseline_2d":
        model = LSNet2D(args)
    elif args.model == "ours":
        model = LSNet(args, align_head=False)
    else:
        return ValueError(f"unrecognised model {args.model}")

    model = torch.nn.DataParallel(model.cuda())
    optimiser = Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=save_dir)

    num_epochs = 50
    start_epoch = 0
    step_count = 0
    best_metric = 0
    loss_meter = LossMeter(writer=writer)

    for epoch in range(start_epoch, num_epochs):
        print(f"-----------epoch: {epoch}----------")

        model.train()
        for step, (query, support, cls) in enumerate(train_loader):

            optimiser.zero_grad()
            cuda_batch(query)
            cuda_batch(support)
            loss_dict = model(query, support)
            loss = 0
            for k, v in loss_dict.items():
                loss_dict[k] = torch.mean(v)
                loss = loss + torch.mean(v)
            loss.backward()
            optimiser.step()
            loss_dict["total"] = loss
            loss_meter.update(loss_dict)
            step_count += 1

        loss_meter.get_average(step_count)
        ckpt = {
            "epoch": epoch,
            "step_count": step_count,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
        }
        print("validating...")

        val_metric, _, _ = validation(args, model, val_loader, writer, step_count)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, f'{save_dir}/best_ckpt.pth')

    if align:
        state_dict = torch.load(f"{save_dir}/best_ckpt.pth")["model"]
        model = LSNet(args, align_head=True)
        model = torch.nn.DataParallel(model.cuda())
        align_model = RegistrationModel(args)
        align_model = torch.nn.DataParallel(align_model.cuda())
        align_optimiser = Adam(align_model.parameters(), lr=1e-8)
        num_epochs = 100
        start_epoch = 50
        best_metric = 0

        for epoch in range(start_epoch, num_epochs):
            print(f"-----------epoch: {epoch}----------")

            align_model.train()
            for step, (query, support, cls) in enumerate(train_loader):

                align_optimiser.zero_grad()
                cuda_batch(query)
                cuda_batch(support)
                loss_dict = align_model(moving=support, fixed=query)
                loss = 0
                for k, v in loss_dict.items():
                    loss_dict[k] = torch.mean(v)
                    loss = loss + torch.mean(v)
                loss.backward()
                align_optimiser.step()
                loss_dict["total"] = loss
                loss_meter.update(loss_dict)
                step_count += 1

            loss_meter.get_average(step_count)

            # load align head weight
            for k, v in align_model.state_dict().items():
                state_dict[k.replace("model", "align_head")] = v
            model.load_state_dict(state_dict)

            val_metric, _, _ = validation(
                args, model, val_loader, writer, step_count,
            )
            if val_metric > best_metric:
                best_metric = val_metric
                torch.save(
                    {"model": state_dict},
                    f'{save_dir}/best_ckpt.pth'
                )


def val_worker(args):
    set_seed(args.manual_seed)

    save_dir = get_save_dir(args)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    print(save_dir)

    args.query_ins = args.novel_ins
    val_dataset = FewShotDataset(args=args, mode="test")
    val_loader = DataLoader(val_dataset, batch_size=4 if args.model == "baseline_2d" else 1)

    if args.model == "baseline_2d":
        model = LSNet2D(args)
    elif args.model == "ours":
        model = LSNet(args, align_head=True)
    else:
        return ValueError(f"unrecognised model {args.model}")
    print(f"model includes {sum(p.numel() for p in model.parameters())} parameters")
    model = torch.nn.DataParallel(model.cuda())
    state_dict = torch.load(f"{save_dir}/best_ckpt.pth")["model"]
    model.load_state_dict(state_dict, strict=True)

    if args.vis:
        vis = Visualisation(save_path=f"{save_dir}/vis")
    else:
        vis = None

    _, dice_result_dict, hausdorff_result_dict = validation(
        args, model, val_loader, vis=vis, test=True
    )

    if not args.vis:
        save_result_dicts(args, save_dir, dice_result_dict, hausdorff_result_dict)


def val_base_worker(args):
    set_seed(args.manual_seed)

    save_dir = get_save_dir(args)
    print(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.model == "baseline_2d":
        model = LSNet2D(args)
    elif args.model == "ours":
        model = LSNet(args, align_head=True)
    else:
        return ValueError(f"unrecognised model {args.model}")

    model = torch.nn.DataParallel(model.cuda())
    state_dict = torch.load(f"{save_dir}/best_ckpt.pth")["model"]
    model.load_state_dict(state_dict, strict=True)

    for query_ins in range(1, 8):
        if query_ins == args.novel_ins:
            continue

        args.query_ins = query_ins
        val_dataset = FewShotDataset(args=args, mode="test")
        val_loader = DataLoader(val_dataset, batch_size=4 if args.model == "baseline_2d" else 1)

        _, dice_result_dict, hausdorff_result_dict = validation(
            args, model, val_loader, vis=None, test=True
        )

        save_result_dicts(args, save_dir, dice_result_dict, hausdorff_result_dict)


def validation(args, model, loader, writer=None, step=None, vis=None, test=False):
    dice_meter = DiceMeter(writer, few_shot=True, test=test)
    hausdorff_meter = HausdorffMeter(writer, few_shot=True, test=test)
    model.eval()

    with torch.no_grad():
        for val_step, (query, support, cls) in enumerate(loader):

            cuda_batch(query)
            cuda_batch(support)
            binary = model(query, support)  # (1, 1, ...)
            # print(query["name"][0], support["ins"][0], cls[0])
            if not test:
                dice_meter.update(
                    binary["mask"], query["mask"], cls,
                    name=query["name"], support_ins=support["ins"]
                )
            else:
                if args.model == "baseline_2d":
                    query = {
                        # (4, 1, W, H, D) -> (1, W, H, D, 4) -> (1, 1, W, H, D)
                        "t2w": query["t2w"].permute(1, 2, 3, 0, 4).reshape(1, 1, *args.size),  # (1, 1, W, H, D)
                        "mask": query["mask"].permute(1, 2, 3, 0, 4).reshape(1, 1, *args.size),  # (1, 1, W, H, D)
                        "name": query["name"][:1],
                        "ins": query["ins"][:1],
                        "t2w_meta_dict": query["t2w_meta_dict"]
                    }
                    binary = {
                        "mask": binary["mask"].permute(1, 2, 3, 0, 4).reshape(1, 1, *args.size),  # (1, 1, W, H, D)
                    }
                    cls = cls[:1]
                    support["ins"] = support["ins"][:1]
                    dice_meter.update(
                        binary["mask"], query["mask"], cls,
                        name=query["name"], support_ins=support["ins"]
                    )
                    hausdorff_meter.update(
                        binary["mask"], query["mask"], cls,
                        name=query["name"], support_ins=support["ins"]
                    )

                else:
                    dice_meter.update(
                        binary["mask"], query["mask"], cls,
                        name=query["name"], support_ins=support["ins"]
                    )

                    # resample to resolution = (1, 1, 1)
                    spacingd = Spacingd(["pred", "gt"], pixdim=[1, 1, 1], mode="nearest")
                    meta_data = {"affine": query["t2w_meta_dict"]["affine"][0]}
                    resampled = spacingd(
                        {
                            "pred": binary["mask"][0],
                            "gt": query["mask"][0],
                            "pred_meta_dict": meta_data,
                            "gt_meta_dict": meta_data.copy()
                         }
                    )
                    hausdorff_meter.update(
                        resampled["pred"].unsqueeze(0), resampled["gt"].unsqueeze(0), cls,
                        name=query["name"], support_ins=support["ins"]
                    )

            if vis is not None:
                vis.vis(
                    query=query,
                    support=support,
                    pred=binary,
                    cls=cls
                )

        dice_metric, dice_result_dict = dice_meter.get_average(step)
        if test:
            hausdorff_metric, hausdorff_result_dict = hausdorff_meter.get_average(step)
            # _ = get_result(args, dice_result_dict, metric="Dice")
            # _ = get_result(args, hausdorff_result_dict, metric="95% Hausdorff Distance")
        else:
            hausdorff_result_dict = None

    return dice_metric, dice_result_dict, hausdorff_result_dict


if __name__ == '__main__':
    main()