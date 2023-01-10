import torch
from monai.transforms import Spacingd
from torch.backends import cudnn
from torch.cuda import device_count
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from analyse_result import get_result
from dataset.few_shot_dataset import FewShotDataset
from dataset.segmentation_dataset import SegmentationDataset
from model.baseline_finetune import FineTune, Pretrain
from utils.meter import LossMeter, DiceMeter, HausdorffMeter
from utils.train_eval_utils import get_parser, set_seed, cuda_batch, save_result_dicts
from utils.visualisation import Visualisation


def main():
    args = get_parser()
    cudnn.benchmark = False
    cudnn.deterministic = True
    set_seed(args.manual_seed)

    if args.test:
        val_worker(args)
    else:
        train_worker(args)


def train_worker(args):
    set_seed(args.manual_seed)
    save_dir = f"./ckpt/finetune/fold{args.fold}_ins{args.novel_ins}"

    train_dataset = SegmentationDataset(args=args, mode="train")
    val_dataset = SegmentationDataset(args=args, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=device_count(),
        shuffle=True,
        drop_last=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = Pretrain(args)
    model = torch.nn.DataParallel(model.cuda())
    optimiser = Adam(model.parameters(), lr=1e-4)
    writer = SummaryWriter(log_dir=save_dir)
    num_epochs = 100
    start_epoch = 0
    step_count = 0
    best_metric = 0
    loss_meter = LossMeter(writer=writer)

    validation(args, model, val_loader, writer, step_count)
    for epoch in range(start_epoch, num_epochs):
        print(f"-----------epoch: {epoch}----------")

        model.train()
        for step, batch in tqdm(enumerate(train_loader)):
            optimiser.zero_grad()
            cuda_batch(batch)
            loss_dict = model(batch)
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
        torch.save(ckpt, f'{save_dir}/last_ckpt.pth')
        print("validating...")

        val_metric = validation(args, model, val_loader, writer, step_count)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, f'{save_dir}/best_ckpt.pth')


def validation(args, model, loader, writer=None, step=None, vis=None):
    seg_dice_meter = DiceMeter(writer, few_shot=False)
    model.eval()

    with torch.no_grad():
        for val_step, batch in enumerate(loader):
            cuda_batch(batch)
            binary = model(batch)  # (1, 1, ...)
            seg_dice_meter.update(binary["seg"], batch["seg"], cls=None)

        seg_dice_metric, _ = seg_dice_meter.get_average(step)

    return seg_dice_metric


def val_worker(args):
    set_seed(args.manual_seed)
    save_dir = f"./ckpt/finetune/fold{args.fold}_ins{args.novel_ins}"
    print(save_dir)

    args.query_ins = args.novel_ins
    test_dataset = FewShotDataset(args=args, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = FineTune(args)
    model = torch.nn.DataParallel(model.cuda())
    state_dict = torch.load(f"{save_dir}/best_ckpt.pth")["model"]
    for k, v in state_dict.items():
        if "seg" in k:
            state_dict[k] = model.state_dict()[k]
    model.load_state_dict(state_dict, strict=True)

    if args.vis:
        vis = Visualisation(save_path=f"{save_dir}/vis")
    else:
        vis = None
    dice_result_dict, hausdorff_result_dict = test(
        args, model, state_dict, test_loader, vis=vis)

    save_result_dicts(args, save_dir, dice_result_dict, hausdorff_result_dict)


def val_base_worker(args):
    set_seed(args.manual_seed)
    save_dir = f"./ckpt/finetune/fold{args.fold}_ins{args.novel_ins}"
    print(save_dir)
    model = FineTune(args)
    model = torch.nn.DataParallel(model.cuda())
    state_dict = torch.load(f"{save_dir}/best_ckpt.pth")["model"]
    for k, v in state_dict.items():
        if "seg" in k:
            state_dict[k] = model.state_dict()[k]
    model.load_state_dict(state_dict, strict=True)

    for query_ins in range(1, 8):
        if query_ins == args.novel_ins:
            continue
        args.query_ins = query_ins
        test_dataset = FewShotDataset(args=args, mode="test")
        test_loader = DataLoader(test_dataset, batch_size=1)

        dice_result_dict, hausdorff_result_dict = test(
            args, model, state_dict, test_loader, vis=None)

        save_result_dicts(args, save_dir, dice_result_dict, hausdorff_result_dict)


def test(args, model, pretrained_state_dict, test_loader, vis=None):
    dice_meter = DiceMeter(writer=None, few_shot=True, test=True)
    hausdorff_meter = HausdorffMeter(writer=None, few_shot=True, test=test)

    for test_step, (query, support, cls) in enumerate(test_loader):
        cuda_batch(query)
        cuda_batch(support)
        model.load_state_dict(pretrained_state_dict, strict=True)
        finetune(model, support, cls)
        with torch.no_grad():
            model.eval()
            binary = model(query)  # (1, 1, ...)

            dice_meter.update(
                binary["mask"], query["mask"], cls,
                name=query["name"], support_ins=support["ins"]
            )

            spacingd = Spacingd(["pred", "gt"], pixdim=[1, 1, 1], mode="nearest")
            meta_data = {"affine": query["seg_meta_dict"]["affine"][0]}
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

    dice_metric, dice_result_dict = dice_meter.get_average(step=None)
    hausdorff_metric, hausdorff_result_dict = hausdorff_meter.get_average(step=None)

    _ = get_result(args, dice_result_dict, metric="Dice")
    _ = get_result(args, hausdorff_result_dict, metric="95% Hausdorff Distance")

    return dice_result_dict, hausdorff_result_dict


def finetune(model, support, cls):
    finetune_iters = 10
    optimiser = Adam(model.parameters(), lr=1e-3)
    best_loss, best_ckpt = None, None

    for finetune_step in range(finetune_iters):
        model.train()
        optimiser.zero_grad()
        loss_dict = model(support)
        loss = loss_dict["label"]
        loss.backward()
        optimiser.step()

        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_ckpt = model.state_dict()

    model.load_state_dict(best_ckpt, strict=True)


if __name__ == '__main__':
    main()