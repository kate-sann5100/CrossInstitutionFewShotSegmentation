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
    save_dir = f"./ckpt/supervised/ins{args.novel_ins}"

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

        val_metric, _, _ = validation(args, model, val_loader, writer, step_count)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(ckpt, f'{save_dir}/best_ckpt.pth')


def validation(args, model, loader, writer=None, step=None, vis=None, test=False):
    dice_meter = DiceMeter(writer, few_shot=False, test=test)
    hausdorff_meter = HausdorffMeter(writer, few_shot=False, test=test)
    model.eval()

    with torch.no_grad():
        for val_step, batch in enumerate(loader):
            cuda_batch(batch)
            binary = model(batch)  # (1, 1, ...)
            dice_meter.update(
                binary["seg"], batch["seg"], name=batch["name"],
                cls=None, support_ins=None
            )

            if test:
                # resample to resolution = (1, 1, 1)
                spacingd = Spacingd(["pred", "gt"], pixdim=[1, 1, 1], mode="nearest")
                meta_data = {"affine": batch["t2w_meta_dict"]["affine"][0]}
                resampled = spacingd(
                    {
                        "pred": binary["seg"][0],
                        "gt": batch["seg"][0],
                        "pred_meta_dict": meta_data,
                        "gt_meta_dict": meta_data.copy()
                    }
                )
                hausdorff_meter.update(
                    resampled["pred"].unsqueeze(0), resampled["gt"].unsqueeze(0),
                    cls=None, name=batch["name"], support_ins=None
                )
                # for cls in range(1, 9):
                #     pred = resampled["pred"] == cls
                #     gt = resampled["gt"] == cls
                #     hausdorff_meter.update(
                #         pred.unsqueeze(0), gt.unsqueeze(0), torch.tensor([cls]),
                #         name=batch["name"], support_ins=None
                #     )

        dice_metric, dice_result_dict = dice_meter.get_average(step)
        if test:
            hausdorff_metric, hausdorff_result_dict = hausdorff_meter.get_average(step)
        else:
            hausdorff_result_dict = None

    return dice_metric, dice_result_dict, hausdorff_result_dict


def val_worker(args):
    set_seed(args.manual_seed)
    save_dir = f"./ckpt/supervised/ins{args.novel_ins}"
    print(save_dir)

    args.query_ins = args.novel_ins
    test_dataset = SegmentationDataset(args=args, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = Pretrain(args)
    model = torch.nn.DataParallel(model.cuda())
    state_dict = torch.load(f"{save_dir}/best_ckpt.pth")["model"]
    model.load_state_dict(state_dict, strict=True)

    # vis = Visualisation(save_path=f"{save_dir}/vis") if args.vis else None

    _, dice_result_dict, hausdorff_result_dict = validation(
        args, model, test_loader, writer=None, step=None, vis=None, test=True)

    save_result_dicts(args, save_dir, dice_result_dict, hausdorff_result_dict)


if __name__ == '__main__':
    main()