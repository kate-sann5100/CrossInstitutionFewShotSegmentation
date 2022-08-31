import torch
from torch.utils.data import Dataset

from dataset.dataset_utils import get_transform, get_img, get_institution_patient_dict, mask_trim


class SegmentationDataset(Dataset):

    def __init__(self, args, mode):
        super(SegmentationDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.novel_cls = [args.fold + 1, args.fold + 5] if args.fold > 0 else []

        self.seg_path, self.image_path = f"{args.data_path}/data", f"{args.data_path}/data"

        institution_patient_dict = get_institution_patient_dict(
            data_path=args.data_path,
            mode=mode,
            novel_ins=args.novel_ins
        )
        self.img_list = []
        for ins, patient_list in institution_patient_dict.items():
            self.img_list.extend([(p, ins) for p in patient_list])

        self.transform = get_transform(
            augmentation=self.mode == "train",
            size=[args.size[0], args.size[1], 76],
            resolution=args.resolution
        )

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        img = get_img(img, self.transform, self.image_path, self.seg_path)
        img = mask_trim(img, None, self.novel_cls, self.args.size)
        for n_cls in self.novel_cls:
            img["seg"][img["seg"] == n_cls] = 0

        return img
