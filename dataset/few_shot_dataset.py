import numpy as np
import torch
from monai.transforms import RandSpatialCropd

from torch.utils.data import Dataset

from dataset.dataset_utils import get_transform, sample_pair, get_img, mask_trim, get_institution_patient_dict


class FewShotDataset(Dataset):

    def __init__(self, args, mode):
        super(FewShotDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.shot = args.shot

        self.novel_cls = [args.fold, args.fold + 4]
        self.query_cls = [i for i in range(1, 9) if i not in self.novel_cls] if self.mode == "train" else self.novel_cls

        self.seg_path, self.image_path = f"{args.data_path}/data", f"{args.data_path}/data"

        institution_patient_dict = get_institution_patient_dict(
            data_path=args.data_path,
            mode=mode,
            novel_ins=args.novel_ins,
            train_ratio=args.train_ratio
        )

        if self.mode == "train":
            self.img_list = []
            for ins, patient_list in institution_patient_dict.items():
                self.img_list.extend([(p, ins) for p in patient_list])
            print(len(self.img_list))
        else:
            if self.shot == 1:
                self.fixed_pair = []
                query_list = institution_patient_dict[args.query_ins]
                # for each query image
                for query in query_list:
                    # for each institution
                    for ins, patient_list in institution_patient_dict.items():
                        while True:
                            support = patient_list[np.random.randint(0, len(patient_list))]
                            if support != query:
                                break
                        # for each class
                        for cls in self.novel_cls:
                            if args.model == "baseline_2d":
                                for start_depth in (0, 10, 20, 30):
                                    self.fixed_pair.append([(query, args.novel_ins), (support, ins), cls, start_depth])
                            else:
                                self.fixed_pair.append([(query, args.novel_ins), (support, ins), cls])
            else:
                self.fixed_pair = []
                query_list = institution_patient_dict[args.query_ins]
                # for each query image
                for query in query_list:
                    # for each institution
                    for ins, patient_list in institution_patient_dict.items():
                        if len(patient_list) < self.shot:
                            continue
                        support_list = []
                        for i in range(self.shot):
                            while True:
                                support = patient_list[np.random.randint(0, len(patient_list))]
                                if support != query and (support not in support_list):
                                    break
                            support_list.append(support)
                        # for each class
                        for cls in self.novel_cls:
                            if args.model == "baseline_2d":
                                for start_depth in (0, 10, 20, 30):
                                    self.fixed_pair.append(
                                        [
                                            (query, args.novel_ins),
                                            [(support, ins) for support in support_list],
                                            cls, start_depth
                                        ]
                                    )
                            else:
                                self.fixed_pair.append(
                                    [
                                        (query, args.novel_ins),
                                        [(support, ins) for support in support_list],
                                        cls
                                    ]
                                )

        # exclude query-support pairs not specified in "vis_list.pth"
        vis_list = torch.load("vis_list.pth")
        self.fixed_pair = [
            [(query, args.novel_ins), (support, ins), cls]
            for (query, args.novel_ins), (support, ins), cls in self.fixed_pair
            if (query, ins, cls) in vis_list
        ]

        self.transform = get_transform(
            augmentation=self.mode == "train",
            size=[args.size[0], args.size[1], 76],
            resolution=args.resolution
        )

        self.random_crop = RandSpatialCropd(
            keys=["t2w", "mask", "seg"],
            roi_size=(args.size[0], args.size[1], 10),
            random_size=False
        )

    def __len__(self):
        return len(self.img_list) if self.mode == "train" else len(self.fixed_pair)

    def __getitem__(self, idx):
        if self.mode == "train":
            query = idx
            support = sample_pair(idx, len(self.img_list))
            query, support = self.img_list[query], self.img_list[support]
            cls = self.query_cls[np.random.randint(len(self.query_cls))]
        else:
            if self.args.model == "baseline_2d":
                query, support, cls, start_depth = self.fixed_pair[idx]
            else:
                query, support, cls = self.fixed_pair[idx]

        query = get_img(query, self.transform, self.image_path, self.seg_path)
        query = mask_trim(query, cls, self.novel_cls, self.args.size)

        if self.shot == 1:
            support = get_img(support, self.transform, self.image_path, self.seg_path)
            support = mask_trim(support, cls, self.novel_cls, self.args.size)
        else:
            support_list = support
            support_list = [
                get_img(support, self.transform, self.image_path, self.seg_path)
                for support in support_list
            ]
            support_list = [
                mask_trim(support, cls, self.novel_cls, self.args.size)
                for support in support_list
            ]
            support = {
                "t2w": torch.stack([s["t2w"] for s in support_list]),  # (S, 1, W, H, D)
                "mask": torch.stack([s["mask"] for s in support_list]),  # (S, 1, W, H, D)
                "seg": torch.stack([s["seg"] for s in support_list]),  # (S, 1, W, H, D)
                "name": [s["name"] for s in support_list],
                "ins": support_list[0]["ins"]
            }

        if self.args.model == "baseline_2d":
            if self.mode == "train":
                query = self.random_crop(query)
                support = self.random_crop(support)
            else:
                query = self.depth_crop(query, start_depth)
                support = self.depth_crop(support, start_depth)
        return query, support, cls

    def depth_crop(self, input, start_depth):
        """
        :param input
            "t2w": (1, ...)
            "mask": (1, ...)
            "seg": (1, ...)
            "name": str
        :param start_depth
        """
        for k in ["t2w", "mask", "seg"]:
            input[k] = input[k][..., start_depth: start_depth + 10]
        input["slice"] = start_depth
        return input
