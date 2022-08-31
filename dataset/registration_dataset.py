from torch.utils.data import Dataset

from dataset.dataset_utils import get_transform, get_img_list, get_img, get_fixed_pair, sample_pair


class RegistrationDataset(Dataset):

    def __init__(self, args, mode):
        super(RegistrationDataset, self).__init__()
        self.args = args
        self.mode = mode

        dataset_path = "/raid/candi/yiwen/data/pelvic_big"
        self.seg_path, self.image_path = dataset_path, dataset_path

        self.img_list = get_img_list(self.seg_path, self.mode)
        if self.mode == "val":
            self.fixed_pair = get_fixed_pair(self.img_list)

        self.transform = get_transform(
            augmentation=self.mode == "train",
            size=args.size,
            resolution=args.resolution
        )

    def __len__(self):
        return len(self.img_list) if self.mode == "train" else len(self.fixed_pair)

    def __getitem__(self, idx):
        if self.mode == "val":
            moving, fixed = self.fixed_pair[idx]
        else:
            moving = idx
            fixed = sample_pair(idx, len(self.img_list))
            moving, fixed = self.img_list[moving], self.img_list[fixed]

        moving = get_img(moving, self.transform, self.image_path, self.seg_path)
        fixed = get_img(fixed, self.transform, self.image_path, self.seg_path)
        return moving, fixed
