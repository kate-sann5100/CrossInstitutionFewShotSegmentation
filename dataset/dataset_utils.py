import os

import torch
from torch.nn import functional as F
import numpy as np
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    ToTensord,
    RandAffined, Resized, RandSpatialCropd,
)

organ_list = ["BladderMask", "BoneMask", "ObdInternMask", "TZ",
              "CG", "RectumMask", "SV", "NVB"]
organ_index_dict = {organ: i + 1 for i, organ in enumerate(organ_list)}

institution_list = ["UCL", "Prostate3T", "ProstateDx", "ProstateMRI", "bergen", "Nijmegen", "Rutgers"]


def get_transform(augmentation, size, resolution):
    pre_augmentation = [
        LoadImaged(keys=["t2w", "seg"]),
        AddChanneld(keys=["t2w", "seg"]),
        Spacingd(
            keys=["t2w", "seg"],
            pixdim=resolution,
            mode=("bilinear", "nearest"),
        ),
    ]

    post_augmentation = [
        NormalizeIntensityd(keys=["t2w"]),
        ScaleIntensityd(keys=["t2w"]),
        ToTensord(keys=["t2w", "seg"])
    ]

    if augmentation:
        middle_transform = [
            RandAffined(
                keys=["t2w", "seg"],
                spatial_size=size,
                prob=1.0,
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 36),
                shear_range=None,
                translate_range=(20, 20, 4),
                scale_range=(0.15, 0.15, 0.15),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                as_tensor_output=False,
                device=torch.device('cuda'),
                allow_missing_keys=False
            )
        ]
    else:
        middle_transform = [
            CenterSpatialCropd(keys=["t2w", "seg"], roi_size=size),
            SpatialPadd(
                keys=["t2w", "seg"],
                spatial_size=size,
                method='symmetric',
                mode='constant',
                allow_missing_keys=False
            )
        ]

    return Compose(pre_augmentation + middle_transform + post_augmentation)


def get_institution_patient_dict(data_path, mode, novel_ins, train_ratio=1):
    """
    choose images based on institution
    :param data_path: str
    :param mode: train/val/test
    :param novel_ins: int
    :param train_ratio: for ablation studies
    :return: dict
    """

    # divide images by institution
    institution_patient_dict = {i: [] for i in range(1, 8)}
    with open(f'{data_path}/institution.txt') as f:
        patient_ins_list = f.readlines()
    for patient_ins in patient_ins_list:
        patient, ins = patient_ins[:-1].split(" ")
        institution_patient_dict[int(ins)].append(patient)

    for k, v in institution_patient_dict.items():
        if mode == "train":
            if k == novel_ins:
                institution_patient_dict[k] = []
            else:
                train_list = v[:-len(v)//4]
                if train_ratio == "ins1only":
                    institution_patient_dict[k] = institution_patient_dict[k][:191] if k == 1 else []
                else:
                    train_size = int(len(train_list) * train_ratio)
                    train_list = train_list[:train_size]
                    institution_patient_dict[k] = train_list
        elif mode == "val":
            if k == novel_ins:
                institution_patient_dict[k] = v[:2]
            else:
                institution_patient_dict[k] = v[-len(v)//4: -len(v)//4 + 2]
        else:
            if k == novel_ins:
                institution_patient_dict[k] = v[2:]
            else:
                institution_patient_dict[k] = v[-len(v) // 4 + 2:]
    return institution_patient_dict


def get_fixed_pair(img_list):
    """
    :param img_list: list of str
    :return:
    """
    num_pair = len(img_list) // 2
    fixed_pair = [
        [img_list[i], img_list[num_pair + i]]
        for i in range(num_pair)
    ]
    return fixed_pair


def get_img(img, transform, image_path, seg_path):
    """
    :param img: tuple (name, ins)
    :param transform:
    :param image_path: str
    :param seg_path: str
    :return:
    t2w: (1, ...)
    seg: (1, ...)
    name: str
    """
    img_name, ins = img
    while True:
        out = transform({
            "t2w": f"{image_path}/{img_name}_img.nii",
            "seg": f"{seg_path}/{img_name}_mask.nii",
            "name": img_name,
            "ins": ins
    })
        if len(torch.unique(out["seg"])) == 9:
            return out
        else:
            print(img_name)


def mask_trim(x, cls, novel_cls, size):
    """
    :param x: dict including t2w, seg and name
    :param cls: int, chosen query class
    :param novel_cls: list of int
    :return:
    """
    if cls == None:
        x["mask"] = x["seg"] > 0
    else:
        # mask cls
        x["mask"] = x["seg"] == cls

    for n_cls in novel_cls:
        x["seg"][x["seg"] == n_cls] = 0

    # trim
    target_slice = torch.sum(x["mask"], dim=(0, 1, 2)) != 0
    for k in ["t2w", "mask", "seg"]:
        x[k] = x[k][..., target_slice]
        x[k] = F.interpolate(
            x[k].unsqueeze(0).to(torch.float),
            size=size,
            mode="trilinear" if k == "t2w" else "nearest"
        ).squeeze(0)

    return x


def sample_pair(idx, img_list_len):
    """
    :param idx: int
    :param img_list_len: int
    :return: int
    """
    out = idx
    while out == idx:
        out = np.random.randint(img_list_len)
    return out
