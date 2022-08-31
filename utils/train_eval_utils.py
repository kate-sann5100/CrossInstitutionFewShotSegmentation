import argparse
import os
import pickle
import random
from collections import OrderedDict

import torch

import numpy as np

from utils import config


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', default=-1, dest='fold')
    parser.add_argument('--ins', default=3, dest='novel_ins')
    parser.add_argument('--shot', default=1, dest='shot')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_base_ins', action='store_true')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--config', type=str, default='config/fewshot.yaml')
    parser.add_argument('--manual_seed', default=321, dest='manual_seed')
    args = parser.parse_args()

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.fold = int(args.fold)
    cfg.novel_ins = int(args.novel_ins)
    cfg.shot = int(args.shot)
    cfg.test = args.test
    cfg.test_base_ins = args.test_base_ins
    cfg.vis = args.vis
    cfg.manual_seed = args.manual_seed
    print(cfg)
    return cfg


def set_seed(manual_seed):
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    random.seed(manual_seed)


def cuda_batch(batch):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.cuda()


def trim_depth(mask, img_list):
    """
    Crop depthwise based on mask
    !!! Note: this function only support batch_size = 1 !!!
    :param mask: (1, 1, ...)
    :param img_list: N x (1, 1, ...)
    :return:
    """
    target_slice = torch.sum(mask, dim=(0, 1, 2, 3)) != 0
    mask = mask[..., target_slice]
    img_list = [i[..., target_slice] for i in img_list]
    return mask, img_list


def get_save_dir(args):
    if args.model == "baseline_2d":
        save_dir = f"./ckpt/baseline_2d/fold{args.fold}_ins{args.novel_ins}"
    elif args.model == "finetune":
        save_dir = f"./ckpt/finetune/fold{args.fold}_ins{args.novel_ins}"
    elif args.model == "ours":
        save_dir = f"./ckpt/few_shot/fold{args.fold}_ins{args.novel_ins}"
        if args.con:
            save_dir = f"{save_dir}_con"
        if args.align:
            save_dir = f"{save_dir}_align"
        if args.train_ratio != 1:
            save_dir = f"{save_dir}_{args.train_ratio}"
    else:
        raise ValueError(f"unrecognised model {args.model}")
    return save_dir


def save_result_dicts(args, save_dir, dice_result_dict, hausdorff_result_dict):
    with open(f'{save_dir}/ins{args.query_ins}_{args.shot}shot_dice_result_dict.pkl', 'wb') as f:
        pickle.dump(dice_result_dict, f)
    with open(f'{save_dir}/ins{args.query_ins}_{args.shot}shot_hausdorff_result_dict.pkl', 'wb') as f:
        pickle.dump(hausdorff_result_dict, f)