import os
import nibabel as nib
import numpy as np


class Visualisation:

    def __init__(self, save_path):
        self.save_path = save_path
        print(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    def vis(self, query, support, pred, cls):
        affine = np.array([[0.75, 0, 0, 0], [0, 0.75, 0, 0], [0, 0, 2.5, 0], [0, 0, 0, 1]])
        name = query.pop("name")[0]
        support_ins = support["ins"][0]
        description = f"{name}_ins{support_ins}_cls{cls[0]}"
        sz = query["t2w"].shape

        # q_t2w, s_t2w, q_mask_gt, s_mask, q_mask
        # q_seg, s_seg, q_seg_gt, s_seg_gt
        # aligned_s_t2w, aligned_s_mask, aligned_s_seg_gt

        vis_dict = {
            "q_t2w": query["t2w"],
            "s_t2w": support["t2w"],
            "q_mask_gt": query["mask"],
            "s_mask": support["mask"],
            "q_mask": pred["mask"]
        }
        if "q_seg" in pred.keys():
            vis_dict.update(
                {
                    "q_seg": pred["q_seg"],
                    "s_seg": pred["s_seg"],
                    "q_seg_gt": query["seg"],
                    "s_seg_gt": support["seg"],
                    "aligned_s_t2w": pred["aligned_s_t2w"],
                    "aligned_s_mask": pred["aligned_s_mask"],
                    "aligned_s_seg_gt": pred["aligned_s_seg_gt"]
                }
            )

        for k, v in vis_dict.items():
            img = nib.Nifti1Image(
                v[0].reshape(*sz[-3:]).detach().cpu().numpy().astype(dtype=np.float32),
                affine=affine
            )
            nib.save(img, f"{self.save_path}/{description}_{k}.nii")
