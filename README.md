# Cross Institution Few Shot Segmentation
This repository includes implementation of methods proposed in the paper
[Prototypical few-shot segmentation for cross-institution male pelvic structures with spatial registration](https://arxiv.org/pdf/2209.05160)

### Install dependencies
To install dependencies, run `pip install -r requirements.txt` (python version 3.8 is recommended).

### Data preparation
Data could be downloaded [here](https://zenodo.org/record/7013610).

Put the downloaded data under `data_folder` as the following structure
```
data_folder
├── instiution.txt
├── data
    ├──001000_img.nii
    ├──001000_mask.nii
    ├──...
```
Update the path to `data_folder` in config files.

### Evaluate trained model
All trained model could be downloaded [here](https://drive.google.com/file/d/1CtNYaqFw13pn-6FoiF99tIGEBuVnH9Hz/view?usp=sharing)

Rename the `upload_ckpt` folder as `ckpt` and put it under the root directory such that:
```
CrossInstitutionFewShotSegmentation
├── ckpt
    ├──baseline_2d
    ├──few_shot
    ├──finetune
```

To evaluate the proposed method (`3d_con_align`), execute the following command:
```
python fewshot.py --config cofig/few_shot.yaml \
    --fold ${novel organ fold}
    --ins ${novel institution}
    --test
```
To evaluate the 2d baseline (`2d`), download the resnet50 weight pretrained on 
ImageNet from [here](https://drive.google.com/file/d/1tvbnA7wCpZtZfGe1HPIkaPo-ig1Wy0g2/view?usp=sharing) 
and place under the `model` directory, execute the following 
command:
```
python fewshot.py --config config/baseline_2d.yaml \
    --fold ${novel organ fold}
    --ins ${novel institution}
    --test
```
To evaluate the finetune baseline (`3d_finetune`), execute the following command:
```
python finetune.py --config config/finetune.yaml \
    --fold ${novel organ fold}
    --ins ${novel institution}
    --test
```

### Train
To train the proposed method (`3d_con_align`), execute the following command:
```
python fewshot.py --config config/few_shot.yaml \
    --fold ${novel organ fold}
    --ins ${novel institution}
```
To train the 2d baseline (`2d`), download the resnet50 weight pretrained on 
ImageNet from [here](https://drive.google.com/file/d/1tvbnA7wCpZtZfGe1HPIkaPo-ig1Wy0g2/view?usp=sharing) 
and place under the `model` directory, execute the following 
command:
```
python fewshot.py --config config/few_shot.yaml \
    --fold ${novel organ fold}
    --ins ${novel institution}
```
To train the finetune baseline (`3d_finetune`), execute the following command:
```
python finetune.py --config config/finetune.yaml \
    --fold ${novel organ fold}
    --ins ${novel institution}
```