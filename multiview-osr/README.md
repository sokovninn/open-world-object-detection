<div id="top"></div>

# Multi-view Open Set Recognition using Vision Transformer
This repository provides the implementation of the multi-view open set recognition using Vision Transformer (ViT).

<p align="right">(<a href="#top">back to top</a>)</p>


## Getting Started

### Prerequisites

Create an conda environment and install the dependencies
```bash
conda env create -f environment.yml 
```

Activate the environment
```bash
conda activate osr
```

### Pretrained ViT model
Download pretrained ViT-B/16 from this [[link](https://drive.google.com/file/d/1gEcyb4HUDzIvu7lQWTOyDC1X00YzCxFx/view?usp=sharing)]
and put it in the folder of "./pretrained_model/"

### Datasets
Download the TinyImageNet dataset from this [[link](https://drive.google.com/file/d/1oJe95WxPqEIWiEo8BI_zwfXDo40tEuYa/view)] and extract it in the folder of "./data/"

The other datasets will be downloaded automatically if they are not existed in the "./data/" folder.


### Training

To train the model, run the following command:

```bash
python train_classifier.py --exp-name=tinrweights20k5u --n-gpu 1 --tensorboard --image-size 64 --batch-size 8 --num-workers 4 --train-steps 128000 --lr 0.001 --wd 1e-5 --dataset TinyImageNet --num-classes 25 --random-seed 0  --checkpoint-path pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth --dataset-path=tin_rweights_20k5u | python train_classifier.py --exp-name=tinrweights20k50u --n-gpu 1 --tensorboard --image-size 64 --batch-size 8 --num-workers 4 --train-steps 128000 --lr 0.001 --wd 1e-5 --dataset TinyImageNet --num-classes 70 --random-seed 0  --checkpoint-path pretrained_model/imagenet21k+imagenet2012_ViT-B_16.pth --dataset-path=tin_rweights_20k50u
```

### Testing
```bash
 python test_multiview.py --checkpoint-path=experiments/save/tinclassifier10last_TinyImageNet_b16_bs32_lr0.001_wd1e-05_nc10_rs0_230120_182518/checkpoints/ckpt_epoch_best.pth  --image-size=128 --batch-size=1 --num-classes=10 --cuda --exp-reps=1
```

## Acknowledgements

This repository is based on the following work:

_Open Set Recognition using Vision Transformer with an Additional Detection Head_<br>Feiyang Cai, Zhenkai Zhang, Jie Liu, and Xenofon Koutsoukos
[[PDF](https://arxiv.org/pdf/2203.08441.pdf)]
```
@article{cai2022open,
       author = {Cai, Feiyang and Zhang, Zhenkai and Liu, Jie and Koutsoukos, Xenofon},
        title = {Open Set Recognition using Vision Transformer with an Additional Detection Head},
      journal = {arXiv preprint arXiv:2203.08441},
         year = 2022,
}
```
