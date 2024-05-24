import argparse
import os
import torch
from src.config import *
from src.model import VisionTransformer as ViT
from torch.utils.data import DataLoader
from src.dataset import *
from torch.nn import functional as F
import sklearn.metrics as skm
from tqdm import tqdm
from multiview_dataset import MultiviewDataset
from measure_osrdetector import get_distances, get_roc_sklearn, euclidean_dist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import json
from sklearn.manifold import TSNE
import torch.nn as nn
import random
from shutil import copy
import shutil


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


known_classes_tiny = {
    "keyboard": [0],
    "water_bottle": [1],
    "flashlight": [2], 
    "pitcher": [3],
    "plate": [4],
    "bell_pepper": [5],
    "orange": [6],
    "lemon": [7],
    "banana": [8],
    "coffee_mug": [9], 
}


def parse_option():
    parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')

    parser.add_argument("--exp-name", type=str, default="ft", help="experiment name")
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='batch size for data loader')
    parser.add_argument('--in-dataset', default='cifar10', required=False, help='cifar10 | cifar100 | stl10 | ImageNet30')
    parser.add_argument("--in-num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument('--out-dataset', default='cifar10', required=False, help='cifar10 | cifar100 | stl10 | ImageNet30')
    parser.add_argument("--out-num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--image-size", type=int, default=224, help="input image size", choices=[64, 128, 160, 224, 384, 448])

    opt = parser.parse_args()

    return opt

def run_model(model, loader, savedir, known_classes):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    os.makedirs(savedir, exist_ok=True)
    print(known_classes)
    for images, target in tqdm(loader):
        target_label = target[1]
        if target_label in known_classes:
            dest = savedir + '/' + str(known_classes.index(target_label))
        else:
            images = images.cuda()
            output, classifier = model(images,feat_cls=True)

            pred_label = torch.argmax(classifier, dim=1).data.cpu()
            
            dest = savedir + '/' + str(len(known_classes) + int(pred_label))

        image_path = target[0][0]
        file = image_path.split('/')[-1]

        if not os.path.exists(dest):
            os.mkdir(dest)
            os.mkdir(dest + '/images')

        dest = dest + '/images/' + file
        copy(image_path, dest)

    return None


def main(opt, model):
    if opt.cuda:
        model = model.cuda()
    model.eval()

    #if not os.path.exists(opt.save_path):
    #os.mkdir(opt.save_path, exist_ok=True)
        #os.mkdir(dest + '/images')
    if os.path.exists(opt.save_path):
        shutil.rmtree(opt.save_path)
    random.seed(opt.random_seed)
    known_classes = random.sample(range(0, 200), opt.num_known_classes)
    
    train_dataset = getTinyImageNetDataset(image_size=opt.image_size, split='train', data_path="data", known_classes=range(200), unknown_filter=True)
    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    valid_dataset = getTinyImageNetDataset(image_size=opt.image_size, split='in_test', data_path="data", known_classes=range(200), unknown_filter=True)
    print(len(valid_dataset))
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    test_dataset = getTinyImageNetDataset(image_size=opt.image_size, split='test', data_path="data", known_classes=range(200), unknown_filter=True)
    print(len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)

    
    run_model(model, train_dataloader, os.path.join(opt.save_path, "train"), known_classes)
    run_model(model, valid_dataloader, os.path.join(opt.save_path, "val"), known_classes)
    run_model(model, test_dataloader, os.path.join(opt.save_path, "test"), known_classes)


    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic config
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['t16','vs16','s16', 'b16', 'b32', 'l16', 'l32', 'h14'])
    #parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=64, help="input image size", choices=[64, 128, 160, 224, 384, 448])
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num-known-classes", type=int, default=20, help="number of classes in dataset")
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str,
                        help="ViT pre-trained model type")
    parser.add_argument("--cuda", action='store_true')
    #parser.add_argument("--dataset", type=str, default='TinyImageNet', help="dataset for fine-tunning/evaluation")
    #parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument("--num-unknown-classes", type=int, default=1, help="number of classes in dataset")
    parser.add_argument("--save-path", type=str, default='data/tiny_osr', help='save path')
    parser.add_argument("--random-seed", type=int, default=0, help="random seed for choosing the training classes")




    config = parser.parse_args()
    config = get_b16_config(config)

    model = ViT(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_unknown_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate,
        )

    def init_weights(m):
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)


    main(config, model)


