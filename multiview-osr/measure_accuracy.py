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
    parser.add_argument("--image-size", type=int, default=64, help="input image size", choices=[64, 128, 160, 224, 384, 448])

    opt = parser.parse_args()

    return opt

def run_model_unknown_unknown(model, loader, savedir, known_classes, split):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    total = np.zeros(120)
    correct = np.zeros(120)
    total_num = 0
    correct_num = 0
    #known_classes = [2, 6, 10, 27, 29, 34, 36, 56, 66, 105, 106, 117, 129, 151, 161, 163, 170, 175, 191, 198]
    #known_classes = list(range(20))
    print(known_classes)
    for images, target in tqdm(loader):
        images = images[0]
        if target[0] in known_classes_tiny:
            continue
        total_num += 1
        #total += images.size(0)
        images = images.cuda()
        output, classifier = model(images,feat_cls=True)

        pred_label = torch.argmax(classifier, dim=1).data.cpu()

        if pred_label not in known_classes:
            correct_num +=1

    print(correct_num, total_num)
    print(correct_num / total_num)
    return correct, total, correct / total

def run_model(model, loader, savedir, known_classes, split):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    total = np.zeros(120)
    correct = np.zeros(120)
    #known_classes = [2, 6, 10, 27, 29, 34, 36, 56, 66, 105, 106, 117, 129, 151, 161, 163, 170, 175, 191, 198]
    #known_classes = list(range(20))
    print(known_classes)
    for images, target in tqdm(loader):
        #total += images.size(0)
        images = images.cuda()
        output, classifier = model(images,feat_cls=True)
    #print(classifier.shape)

        pred_label = torch.argmax(classifier, dim=1).data.cpu()
        # total[int(target)] += 1
        #print(target)
        if target in known_classes:
            total[target] += 1
        else:
            total[3] += 1
        #print(int(target))
        #print(total)
        #print(pred_label, target)
        # if pred_label == target:
        #         correct[target] += 1
        if pred_label in known_classes:
            if pred_label == target:
                correct[target] += 1
        else:
            if target not in known_classes:
                correct[3] += 1
        #print(correct / total)

        #target_label = target[1]
        #image_path = target[0][0]
        #print(pred_label, target_label, image_path)
    return correct, total, correct / total


def main(opt, model):
    if opt.cuda:
        model = model.cuda()
    #model.eval()
    random.seed(0)
    #known_classes_20 = random.sample(range(0, 200), 20)
    #print(known_classes)
    #known_classes_200 = range(200)
    known_classes = range(opt.num_classes)
    
    # train_dataset = getTinyImageNetDataset(image_size=opt.image_size, split='train', data_path="data", known_classes=range(200))
    # print(len(train_dataset))
    # train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    #valid_dataset = getTinyImageNetDataset(image_size=opt.image_size, split='in_test', data_path="data", known_classes=known_classes, dataset_path=opt.dataset_path)
    valid_dataset = MultiviewDataset("/home/nikita/Downloads/rgbd-dataset", image_size=(opt.image_size, opt.image_size), each_n=1, n_views=1)
    #valid_dataset = getTinyImageNetOpenDataset(image_size=opt.image_size, split='in_test', data_path="data", known_classes=known_classes_20)
    print(len(valid_dataset))
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)


    
    #run_model(model, train_dataloader, "data/tiny_osr/train", known_classes, "train")

    #known_classes = list(range(120))
    #known_classes = print(np.array(list(range(120)))[np.array(total) == 25.0])
    #known_classes = list(range(21))
    #known_classes = list(range(13)) + [14, 15, 16, 17, 18, 19, 20]
    #known_classes = list(range(13)) + [15, 16, 17, 18, 19, 20, 21]
    #known_classes = list(range(13)) + [18, 19, 20, 21, 22, 23, 24]
    #known_classes = list(range(13)) + [23, 24, 25, 26, 27, 28, 29]
    #known_classes = list(range(13)) + [23,34,45,56, 67, 68, 69]
    known_classes = [  0,   1,   2,  13,  24,  25,  26,  27,  28,  29,  30,  31,  32,  35,  43,   54, 65,  76,  87,  98]
    # for i in range(13,63):
    #     known_classes.remove(i)
    # known_classes.remove(13)
    # known_classes.remove(14)
    # known_classes.remove(15)
    # known_classes.remove(16)
    # known_classes.remove(17)
    #known_classes.remove(14)

    #correct, total, accuracy = run_model(model, valid_dataloader, "data/tiny_osr/val", known_classes, "val")
    correct, total, accuracy = run_model_unknown_unknown(model, valid_dataloader, "data/tiny_osr/val", known_classes, "val")

    print(np.array(list(range(120)))[np.array(total) == 25.0])
    print(np.array(accuracy)[np.array(total) == 25.0])

    total_results = {"correct": list(correct), "total": list(total), "accuracy": list(accuracy), "mean_known": np.mean(accuracy[known_classes])}
    print(total_results)
    with open(f'acc_osr_results_{opt.num_classes}.json', 'w') as fp:
       json.dump(total_results, fp)

    # with open('acc_osr_results_200.json', 'r') as fp:
    #     total_results = json.load(fp)

    acc_20 = []

    #for acc in total_results["accuracy"]:
    # for i, acc in enumerate(total_results["accuracy"]):
    #     if acc > 0.5:
    #         print(i)
    #         acc_20.append(acc)

    # for class_id in known_classes_20:
    #     acc_20.append(total_results["accuracy"][class_id])
    # print(known_classes_20)
    # print(acc_20)
    # print(np.mean(acc_20[:20]))
    # print(len(acc_20))


    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic config
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['t16','vs16','s16', 'b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=64, help="input image size", choices=[64, 128, 160, 224, 384, 448])
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--num-classes", type=int, default=1000, help="number of classes in dataset")
    parser.add_argument('--model_name', default='vit_base_patch16_224', type=str,
                        help="ViT pre-trained model type")
    parser.add_argument("--eval", action='store_true',help='evaluate on dataset')
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--cls-method", type=str, default="CCD", help='known/unknwon classification method', choices=['CCD','MSP','MLS','entropy'])
    parser.add_argument("--fusion-method", type=str, default="mean", help='feature fusion method', choices=['mean','max'])
    parser.add_argument("--each-n", type=int, default=10, help="take every nth view")
    parser.add_argument("--n-views", type=int, default=1, help="number of views")
    parser.add_argument("--dataset", type=str, default='TinyImageNet', help="dataset for fine-tunning/evaluation")
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    parser.add_argument("--compute-cmeans", action='store_true')
    parser.add_argument("--exp-reps", type=int, default=10, help="Experiment repetitions")
    parser.add_argument("--vis-features", action='store_true')
    parser.add_argument("--dataset-path", type=str, default='tiny-imagenet-200', help='dataset folder')
    parser.add_argument("--num_classes", type=int, default=21, help='dataset folder')

    config = parser.parse_args()
    config = get_b16_config(config)

    model = ViT(
        image_size=(config.image_size, config.image_size),
        patch_size=(config.patch_size, config.patch_size),
        emb_dim=config.emb_dim,
        mlp_dim=config.mlp_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_classes=config.num_classes,
        attn_dropout_rate=config.attn_dropout_rate,
        dropout_rate=config.dropout_rate,
        )

    # load checkpoint
    state_dict = torch.load(config.checkpoint_path, map_location=torch.device("cpu"))['state_dict']
    print("Loading pretrained weights from {}".format(config.checkpoint_path))
    if not config.eval and config.num_classes != state_dict['classifier.weight'].size(0)  :#not
        #del state_dict['classifier.weight']
        #del state_dict['classifier.bias']
        print("re-initialize fc layer")
        missing_keys = model.load_state_dict(state_dict, strict=False)
    else:
        missing_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys from checkpoint ",missing_keys.missing_keys)
    print("Unexpected keys in network : ",missing_keys.unexpected_keys)


    main(config, model)


# 200: 0.86 (0.852 10 epochs) known, 0.983 (0.872 10 epochs) unknown
#  20: 0.72 (0.772 10 epochs) known, 0.986 (0.989 10 epochs) unknown
# [0.8, 0.72, 0.88, 0.8, 0.64, 0.88, 0.64, 0.52, 0.88, 0.88, 0.92, 0.8, 0.6, 0.92, 0.72, 0.68, 0.96, 0.68, 0.8, 0.72, 0.9893333333333333]
# 20 only 96.37

# 20 + 1: 0.534 0.98
# 20 + 2: 0.486 0.981
# 20 + 4: 0.474 0.965
# 20 + 10: 0.528 0.942
# 20 + 50: 0.588 0.86
# 20 + 100: 0.546 0.89


