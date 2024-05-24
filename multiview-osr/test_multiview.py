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
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from math import factorial
from sklearn import metrics



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

def run_model(model, loader, softmax=False):
    #run the resnet model
    total = 0
    out_list = []
    tgt_list = []
    cls_list = []
    for images, target in tqdm(loader):
        total += images.size(0)
        images = images.cuda()
        output, classifier = model(images,feat_cls=True)

        out_list.append(output.data.cpu())
        cls_list.append(F.softmax(classifier, dim=1).data.cpu())
        tgt_list.append(target)

    return  torch.cat(out_list), torch.cat(tgt_list), torch.cat(cls_list)


def plot_roc(in_data, out_data, label):
    labels = [0] * len(torch.tensor(in_data).squeeze()) + [1] * len(torch.tensor(out_data).squeeze())
    data = np.concatenate((torch.tensor(in_data).squeeze(), torch.tensor(out_data).squeeze()))
    fpr, tpr, thresholds = metrics.roc_curve(labels, data)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    plt.plot(fpr,tpr,label=f"{label}, AUC="+str(np.round(roc_auc, 3)))
    # display.plot()
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")


def main(opt, model):
    if opt.cuda:
        model = model.cuda()
    model.eval()

    with open("tinyimagenet_classes.json") as f:
        idx2label = eval(f.read())
    
    dataset = MultiviewDataset("/home/nikita/broca_multiview_dataset", image_size=(opt.image_size, opt.image_size), each_n=opt.each_n, n_views=opt.n_views, seed=opt.seed)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    pdist = torch.nn.PairwiseDistance(p=2)

    #classes_mean_info = np.load("classes_mean_10.npz")

    # GMMs = []
    # classes_mean_info = np.load("classes_mean_10_gmms.npz", allow_pickle=True)
    # for i in range(10):
    #     GMM = BayesianGaussianMixture(n_components = 3)

    #     GMM.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(classes_mean_info["gmm_params"][i]["covariances"]))
    #     GMM.weights_ = classes_mean_info["gmm_params"][i]["weights"]
    #     GMM.means_ = classes_mean_info["gmm_params"][i]["means"]
    #     GMM.covariances_ = classes_mean_info["gmm_params"][i]["covariances"]
    #     GMM.degrees_of_freedom_ = classes_mean_info["gmm_params"][i]["degrees_of_freedom"]
    #     GMM.mean_precision_ = classes_mean_info["gmm_params"][i]["mean_precision"]
    #     GMM.weight_concentration_ = classes_mean_info["gmm_params"][i]["weight_concentration"]
    #     #print(classes_mean_info["gmm_params"][i])
    #     GMMs.append(GMM)



    # classes_mean = torch.tensor(classes_mean_info["classes_mean"])
    # #classes_mean = torch.diag(torch.Tensor([10 for i in range(10)])).cuda()	
    # known_classes = classes_mean_info["known_classes"]
    # print(classes_mean, known_classes)

    image_num = 0
    correct_num = 0
    in_emb = []
    out_emb = []
    in_sfmx = []
    out_sfmx = []
    in_logit = []
    out_logit = []
    in_dist = []
    out_dist = []


    in_entropy = []
    out_entropy = []

    in_gmm = []
    out_gmm = []


    for images, target in tqdm(dataloader):

        images = images.squeeze(0)
        if opt.cuda:
            images = images.cuda()
        #print(images.shape)
        #outputs = []
        cls_list = []
        out_list = []
        logit_list = []
        for image in images:
            output, classifier = model(image.unsqueeze(0),feat_cls=True)  #image.unsqueeze(0)
            #outputs.append(classifier)
            #print(classifier.shape)
            cls_list.append(F.softmax(classifier, dim=1).data.cpu())
            out_list.append(output.data.cpu())
            logit_list.append(classifier.data.cpu())



        outputs = torch.cat(out_list, 0)
        #dists = euclidean_dist(outputs, classes_mean)
        #print("dists", dists.shape)
        #print(torch.max(dists[:,torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))]))
        #if opt.fusion_method == "mean":
        #    cluster_dist = float(torch.mean(dists[:,torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))]))
        #elif opt.fusion_method == "max":
        #    cluster_dist = float(torch.max(dists[:,torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))]))
        # elif opt.fusion_method == "sum":
        #     cluster_dist = float(torch.sum(dists[:,torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))]))
        
        print(cls_list)
        _, s, _ = np.linalg.svd(torch.stack(cls_list, 0).T)

        print(s, s.shape, np.sum(s))
        s /= np.sum(s)
        # s_e = torch.stack(cls_list, 0)
        # #print(s.shape)
        # s_e = s / s.shape[0]
        # s_e = np.sum(s, 0).T
        # print(s.shape)
        # print(s)
        # print(entropy(s.T))
        # s = s + s_e
        # s /= np.sum(s)

        #print(s, np.sum(s))
        print(cls_list[0])

        #s = s / np.log2(factorial(10))

        print(torch.stack(cls_list, 0).T.shape)

        gmm_scores = np.zeros(10)
        #for i in range(10):
            #print(GMMs[i].means_.shape)
            #print(GMMs[i].covariances_)
            #print(torch.stack(out_list, 0).shape)
            #_, sg, _ = np.linalg.svd(torch.stack(out_list, 0).T)
            #print("S: ", sg.shape)
            #print(torch.stack(out_list, 0).squeeze(1).shape)
            #gmm_score = GMMs[i].score(sg.T)
            #gmm_score = GMMs[i].score(torch.stack(out_list, 0).squeeze(1))
            #gmm_score_samples = GMMs[i].score_samples(torch.stack(out_list, 0).squeeze(1))
            # print(gmm_score)
            #print(GMMs[i].score(torch.stack(out_list, 0).squeeze()))
            #gmm_scores[i] = np.min(gmm_score_samples)
            #gmm_scores[i] = gmm_score

            #epsilon = 0.005    
            #print(GMMs[i].predict_proba([GMMs[i].means_[0]+epsilon]))

        # gmm_scores = np.zeros(1)
        # gmm_scores[0] = GMMs[torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))].score(torch.stack(out_list, 0).squeeze(1))
        print("Target: ", target)

        if target[0] in known_classes_tiny:
            print("Known")
            print(entropy(s))
            in_entropy.append(entropy(s))

            in_gmm.append(np.mean(gmm_scores))

            image_num += 1
            if int(torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))) in known_classes_tiny[target[0]]:
                correct_num += 1
                #print("correct")
            # else:
            #    #  print(idx2label[int(torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0)))])
            # print(idx2label[str(int(torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))))][1][0])
            # print(target)
            in_emb.append(torch.mean(torch.stack(out_list, 0), dim=0))
            in_sfmx.append(torch.mean(torch.stack(cls_list, 0), dim=0))
            in_logit.append(torch.mean(torch.stack(logit_list, 0), dim=0))
            #in_dist.append(cluster_dist)
        else:
            print("Unknown")
            print(entropy(s))
            out_entropy.append(entropy(s))

            # out_gmm.append(np.mean(gmm_scores))
            out_gmm.append(np.mean(gmm_scores))

            # print(idx2label[int(torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0)))])
            # print(idx2label[str(int(torch.argmax(torch.mean(torch.stack(cls_list, 0), dim=0))))][1][0])
            # print(target)
            out_emb.append(torch.mean(torch.stack(out_list, 0), dim=0))
            out_sfmx.append(torch.mean(torch.stack(cls_list, 0), dim=0))
            out_logit.append(torch.mean(torch.stack(logit_list, 0), dim=0))
            #out_dist.append(cluster_dist)

    in_emb = torch.stack(in_emb, 0).squeeze()
    out_emb = torch.stack(out_emb, 0).squeeze()
    in_sfmx = torch.stack(in_sfmx, 0).squeeze()
    out_sfmx = torch.stack(out_sfmx, 0).squeeze()
    in_logit = torch.stack(in_logit, 0).squeeze()
    out_logit = torch.stack(out_logit, 0).squeeze()
    print(correct_num, image_num)
    print(f"Accuracy: {float(correct_num) / image_num}")

    if opt.compute_cmeans:
        print('Compute sample mean for training data....')
        known_classes = [92, 135, 163, 172, 177, 184, 186, 187, 188, 193]
        train_dataset = eval("get{}Dataset".format(config.dataset))(image_size=config.image_size, split='train', data_path=config.data_dir, known_classes=known_classes)
        print(len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        train_emb, train_targets, train_sfmx = run_model(model, train_dataloader)
        train_acc = float(torch.sum(torch.argmax(train_sfmx, dim=1) == train_targets)) / len(train_sfmx)

        in_classes = torch.unique(train_targets)
        class_idx = [torch.nonzero(torch.eq(cls, train_targets)).squeeze(dim=1) for cls in in_classes]
        classes_feats = [train_emb[idx] for idx in class_idx]
        classes_mean = torch.stack([torch.mean(cls_feats, dim=0) for cls_feats in classes_feats],dim=0)
        print(classes_mean, classes_mean.shape)
        #np.savez("classes_mean_10.npz", classes_mean=classes_mean.cpu().numpy(), known_classes=known_classes)

        GMMs = []
        for i in range(len(known_classes)):
            print(classes_feats[i].shape)
            GMM = BayesianGaussianMixture(n_components = 3).fit(classes_feats[i])
            GMMs.append({"means": GMM.means_, "covariances": GMM.covariances_, "weights": GMM.weights_,
                         "degrees_of_freedom": GMM.degrees_of_freedom_, "mean_precision": GMM.mean_precision_,
                         "weight_concentration": GMM.weight_concentration_})
            print(GMM.predict_proba(classes_feats[i]).shape)

        np.savez("classes_mean_10_gmms.npz", classes_mean=classes_mean.cpu().numpy(), known_classes=known_classes, gmm_params = GMMs)


    # else:
    #     classes_mean_info = np.load("classes_mean_10.npz")
    #     classes_mean = torch.tensor(classes_mean_info["classes_mean"])
    #     known_classes = classes_mean_info["known_classes"]
    # print(classes_mean, known_classes)

    if opt.vis_features:
        embs = np.concatenate([classes_mean, in_emb, out_emb], axis=0)
        print(embs.shape)
        targets = np.concatenate([np.zeros(classes_mean.shape[0]), np.ones(in_emb.shape[0]), np.ones(out_emb.shape[0]) * 2], axis=0)

        X_2d = TSNE(n_components=2, learning_rate='auto', early_exaggeration = 12, init='pca', perplexity=20, n_iter=10000).fit_transform(embs)
        print(X_2d.shape, targets.shape)

    
        cluster_centres = X_2d[targets == 0, :]
        in_features =  X_2d[targets == 1, :]
        out_features = X_2d[targets == 2, :]
        plt.scatter(x=cluster_centres[:, 0], y=cluster_centres[:, 1], marker="D", label="Cluster center")
        for i, txt in enumerate(known_classes_tiny.keys()):
            plt.annotate(txt, (cluster_centres[i, 0], cluster_centres[i, 1]))
        plt.scatter(x=out_features[:, 0], y=out_features[:, 1], marker=".", label="Unknown")
        plt.scatter(x=in_features[:, 0], y=in_features[:, 1], marker=".", label="Known")

        plt.legend()
        plt.savefig("clusters.png")
        plt.show()
    results = {}
    results["acc"] = float(correct_num) / image_num

    # results["CCD"] = get_roc_sklearn(in_dist, out_dist)
    #plot_roc(in_dist, out_dist, "CCD")

    in_score, _ = torch.max(in_sfmx, dim=1)
    out_score, _ = torch.max(out_sfmx, dim=1)
    results["MSP"] = get_roc_sklearn(out_score.cpu().numpy(), in_score.cpu().numpy())
    #plot_roc(out_score, in_score, "MSP")

    in_score, _ = torch.max(in_logit, dim=1)
    out_score, _ = torch.max(out_logit, dim=1)
    results["MLS"] = get_roc_sklearn(out_score.cpu().numpy(), in_score.cpu().numpy())
    #plot_roc(out_score, in_score, "MLS")

    in_score = entropy(in_sfmx, axis=1)
    out_score = entropy(out_sfmx, axis=1)
    results["entropy"] = get_roc_sklearn(in_score, out_score)
    #plot_roc(in_score, out_score, "Entropy")


    print("Matrix entropy")
    print(get_roc_sklearn(in_entropy, out_entropy))
    print(torch.tensor(in_entropy).shape, in_score.shape)
    print(torch.tensor(out_entropy).shape, out_score.shape)
    results["matrix_entropy"] = get_roc_sklearn(torch.tensor(in_entropy).squeeze(), torch.tensor(out_entropy).squeeze())
    print(results["matrix_entropy"])
    #plot_roc(in_entropy, out_entropy, "SVD entropy")

    # labels = [0] * len(torch.tensor(in_entropy).squeeze()) + [1] * len(torch.tensor(out_entropy).squeeze())
    # data = np.concatenate((torch.tensor(in_entropy).squeeze(), torch.tensor(out_entropy).squeeze()))
    # fpr, tpr, thresholds = metrics.roc_curve(labels, data)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    # plt.plot(fpr,tpr,label=f"SVD entropy ({opt.n_views} views), AUC="+str(roc_auc))
    # display.plot()

    # if opt.n_views == 20:
    #     plt.xlabel("False positive rate")
    #     plt.ylabel("True positive rate")
    #     plt.legend()
        
    #     plt.show()


    print("GMM score")
    #print(get_roc_sklearn(in_entropy, out_entropy))
    #print(torch.tensor(in_entropy).shape, in_score.shape)
    #print(torch.tensor(out_entropy).shape, out_score.shape)
    # results["gmm_score"] = get_roc_sklearn(-torch.tensor(in_gmm).squeeze(), -torch.tensor(out_gmm).squeeze())
    # print(results["gmm_score"])
    #plot_roc(-torch.tensor(in_gmm), -torch.tensor(out_gmm), "GMM score")


    #plt.legend()
    #plt.show()


    # if opt.cls_method == "CCD":
    #     auroc = get_roc_sklearn(in_dist, out_dist)
    # elif opt.cls_method == "MSP":
    #     in_score, _ = torch.max(in_sfmx, dim=1)
    #     out_score, _ = torch.max(out_sfmx, dim=1)
    #     auroc = get_roc_sklearn(out_score.cpu().numpy(), in_score.cpu().numpy())
    # elif opt.cls_method == "MLS":
    #     in_score, _ = torch.max(in_logit, dim=1)
    #     out_score, _ = torch.max(out_logit, dim=1)
    #     auroc = get_roc_sklearn(out_score.cpu().numpy(), in_score.cpu().numpy())
    # elif opt.cls_method == "entropy":
    #     in_score = entropy(in_sfmx, axis=1)
    #     out_score = entropy(out_sfmx, axis=1)
    #     auroc = get_roc_sklearn(in_score, out_score)
    print(results)



    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visual Transformer Train/Fine-tune")

    # basic config
    parser.add_argument("--model-arch", type=str, default="b16", help='model setting to use', choices=['t16','vs16','s16', 'b16', 'b32', 'l16', 'l32', 'h14'])
    parser.add_argument("--checkpoint-path", type=str, default=None, help="model checkpoint to load weights")
    parser.add_argument("--image-size", type=int, default=384, help="input image size", choices=[64, 128, 160, 224, 384, 448])
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
        del state_dict['classifier.weight']
        del state_dict['classifier.bias']
        print("re-initialize fc layer")
        missing_keys = model.load_state_dict(state_dict, strict=False)
    else:
        missing_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys from checkpoint ",missing_keys.missing_keys)
    print("Unexpected keys in network : ",missing_keys.unexpected_keys)

    #main(config, model)
    total_results = {"acc": [], "CCD": [], "MSP": [], "MLS": [], "entropy": [], "matrix_entropy": [], "gmm_score": [], "n_views_list":[5]}

    for n_views in total_results["n_views_list"]:
        config.n_views = n_views
        inter_results = {"acc": [], "CCD": [], "MSP": [], "MLS" : [], "entropy": [], "matrix_entropy": [], "gmm_score": []}
        for i in range(config.exp_reps):
            config.seed = i
            results = main(config, model)
            for key, value in results.items():
                inter_results[key].append(value)
        
        for key, value in inter_results.items():
            total_results[key].append(np.mean(value))


    print(total_results)
    with open('multiview_results_gmm_matrix.json', 'w') as fp:
        json.dump(total_results, fp)

    # with open('multiview_results.json', 'r') as fp:
    #     total_results = json.load(fp)

    # n_views_list = [1,3]
    # total_results = { 'acc': [0.8518518518518519, 0.8888888888888888], 'CCD': [0.8711231556760012, 0.859153869316471], 'MSP': [0.819482083709726, 0.8112014453477868], 'MLS': [0.8417645287563986, 0.8444745558566696], 'entropy': [0.8352905751279736, 0.8452273411623005]}
    plt.plot(total_results["n_views_list"], total_results['CCD'], label='CCD', linestyle="-", marker='o')
    plt.plot(total_results["n_views_list"], total_results['MSP'], label='MSP', linestyle="-", marker='o')
    plt.plot(total_results["n_views_list"], total_results['MLS'], label='MLS', linestyle="-", marker='o')
    plt.plot(total_results["n_views_list"], total_results['entropy'], label='entropy', linestyle="-", marker='o')
    plt.plot(total_results["n_views_list"], total_results['matrix_entropy'], label='matrix_entropy', linestyle="-", marker='o')
    plt.plot(total_results["n_views_list"], total_results['gmm_score'], label='gmm_score', linestyle="-", marker='o')
    #plt.title("Open-set" weight="bold", size=25)
    plt.xlabel('Number of views', color='#333333', size=15)
    plt.ylabel('AUROC', color='#333333', size=15)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.legend(fontsize='large')
    plt.savefig("multiview_plot_auroc.png")
    plt.tight_layout()
    plt.show()

    plt.plot(total_results["n_views_list"], total_results['acc'], linestyle="-", marker='o')
    #plt.title("Open-set" weight="bold", size=25)
    plt.xlabel('Number of views', color='#333333', size=15)
    plt.ylabel('Accuracy', color='#333333', size=15)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.savefig("multiview_plot_acc.png")
    plt.tight_layout()
    plt.show()

# 5
#     0.911269944408346
# {'acc': 0.8421052631578947, 'CCD': 0.8944480542921089, 'MSP': 0.8435492022236661, 'MLS': 0.8941592664789546, 'entropy': 0.904194642986066}

# 10
# 0.9142300194931774
# {'acc': 0.7719298245614035, 'CCD': 0.892570933506606, 'MSP': 0.8519240488051404, 'MLS': 0.8987798714894231, 'entropy': 0.9076600967439173}




