from ultralytics import YOLO
import os
import torch
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
import argparse

import matplotlib.pyplot as plt
from scipy.stats import entropy
import json
from ultralytics.nn.modules.head import Classify
from collections import defaultdict
from sklearn.mixture import BayesianGaussianMixture


class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out


def plot_roc(in_data, out_data, label):
    labels = [0] * len(torch.tensor(in_data).squeeze()) + [1] * len(torch.tensor(out_data).squeeze())
    data = np.concatenate((torch.tensor(in_data).squeeze(), torch.tensor(out_data).squeeze()))
    fpr, tpr, thresholds = metrics.roc_curve(labels, data)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold for {label}: {optimal_threshold}, TPR: {tpr[optimal_idx]}, FPR: {fpr[optimal_idx]}")
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    plt.plot(fpr,tpr,label=f"{label}, AUC="+str(np.round(roc_auc, 3)))
    # display.plot()
    plt.xlabel("False positive rate", fontsize=16)
    plt.ylabel("True positive rate", fontsize=16)

def visualize_tsne(embeddings, labels, title, output_path=""):
    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Generate unique colors for each class
    num_classes = len(np.unique(labels))
    colors = plt.cm.get_cmap('plasma', num_classes)

    # Plot t-SNE embeddings with color-coded classes
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(np.unique(labels)):
        class_indices = np.where(labels == class_name)[0]
        plt.scatter(tsne_embeddings[class_indices, 0], tsne_embeddings[class_indices, 1], color=colors(i), label=class_name)

    # Adjust text size
    plt.title(title, fontsize=20)
    plt.xlabel('t-SNE Component 1', fontsize=18)
    plt.ylabel('t-SNE Component 2', fontsize=18)
    if num_classes < 10:
        plt.legend(fontsize=16)
    
    plt.savefig(os.path.join(output_path, f"{title}.png"))
    plt.close()


def main(args):


    model = YOLO(args.model_path)
    print(model.model)
    dataset_path = args.dataset_path
    output_path = os.path.join("eval_outputs", args.output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    classify_hook = SaveIO()
    for i, module in enumerate(model.model.modules()):
        if type(module) is Classify:

            print(module)
            module.linear.register_forward_hook(classify_hook)

    train_probs = defaultdict(list)
    train_logits = defaultdict(list)

    for class_name in os.listdir(os.path.join(dataset_path, "train")):
        class_path = os.path.join(dataset_path, "train", class_name)
        for i, image_name in enumerate(os.listdir(class_path)):
            image_path = os.path.join(class_path, image_name)
            result = model(image_path, embed=[])[0]
            logits = classify_hook.output
            # print(logits.shape)
            train_logits[class_name].append(logits)
            train_probs[class_name].append(result.probs.data)
            
            if args.limit_num_images:
                if i == 100:
                    break

    train_probs_concat = []
    train_logits_concat = []
    train_labels = []

    for class_name, probs_list in train_probs.items():
        train_probs_concat.extend(probs_list)
        #print(train_logits[class_name])
        train_logits_concat.extend(train_logits[class_name])
        train_labels.extend([class_name] * len(probs_list))

    train_probs_concat = torch.stack(train_probs_concat).cpu().numpy()
    train_logits_concat = torch.stack(train_logits_concat).cpu().numpy().squeeze()

    visualize_tsne(train_logits_concat, np.array(train_labels), "train_logits_embeddings_tsne", output_path)
    visualize_tsne(train_probs_concat, np.array(train_labels), "train_probs_embeddings_tsne", output_path)

    # Read unknown classes from JSON file
    with open(os.path.join(dataset_path, "known_unknown_classes.json"), "r") as f:
        classes = json.load(f)
        known_classes = classes["known"]
        unknown_classes = classes["unknown"]

    classes_mean = []

    for class_name in train_probs:
        class_probs = torch.stack(train_probs[class_name])
        classes_mean.append(torch.mean(class_probs, dim=0))

    classes_mean = torch.stack(classes_mean).squeeze()
    #np.savez("classes_mean_10.npz", classes_mean=classes_mean.cpu().numpy(), known_classes=known_classes)

    GMMs = []
    for class_name in train_logits:
        GMM = BayesianGaussianMixture(n_components = 3, random_state=42).fit(torch.stack(train_logits[class_name]).squeeze().cpu().numpy())
        #GMMs.append({"means": GMM.means_, "covariances": GMM.covariances_, "weights": GMM.weights_,
        #            "degrees_of_freedom": GMM.degrees_of_freedom_, "mean_precision": GMM.mean_precision_,
        #            "weight_concentration": GMM.weight_concentration_})
        GMMs.append(GMM)
        #print(GMM.predict_proba(classes_feats[i]).shape)

    # np.savez("classes_mean_gmms.npz", classes_mean=classes_mean.cpu().numpy(), known_classes=known_classes, gmm_params = GMMs)

    known_max_confs = []
    unknown_max_confs = []

    known_entropy = []
    unknown_entropy = []

    known_max_logit = []
    unknown_max_logit = []

    known_cluster_center_dist = []
    unknown_cluster_center_dist = []

    known_gmm_score = []
    unknown_gmm_score = []


    known_val_logits = []
    unknown_val_logits = []
    known_val_probs = []
    unknown_val_probs = []
    known_val_labels = []

    predicted_unknown_paths = []

    for split in [args.split, "unknown_classes"]:

        for class_name in os.listdir(os.path.join(dataset_path, split)):
            class_path = os.path.join(dataset_path, split, class_name)
            for i, image_name in enumerate(os.listdir(class_path)):
                image_path = os.path.join(class_path, image_name)
                result = model(image_path, embed=[])[0]
                max_conf = result.probs.top1conf
                predicted_class = result.probs.data.cpu().numpy().argmax()
                entropy_value = entropy(result.probs.data.cpu().numpy())

                logits = classify_hook.output

                cluster_center_dists = torch.norm(result.probs.data - classes_mean, dim=1)
                closest_cluster_dist = cluster_center_dists[predicted_class]

                gmm_scores = np.zeros(len(GMMs))
                for j in range(len(GMMs)):
                    gmm_score = GMMs[j].score(logits.cpu().numpy().reshape(1, -1))
                    gmm_scores[j] = gmm_score

                gmm_score = np.mean(gmm_scores)
                max_logit = torch.max(logits)

                if args.decision_criteria == "max_conf":
                    if max_logit < args.max_logit_threshold:
                        predicted_unknown_paths.append(image_path)
                elif args.decision_criteria == "entropy":
                    if entropy_value > args.entropy_threshold:
                        predicted_unknown_paths.append(image_path)

                if split == "unknown_classes":
                    unknown_max_confs.append(max_conf)
                    unknown_entropy.append(entropy_value)
                    unknown_max_logit.append(max_logit)
                    unknown_cluster_center_dist.append(closest_cluster_dist)
                    unknown_gmm_score.append(gmm_score)

                    unknown_val_logits.append(logits)
                    unknown_val_probs.append(result.probs.data)
                    #unknown_val_labels.append(class_name)
                else:
                    known_max_confs.append(max_conf)
                    known_entropy.append(entropy_value)
                    known_max_logit.append(max_logit)
                    known_cluster_center_dist.append(closest_cluster_dist)
                    known_gmm_score.append(gmm_score)

                    known_val_logits.append(logits)
                    known_val_probs.append(result.probs.data)
                    known_val_labels.append(class_name)
                
                # if args.limit_num_images:
                #     if i == 50:
                #         break

    

    known_val_logits = torch.stack(known_val_logits).cpu().numpy().squeeze()
    unknown_val_logits = torch.stack(unknown_val_logits).cpu().numpy().squeeze()
    known_val_probs = torch.stack(known_val_probs).cpu().numpy().squeeze()
    unknown_val_probs = torch.stack(unknown_val_probs).cpu().numpy().squeeze()
    visualize_tsne(np.concatenate((known_val_logits, unknown_val_logits)), np.array(known_val_labels + ["unknown"] * len(unknown_val_logits)), "val_logits_embeddings_tsne", output_path)
    visualize_tsne(np.concatenate((known_val_probs, unknown_val_probs)), np.array(known_val_labels + ["unknown"] * len(unknown_val_probs)), "val_probs_embeddings_tsne", output_path)

    plot_roc(unknown_max_confs, known_max_confs, "MSP")
    plot_roc(known_entropy, unknown_entropy, "entropy")
    plot_roc(unknown_max_logit, known_max_logit, "MLS")
    plot_roc(unknown_cluster_center_dist, known_cluster_center_dist, "CCD")
    plot_roc(known_gmm_score, unknown_gmm_score, "gmm_score")

    plt.legend(fontsize=14)
    plt.title("ROC Curves", fontsize=18)
    plt.savefig(os.path.join(output_path, "roc_curves.png"))


    if args.save_pred_unknown:
        # Copy the predicted unknown images to a new directory
        print(len(predicted_unknown_paths))
        save_path = os.path.join(output_path, "predicted_unknown")
        if os.path.exists(save_path):
            os.system(f"rm -r {save_path}")
        os.makedirs(save_path)
        for image_path in predicted_unknown_paths:
            image_name = image_path.split("/")[-1]
            os.system(f'cp "{image_path}" {os.path.join(save_path, image_name)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a YOLO model.')
    parser.add_argument('--model_path', type=str, help='Model path')
    parser.add_argument('--dataset_path', type=str, help='Dataset path')
    parser.add_argument('--split', type=str, default="val", help='Split to evaluate on')
    parser.add_argument('--output_path', type=str, help='Output path')
    parser.add_argument('--save_pred_unknown', action='store_true', help='Save predicted unknown images')
    parser.add_argument('--limit_num_images', action='store_true', help='Limit number of images to evaluate')
    parser.add_argument('--max_logit_threshold', type=float, default=5.055, help='Max logit threshold for unknown class prediction')
    parser.add_argument('--entropy_threshold', type=float, default=0.998, help='Entropy threshold for unknown class prediction')
    parser.add_argument('--decision_criteria', choices=["max_conf", "entropy"], default="max_conf", help='Decision criteria for unknown class prediction')

    args = parser.parse_args()

    main(args)
    