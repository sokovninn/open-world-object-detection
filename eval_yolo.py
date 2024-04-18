from ultralytics import YOLO
import os
import torch
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE

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
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

def visualize_tsne(embeddings, labels, title):
    # Apply t-SNE to reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Generate unique colors for each class
    num_classes = len(np.unique(labels))
    colors = plt.cm.get_cmap('tab10', num_classes)

    # Plot t-SNE embeddings with color-coded classes
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(np.unique(labels)):
        class_indices = np.where(labels == class_name)[0]
        plt.scatter(tsne_embeddings[class_indices, 0], tsne_embeddings[class_indices, 1], color=colors(i), label=class_name)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.close()


#model = YOLO('runs/classify/train64/weights/best.pt')
model = YOLO('runs/classify/train64/weights/best.pt')
dataset_path = "datasets/imagenette320_70/"
#model = YOLO('runs/classify/train31/weights/best.pt')
#dataset_path = "datasets/cifar100_80"

classify = None
cv2_hooks = None
cv3_hooks = None
classify_hook = SaveIO()
for i, module in enumerate(model.model.modules()):
    if type(module) is Classify:
        #module.register_forward_hook(classify_hook)
        # classify = module
        print(module)
        module.linear.register_forward_hook(classify_hook)

        # print(module)
        # cv2_hooks = [SaveIO() for _ in range(module.nl)]
        # cv3_hooks = [SaveIO() for _ in range(module.nl)]
        # for i in range(module.nl):
        #     module.cv2[i].register_forward_hook(cv2_hooks[i])
        #     module.cv3[i].register_forward_hook(cv3_hooks[i])
        # break



train_probs = defaultdict(list)
train_logits = defaultdict(list)

for class_name in os.listdir(os.path.join(dataset_path, "train")):
    class_path = os.path.join(dataset_path, "train", class_name)
    for i, image_name in enumerate(os.listdir(class_path)):
        image_path = class_path + "/" + image_name
        result = model(image_path, embed=[])[0]
        logits = classify_hook.output
        train_logits[class_name].append(logits)
        train_probs[class_name].append(result.probs.data)
        
        # if i == 50:
        #     break

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

visualize_tsne(train_logits_concat, np.array(train_labels), "train_logits_embeddings_tsne")
visualize_tsne(train_probs_concat, np.array(train_labels), "train_probs_embeddings_tsne")



classes_mean = []

for class_name in train_probs:
    class_logits = torch.stack(train_probs[class_name])
    classes_mean.append(torch.mean(class_logits, dim=0))

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

# np.savez("classes_mean_10_gmms.npz", classes_mean=classes_mean.cpu().numpy(), known_classes=known_classes, gmm_params = GMMs)






# Read unknown classes from JSON file
with open(os.path.join(dataset_path, "known_unknown_classes.json"), "r") as f:
    classes = json.load(f)
    known_classes = classes["known"]
    unknown_classes = classes["unknown"]

# results = model("datasets/cifar100_100/test/apple/apple_s_000022.png")


# print(len(results))

# for result in results:
#     print(result.probs)

#dataset_path = "datasets/imagenette320_70/unknown_classes"

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
known_val_labels = []

predicted_unknown_paths = []
max_logit_threshold = 5.54

for class_name in os.listdir(os.path.join(dataset_path, "val")):
    class_path = os.path.join(dataset_path, "val", class_name)
    for i, image_name in enumerate(os.listdir(class_path)):
        image_path = class_path + "/" + image_name
        result = model(image_path, embed=[])[0]
        max_conf = result.probs.top1conf
        predicted_class = result.probs.data.cpu().numpy().argmax()
        entropy_value = entropy(result.probs.data.cpu().numpy())

        #print(result.names)

        #logits = model(image_path, embed = [10])[0]
        logits = classify_hook.output
        known_val_logits.append(logits)
        known_val_labels.append(class_name)


        #dists = euclidean_dist(logits, classes_mean)
        #cluster_dist

        print(logits.shape, classes_mean.shape)
        cluster_center_dists = torch.norm(result.probs.data - classes_mean, dim=1)
        print(cluster_center_dists.shape)
        closest_cluster_dist = cluster_center_dists[predicted_class]
        #closest_cluster_dist = torch.max(cluster_center_dists)
        print("closest_cluster_dist", closest_cluster_dist)
        known_cluster_center_dist.append(closest_cluster_dist)

        gmm_scores = np.zeros(len(GMMs))
        for j in range(len(GMMs)):
            #print(GMMs[i].means_.shape)
            #print(GMMs[i].covariances_)
            #print(torch.stack(out_list, 0).shape)
            gmm_score = GMMs[j].score(logits.cpu().numpy().reshape(1, -1))
            gmm_scores[j] = gmm_score

        gmm_score = np.mean(gmm_scores)
        known_gmm_score.append(gmm_score)

        print(logits.shape)

        max_logit = torch.max(logits)

        if max_logit < max_logit_threshold:
            predicted_unknown_paths.append(image_path)

        print(max_logit)
        #print(classify_hook.output)
        # print(result)
        # print(result.sum())
        # print(result.probs)

        #print(result.probs.top1conf)
        if class_name in unknown_classes:
            unknown_max_confs.append(max_conf)
            unknown_entropy.append(entropy_value)
            unknown_max_logit.append(max_logit)
        else:
            known_max_confs.append(max_conf)
            known_entropy.append(entropy_value)
            known_max_logit.append(max_logit)
        
        # if i == 50:
        #     break

for class_name in os.listdir(os.path.join(dataset_path, "unknown_classes")):
    class_path = os.path.join(dataset_path, "unknown_classes", class_name)
    for i, image_name in enumerate(os.listdir(class_path)):
        image_path = class_path + "/" + image_name
        result = model(image_path, embed=[])[0]
        max_conf = result.probs.top1conf
        predicted_class = result.probs.data.cpu().numpy().argmax()
        entropy_value = entropy(result.probs.data.cpu().numpy())

        #print(result.names)

        #logits = model(image_path, embed = [10])[0]
        logits = classify_hook.output
        unknown_val_logits.append(logits)

        #dists = euclidean_dist(logits, classes_mean)
        #cluster_dist

        cluster_center_dists = torch.norm(result.probs.data - classes_mean, dim=1)
        closest_cluster_dist = cluster_center_dists[predicted_class]
        #closest_cluster_dist = torch.max(cluster_center_dists)
        unknown_cluster_center_dist.append(closest_cluster_dist)

        gmm_scores = np.zeros(len(GMMs))
        for j in range(len(GMMs)):
            #print(GMMs[i].means_.shape)
            #print(GMMs[i].covariances_)
            #print(torch.stack(out_list, 0).shape)
            gmm_score = GMMs[j].score(logits.cpu().numpy().reshape(1, -1))
            gmm_scores[j] = gmm_score

        gmm_score = np.mean(gmm_scores)
        unknown_gmm_score.append(gmm_score)

        print(logits.shape)

        max_logit = torch.max(logits)

        if max_logit < max_logit_threshold:
            predicted_unknown_paths.append(image_path)

        print(max_logit)
        # print(result)
        # print(result.sum())
        # print(result.probs)

        #print(result.probs.top1conf)
        if class_name in unknown_classes:
            unknown_max_confs.append(max_conf)
            unknown_entropy.append(entropy_value)
            unknown_max_logit.append(max_logit)
        else:
            known_max_confs.append(max_conf)
            known_entropy.append(entropy_value)
            known_max_logit.append(max_logit)
        
        # if i == 50:
        #     break

known_val_logits = torch.stack(known_val_logits).cpu().numpy().squeeze()
unknown_val_logits = torch.stack(unknown_val_logits).cpu().numpy().squeeze()
visualize_tsne(np.concatenate((known_val_logits, unknown_val_logits)), np.array(known_val_labels + ["unknown"] * len(unknown_val_logits)), "val_logits_embeddings_tsne")

plot_roc(unknown_max_confs, known_max_confs, "max_conf")
plot_roc(known_entropy, unknown_entropy, "entropy")
plot_roc(unknown_max_logit, known_max_logit, "max_logit")
plot_roc(unknown_cluster_center_dist, known_cluster_center_dist, "cluster_center_dist")
plot_roc(known_gmm_score, unknown_gmm_score, "gmm_score")

plt.legend()
plt.savefig(f"roc.png")


# Copy the predicted unknown images to a new directory
print(len(predicted_unknown_paths))
if os.path.exists("predicted_unknown"):
    os.system("rm -r predicted_unknown")
os.makedirs("predicted_unknown")
for image_path in predicted_unknown_paths:
    image_name = image_path.split("/")[-1]
    os.system(f"cp '{image_path}' predicted_unknown/{image_name}")