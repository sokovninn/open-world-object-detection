from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.autobackend import AutoBackend
import torch
from scipy.stats import entropy
import os
import shutil
from tqdm import tqdm

from utils import non_max_suppression

unknown_classes = [5, 9, 10, 17, 18]

only_pred = False
conf = 0.25
entropy_threshold = 0.2

#args = dict(model='runs/detect/train9/weights/best.pt', data='datasets/VOC/VOC_short.yaml', conf=conf)
args = dict(model='runs/detect/train9/weights/best.pt', data='VOC.yaml', conf=conf) # train15
#args = dict(model='runs/detect/train9/weights/best.pt', data='datasets/VOC_15/VOC_unknown.yaml', conf=conf)
validator = DetectionValidator(args=args)
validator()
print(len(validator.confusion_matrix.matrix))
print(len(validator.confusion_matrix.matrix[0]))

model = AutoBackend(
    weights=validator.args.model,
    device=torch.device('cuda'),
    dnn=validator.args.dnn,
    data=validator.args.data,
    fp16=validator.args.half,
)

model.eval()


print(validator.dataloader)

unknown_image_paths = []


print(model.names)
model.names[20] = "unknown"
validator.init_metrics(model)

print(validator.nc, validator.names)

for batch in tqdm(validator.dataloader, total=len(validator.dataloader)):
    batch = validator.preprocess(batch)
   # print(batch)

    preds = model(batch["img"])
    #print(preds)



    preds, probs = non_max_suppression(
            preds,
            validator.args.conf,
            validator.args.iou,
            labels=validator.lb,
            multi_label=True,
            agnostic=validator.args.single_cls,
            max_det=validator.args.max_det,
    )

    if only_pred:
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                entr = entropy(probs[i][j].cpu().numpy())
                if entr > entropy_threshold:
                    preds[i][j][5] = 20
                    unknown_image_paths.append(batch["im_file"][i])
    else:
        for i in range(len(preds)):
            #print(batch["cls"][batch["batch_idx"] == i])
            for j in range(len(preds[i])):
                entr = entropy(probs[i][j].cpu().numpy())
                #print(batch["cls"])
                #print(entr)

                if entr > entropy_threshold:
                    img_idx = int(batch["batch_idx"][batch["batch_idx"] == i][0].item())
                    #print(img_idx)
                    unknown_image_paths.append(batch["im_file"][img_idx])
                    # Replace class with unknown class (20)
                    preds[i][j][5] = 20

        #print(batch["cls"])

        # Replace unknown classes with index 20 in batch["cls"]
        for i in range(len(batch["cls"])):
            for j in range(len(batch["cls"][i])):
                if batch["cls"][i][j] in unknown_classes:
                    batch["cls"][i][j] = 20

    validator.update_metrics(preds, batch)

validator.finalize_metrics()
validator.print_results()

#print(len(validator.confusion_matrix.matrix))
#print(len(validator.confusion_matrix.matrix[0]))

# Copy images with unknown classes to unknown_classes_test

output_dir = "eval_outputs/VOC_15/"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)

if os.path.exists(os.path.join(output_dir, "predicted_unknown_voc")):
    shutil.rmtree(os.path.join(output_dir, "predicted_unknown_voc"))
os.mkdir(os.path.join(output_dir, "predicted_unknown_voc"))

print(len(unknown_image_paths))
print(unknown_image_paths)
for path in tqdm(unknown_image_paths):
    #os.system(f"cp {path} {os.path.join(output_dir, 'predicted_unknown_voc', os.path.basename(path))}")

    shutil.copy(path, os.path.join(output_dir, "predicted_unknown_voc", os.path.basename(path)))

    
    

    

    
