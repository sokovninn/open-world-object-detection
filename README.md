
# Master Thesis


## Experiments:
1. Train the model with 20, 40, 60 and 80% of the classes removed from the CIFAR-100 dataset. How does the accuracy change?
2. Set explicit unknown and train the model with 20, 40, 60 and 80% of the classes removed from the CIFAR-100 dataset. How does the accuracy change?
3. Check AUROC for known and unknown classes.
4. Compare mulitple unknown detection methods.
5. Visualize the embeddings of the known and unknown classes.
6. Generate a dataset with datadreamer for 20 unknown classes (40,60, 80) and train the model with the generated dataset.
7. Try automatic unknown classes names retrieval.
8. Generate a dataset with datadreamer for retreived unknown classes and train the model with the generated dataset. 
9. Try different unknown 
10. Try CLIP in CIFA-100 dataset.
11. Try label smoothing.
12. Try equal porb for unknown classes during training.
13. Try 224x224 images.
14. Diffferent amount of syntethic images

## CIFAR-100
Full: train30
80: train31
60: train37
40: train40
80 + 20 synth: train35 (80 pre). train39 (imagenet pre), 41(128)


## Imagenette
full(20 epochs): train51 (0.977)
70 + 30 synth (label smoothing = 0.1, pre): train60
70 + 30 synth (label smoothing = 0.1, pre 70): train65
70: train64 (0.636 full)
70 + 30 synth (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train71 (0.841)
70 + 30 synth3x (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train88 (0.864)
70 + clip50 (label smoothing = 0.1, pre 70, freeze 9, 10 epoch, ): train90 (0.918)
70 + clip (label smoothing = 0.1, pre 70, 10 epoch, ): train101 (0.963)
70 + clip + synth3x (label smoothing = 0.1, pre 70, 10 epoch, ): train10 (0.958) mixup 0.959 train15
yolo classify  train data= datasets/imagenette320_70_30_synth3x_merged_unk_clip/ model=runs/classify/train64/weights/best.pt epochs=10 imgsz=256 label_smoothing=0.1 batch=8 workers=8 amp=False
yolo classify val model=runs/classify/train90/weights/best.pt data=datasets/imagenette320 split=val

'''
python eval_yolo.py --model_path runs/classify/train64/weights/best.pt --dataset_path datasets/imagenette320_70 --output_path imagenette_70 --decision_criteria max_conf --save_pred_unknown
'''

Optimal threshold for MSP: 0.986992597579956, TPR: 0.8795792528110264, FPR: 0.05187319884726225
Optimal threshold for entropy: 0.06933984160423279, TPR: 0.9560518731988472, FPR: 0.12622415669205658
Optimal threshold for MLS: 5.0547871589660645, TPR: 0.8980776206021037, FPR: 0.05475504322766571
Optimal threshold for CCD: 1.3982793092727661, TPR: 0.8774029742473703, FPR: 0.04755043227665706
Optimal threshold for gmm_score: -31.665662851713183, TPR: 0.8627521613832853, FPR: 0.1458106637649619

## Tiny Imagenet
full: train97 (0.706 0.896)
90(180 classes): train98 (0.72 0.905)
90 + 10 synth: train100 (0.641 0.824)
90 + clip: train102 (0.694 0.892)
90 + clip + synth: train14 (0.693 0.887)
yolo classify  train data=datasets/tiny-imagenet-200_90_synth_merged_unk_clip/ model=runs/classify/train98/weights/best.pt epochs=10 imgsz=256 label_smoothing=0.1 batch=64 workers=8 amp=False


```
python eval_yolo.py --model_path runs/classify/train98/weights/best.pt --dataset_path datasets/tiny-imagenet-200_90 --output_path tiny-imagenet_90 --limit_num_images --save_pred_unknown --decision_criteria entropy
```
Optimal threshold for MSP: 0.7040815949440002, TPR: 0.6007777777777777, FPR: 0.196
Optimal threshold for entropy: 0.9983394145965576, TPR: 0.8264, FPR: 0.4073333333333333
Optimal threshold for MLS: 11.277005195617676, TPR: 0.5765555555555556, FPR: 0.2045
Optimal threshold for CCD: 1.0505118370056152, TPR: 0.5581111111111111, FPR: 0.1688
Optimal threshold for gmm_score: -341257410.6266379, TPR: 0.6643, FPR: 0.49033333333333334

## Tiny Imagenet Full Size 
full: train2 (0.848 0.964)
0.44 256 train -> 64 val, 0.594 64 train -> 256 val
90(180 classes): train5 (0.856 0.969)
90 + 10 synth: train7 (0.77 0.907)


## VOC PASCAL
full: train2 (0.816 map-50) yolo detect train data=VOC.yaml model=yolov8n.pt epochs=40 imgsz=640 label_smoothing=0.1
15 classes: train9 (0.835 map-50, 0.596 full) yolo detect train data=datasets/VOC_15/VOC.yaml model=yolov8n.pt epochs=40 imgsz=640 label_smoothing=0.1 (?? unknown 0.2 entropy)
15 + 5 synth: train12 (0.781 map-50 synth, 0.715 real val4) yolo detect train data=datasets/VOC_15/VOC_15real_5synth.yaml model=runs/detect/train9/weights/best.pt epochs=20 imgsz=640 label_smoothing=0.1
15 + 5 owlv2: train13 (0.706 map-50 owlv2, 0.764 real val4) yolo detect train data=datasets/VOC_15/VOC_15real_5owlv2.yaml model=runs/detect/train9/weights/best.pt epochs=20 imgsz=640 label_smoothing=0.1

15 + 5 synth owlv2: train14 (0.77 map-50 synth+owlv2, 0.794 full) yolo detect train data=datasets/VOC_15/VOC_15real_5synth_owlv2.yaml model=runs/detect/train9/weights/best.pt epochs=20 imgsz=640 label_smoothing=0.1

15 + 5 synth owlv2: train16 (0.777 map-50 synth+owlv2, 0.795 full) yolo detect train data=datasets/VOC_15/VOC_15real_5synth_owlv2.yaml model=runs/detect/train9/weights/best.pt epochs=30 imgsz=640 label_smoothing=0.1

15 + 5 synth: train17 (0.752 map-50 synth, 0.685 full) yolo detect train data=datasets/VOC_15/VOC_15real_5synth_light.yaml model=runs/detect/train9/weights/best.pt epochs=20 imgsz=640 label_smoothing=0.1

15 + 5 real synth: train18 (0.791 map-50 real synth, 0.827 real) yolo detect train data=datasets/VOC/VOC_synth.yaml model=runs/detect/train9/weights/best.pt epochs=20 imgsz=640 label_smoothing=0.1




