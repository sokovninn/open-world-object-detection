
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
70: train64
70 + 30 synth (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train71 (0.841)
70 + 30 synth3x (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train88 (0.864)
70 + clip50 (label smoothing = 0.1, pre 70, freeze 9, 10 epoch, ): train90 (0.918)
70 + clip (label smoothing = 0.1, pre 70, 10 epoch, ): train101 (0.963)
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

```
python eval_yolo.py --model_path runs/classify/train98/weights/best.pt --dataset_path datasets/tiny-imagenet-200_90 --output_path tiny-imagenet_90 --limit_num_images --save_pred_unknown --decision_criteria entropy
```
Optimal threshold for MSP: 0.7040815949440002, TPR: 0.6007777777777777, FPR: 0.196
Optimal threshold for entropy: 0.9983394145965576, TPR: 0.8264, FPR: 0.4073333333333333
Optimal threshold for MLS: 11.277005195617676, TPR: 0.5765555555555556, FPR: 0.2045
Optimal threshold for CCD: 1.0505118370056152, TPR: 0.5581111111111111, FPR: 0.1688
Optimal threshold for gmm_score: -341257410.6266379, TPR: 0.6643, FPR: 0.49033333333333334
