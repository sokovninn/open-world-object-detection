
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
full(20 epochs): train51
70 + 30 synth (label smoothing = 0.1, pre): train60
70 + 30 synth (label smoothing = 0.1, pre 70): train65
70: train64
70 + 30 synth (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train71 (0.841)
70 + 30 synth3x (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train88 (0.864)
70 + clip50 (label smoothing = 0.1, pre 70, freeze 9, 10 epoch, ): train90 (0.918)
70 + clip full (label smoothing = 0.1, pre 70, freeze 9, 10 epoch, ): train92 (0.968)
yolo classify val model=runs/classify/train90/weights/best.pt data=datasets/imagenette320 split=val