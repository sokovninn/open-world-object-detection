from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8n-cls.yaml')  # build a new model from YAML
model = YOLO('yolov8n-cls.pt') # load a pretrained model (recommended for training)
#model = YOLO('runs/classify/train31/weights/best.pt')  
#model = YOLO('yolov8n-cls.yaml').load('yolov8n-cls.pt')  # build from YAML and transfer weights
#model = YOLO('runs/classify/train64/weights/best.pt')  # load a model from a specific run
#model = YOLO('runs/classify/train98/weights/best.pt')  # load a model from a specific run
#model = YOLO('runs/classify/train5/weights/best.pt')

# Train the model
#results = model.train(data='datasets/cifar100_100', imgsz=128, epochs=20, batch=256, label_smoothing=0.1, workers=8) #100, imgsz=32, batch=4096, workers=8
#results = model.train(data='datasets/cifar100_80_20_synth32_merged', imgsz=128, epochs=20, batch=256, label_smoothing=0.1, workers=8) #100, imgsz=32, batch=4096, workers=8
#results = model.train(data='datasets/cifar100_40', imgsz=128, epochs=20, batch=256, label_smoothing=0.1, workers=8) #100, imgsz=32, batch=4096, workers=8
#results = model.train(data='datasets/cifar100_80_20_synth32aa_merged', imgsz=128, epochs=20, batch=256, label_smoothing=0.1, workers=8) #100, imgsz=32, batch=4096, workers=8
#results = model.train(data='datasets/imagenette320', imgsz=256, epochs=20, batch=64, label_smoothing=0.1, workers=8)
#results = model.train(data='datasets/imagenette320_70_30_synth_merged', imgsz=256, epochs=10, batch=8, label_smoothing=0.1, workers=8, amp=False) #100, imgsz=32, batch=4096, workers=8
#results = model.train(data='datasets/imagenette320_70', imgsz=256, epochs=10, batch=8, label_smoothing=0.1, workers=8, amp=False)
#results = model.train(data='datasets/imagenette320_70_30_synth_merged', imgsz=256, epochs=1, batch=8, label_smoothing=0.1, workers=8, amp=False, freeze=9)
#results = model.train(data='datasets/imagenette320_70_30_synth3x_merged', imgsz=256, epochs=1, batch=8, label_smoothing=0.1, workers=8, amp=False, freeze=9)
#results = model.train(data='datasets/imagenette320_70_unk_clip', imgsz=256, epochs=10, batch=8, label_smoothing=0.1, workers=8, amp=False)
#results = model.train(data='datasets/tiny-imagenet-200_90', imgsz=256, epochs=10, batch=64, label_smoothing=0.1, workers=8, amp=False)
#results = model.train(data='datasets/tiny-imagenet-200_90_synth_merged', imgsz=256, epochs=5, batch=64, label_smoothing=0.1, workers=8, amp=False, freeze=9)
#results = model.train(data='datasets/tiny-imagenet-200_90_unk_clip', imgsz=256, epochs=10, batch=64, label_smoothing=0.1, workers=8, amp=False)
#results = model.train(data='datasets/tin_full_size', imgsz=256, epochs=10, batch=64, label_smoothing=0.1, workers=8, amp=False)
#results = model.train(data='datasets/tin_full_size_90', imgsz=256, epochs=10, batch=64, label_smoothing=0.1, workers=8, amp=False)
#results = model.train(data='datasets/tin_full_size_90_synth_merged', imgsz=256, epochs=1, batch=64, label_smoothing=0.1, workers=8, amp=False, freeze=9)
results = model.train(data='datasets/tin_20_synth', imgsz=256, epochs=10, batch=64, label_smoothing=0.1, workers=8, amp=False)

#print(model)



# Experiments:
# 1. Train the model with 20, 40, 60 and 80% of the classes removed from the CIFAR-100 dataset. How does the accuracy change?
# 2. Set explicit unknown and train the model with 20, 40, 60 and 80% of the classes removed from the CIFAR-100 dataset. How does the accuracy change?
# 3. Check AUROC for known and unknown classes.
# 4. Compare mulitple unknown detection methods.
# 5. Visualize the embeddings of the known and unknown classes.
# 6. Generate a dataset with datadreamer for 20 unknown classes (40,60, 80) and train the model with the generated dataset.
# 7. Try automatic unknown classes names retrieval.
# 8. Generate a dataset with datadreamer for retreived unknown classes and train the model with the generated dataset. 
# 9. Try different unknown 
# 10. Try CLIP in CIFA-100 dataset.
# 11. Try label smoothing.
# 12. Try equal porb for unknown classes during training.
# 13. Try 224x224 images.
# 14. Diffferent amount of syntethic images

# Full: train30
# 80: train31
# 60: train37
# 40: train40
# 80 + 20 synth: train35 (80 pre). train39 (imagenet pre), 41(128)


# Imagenette
# full(20 epochs): train51
# 70 + 30 synth (label smoothing = 0.1, pre): train60
# 70 + 30 synth (label smoothing = 0.1, pre 70): train65
# 70: train64
#70 + 30 synth (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train71 (0.841)
#70 + 30 synth3x (label smoothing = 0.1, pre 70, freeze 9, 1 epoch, ): train88 (0.864)
#70 + clip50 (label smoothing = 0.1, pre 70, freeze 9, 10 epoch, ): train90 (0.918)
#70 + clip full (label smoothing = 0.1, pre 70, freeze 9, 10 epoch, ): train92 (0.968)
# yolo classify val model=runs/classify/train90/weights/best.pt data=datasets/imagenette320 split=val



# More Freeze and less epochs for synthetic data
