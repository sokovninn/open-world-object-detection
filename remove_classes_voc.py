import os

dataset_dir = "datasets/VOC_15"

os.mkdir(os.path.join(dataset_dir, "unknown_classes"))
os.mkdir(os.path.join(dataset_dir, "unknown_classes_test"))


classes_to_remove = [5, 9, 10, 17, 18]


images_dir = os.path.join(dataset_dir, "images")

image_folders = ["test2007", "train2007", "val2007", "train2012", "val2012"]

labels_dir = os.path.join(dataset_dir, "labels")

for image_folder in image_folders:
    labels_folder = os.path.join(labels_dir, image_folder)
    for label_file in os.listdir(labels_folder):
        to_remove = False
        with open(os.path.join(labels_folder, label_file), "r") as file:
            lines = file.readlines()
        for line in lines:
            class_id = int(line.split()[0])
            if class_id in classes_to_remove:
                to_remove = True
                break
        if to_remove:
            # Move image to unknown_classes
            if image_folder == "test2007":
                os.rename(
                    os.path.join(images_dir, image_folder, label_file[:-4] + ".jpg"),
                    os.path.join(dataset_dir, "unknown_classes_test", label_file[:-4] + ".jpg"),
                )
            else:
                os.rename(
                    os.path.join(images_dir, image_folder, label_file[:-4] + ".jpg"),
                    os.path.join(dataset_dir, "unknown_classes", label_file[:-4] + ".jpg"),
                )

                

        


