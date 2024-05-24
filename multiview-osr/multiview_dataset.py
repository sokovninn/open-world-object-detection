
import torch
from torch.utils.data import Dataset
import os
import glob
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import numpy as np
import cv2
import random

class LetterBox(object):
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32, return_int=False) -> None:
        self.new_shape=new_shape
        self.color=color
        self.auto = auto
        self.scaleup = scaleup
        self.stride = stride
        self.return_int = return_int
        
    def __call__(self, im):
        '''Resize and pad image while meeting stride-multiple constraints.'''
        im = np.array(im)
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding

        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border
        if not self.return_int:
            return im
        else:
            return im



class MultiviewDataset(Dataset):
    def __init__(self, dataset_path, each_n=1, image_size=(128,128), n_views=1, seed=0) -> None:
        self.dataset_path = dataset_path
        self.class_names = []
        self.each_n = each_n
        self.n_views = n_views
        for dir in os.listdir(self.dataset_path ):
            self.class_names.append(dir)

        random.seed(seed)

        self.image_paths = self.load_image_paths()
        self.loader = default_loader
        self.transform = transforms.Compose([
            LetterBox(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4856, 0.4994, 0.4325), (0.2264, 0.2218, 0.2606)),
    ])


    def load_image_paths(self):
        image_paths = []
        for class_name in self.class_names:
            class_path = os.path.join(self.dataset_path, class_name)
            #print(class_path)
            num_objects = len(next(os.walk(class_path))[1])
            print(num_objects)
            print(os.listdir(class_path))
            for object_dir in os.listdir(class_path):
                #print(os.path.join(class_path + f"_{i}", "*_crop.png"))
                #print(os.listdir(class_path + f"_{i}"))
                object_path = os.path.join(class_path, object_dir)
                #print(os.path.join(class_path + f"{class_name}_{i}", "*_crop.png"))
                filenames = glob.glob(os.path.join(object_path, "*_crop.png"))[::self.each_n]
                print(filenames)
                filenames = random.choices(filenames, k=self.n_views)
                #print(filenames)
                image_paths.append({"filenames": filenames, "label": class_name})
        return image_paths





    def __getitem__(self, idx):
        object_paths = self.image_paths[idx]
        object_filenames = object_paths["filenames"]
        object_label = object_paths["label"]
        object_images = []
        for filename in object_filenames:
            img_tensor = self.transform(self.loader(filename))
            object_images.append(img_tensor)
        #print(len(object_images), object_images[0].shape, object_images[1].shape)
        #print("Label")
        #print(object_label)
        #print(self.class_names)
        #print(object_filenames)
        images = torch.stack(object_images)
        return images, object_label


    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    dataset = MultiviewDataset("/home/nikita/Downloads/rgbd-dataset")
    print(len(dataset))
    print(dataset[0][0].shape)
