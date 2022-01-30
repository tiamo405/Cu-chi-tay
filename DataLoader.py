import torch
from torch.utils.data import DataLoader 
from torchvision.transforms import transforms
import numpy as np
import os
import cv2
width        = 224
height       = 224

class CustomDataset(DataLoader):
    def __init__(self, DATA_PATH):
        # super(CustomDataset, self).__init__()
        labels_dict = ['0','1','2','3','4']
        self.images = []
        self.labels = []
        for i in labels_dict : # data chia anh theo tung label
          for image_file in os.listdir(DATA_PATH+'/'+i):
            image_path = os.path.join(DATA_PATH+'/'+i, image_file)
            self.images.append(image_path)
            self.labels.append(int(image_file.split(".")[1]))
    # (N, W, H, C)
    # (N, C, W, H)
    # RGB 190 85 10 => 190/255
    def __getitem__(self, idx):
        image = cv2.imread((self.images[idx]))
        image = cv2.resize(image, (width, height))
        label = self.labels[idx]
        # augmentation:
        aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) # W H C -> C W H 
        image = aug(image)
        return image, label
    
    def __len__(self) -> int:
        return len(self.images)

if __name__ == "__main__":
    dataset = CustomDataset("data\\train")
    print(dataset.__getitem__(354)[1])
    print(dataset.__getitem__(354)[0].shape)
    print(dataset.__len__())