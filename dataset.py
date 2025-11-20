import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ImageNet(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for label, folder in enumerate(["n01440764", "n02102040", "n02979186", "n03000684", "n03028079", "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"], start=0):
            folder_path = os.path.join(image_dir, folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"Warning: could not load image {img_path}: {e}")
            return None

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, int(label)
    

