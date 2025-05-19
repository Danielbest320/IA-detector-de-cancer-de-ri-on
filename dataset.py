from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch

class FFTImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, img_size=224):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.png')):
                    self.image_paths.append((os.path.join(class_dir, filename), label))

    def compute_fft_magnitude(self, image):
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        magnitude = np.log(1 + magnitude)
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        return magnitude

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path, label = self.image_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        fft_mag = self.compute_fft_magnitude(img)
        stacked = np.stack([img, fft_mag], axis=0)  # Shape: (2, H, W)

        return torch.tensor(stacked, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
