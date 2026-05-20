import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import JSON_PATH, IMAGE_SIZE


class TumorDataset(Dataset):
    def __init__(self, samples, class_to_idx, use_crop=False):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.use_crop = use_crop

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        with open(JSON_PATH, "r") as f:
            self.metadata = json.load(f)

    def crop_region(self, image, x, y, crop_size=128):
        h, w = image.shape[:2]

        x1 = max(0, x - crop_size // 2)
        y1 = max(0, y - crop_size // 2)
        x2 = min(w, x + crop_size // 2)
        y2 = min(h, y + crop_size // 2)

        return image[y1:y2, x1:x2]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, class_name = self.samples[idx]

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        relative_path = str(image_path).split("dataset/")[-1]
        relative_path = relative_path.replace("/", "\\")

        if self.use_crop and relative_path in self.metadata:
            point = self.metadata[relative_path]["point"]
            image = self.crop_region(image, point["x"], point["y"])

        image = Image.fromarray(image)
        image = self.transform(image)

        label = self.class_to_idx[class_name]

        return image, label