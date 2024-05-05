import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class RichContextCOCODataset(Dataset):
    def __init__(
        self,
        root,
        split='train',
        label_dir='',
        image_size=768, # image size for output
        image_size_when_labeling=768, # image size when labeling, the bbox xyxy are scaled to this size

    ):
        image_dir = os.path.join(root, f'{split}2017')
        images = os.listdir(image_dir)
        image_names = [os.path.basename(img).split('.')[0] for img in images]

        valid_image_paths = []
        valid_labels = []

        for image_name in image_names:
            label_path = os.path.join(label_dir, f'label_{image_name}.json')
            if os.path.exists(label_path):
                valid_image_paths.append(os.path.join(image_dir, f'{image_name}.jpg'))
                valid_labels.append(label_path)

        print(f"Found {len(valid_image_paths)} images")
        self.valid_image_paths = valid_image_paths
        self.valid_labels = valid_labels
        self.image_size = image_size
        self.image_size_when_labeling = image_size_when_labeling

    def __len__(self):
        return len(self.valid_image_paths)

    def parse_label(self, label_path):
        with open(label_path, 'r') as f:
            meta = json.load(f)

        caption = meta['caption']
        annos = meta['annos']

        bboxes = [anno['bbox'] for anno in annos]
        category_names = [anno['category_name'] for anno in annos]
        labels = [anno['caption'] for anno in annos]

        return caption, bboxes, category_names, labels
    
    def scale_bounding_box(self, height, width, bboxes):
        if height > width:
            scale = height / self.image_size_when_labeling
        else:
            scale = width / self.image_size_when_labeling
        
        scaled_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)
            scaled_bboxes.append([x1, y1, x2, y2])

        return scaled_bboxes

    def __getitem__(self, index):
        image_path = self.valid_image_paths[index]
        label_path = self.valid_labels[index]

        caption, bboxes, category_names, labels = self.parse_label(label_path)

        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        bboxes = self.scale_bounding_box(image.shape[0], image.shape[1], bboxes)

        return {
            'image': image.transpose(2, 0, 1),
            'image_path': image_path,
            'caption': caption,
            'bboxes': bboxes,
            'labels': labels,
            'categories': category_names
        }