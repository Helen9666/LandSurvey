import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transforms as T

class DroneDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load all image files, assuming images are in images folder
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        # Load annotations, assuming annotations are in annotations folder
        self.annotations = list(sorted(os.listdir(os.path.join(root, "annotations"))))
        
    def __getitem__(self, idx):
        # Load image and annotation
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        
        img = Image.open(img_path).convert("RGB")
        
        # Parse annotation file (assuming JSON format)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        
        # Build target dictionary
        boxes = []
        labels = []
        
        for obj in annotation['objects']:
            # Get bounding box coordinates [x_min, y_min, x_max, y_max]
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = obj['bbox'][2]
            ymax = obj['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])  # Class ID
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create image ID, area and iscrowd flag
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # Apply transformations
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.imgs)

def get_dataset(root, train=True):
    """Helper function to get a dataset with appropriate transforms"""
    transforms = T.get_transform(train=train)
    dataset = DroneDataset(root, transforms=transforms)
    return dataset

def get_dataloaders(root, batch_size=2, val_split=0.2, num_workers=4):
    """Create train and validation data loaders"""
    # Get full dataset
    dataset = get_dataset(root, train=True)
    
    # Split dataset
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Random split
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    return train_loader, val_loader