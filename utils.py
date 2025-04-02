import time
import datetime
import torch

def box_iou(box1, box2):
    """
    Compute IoU between boxes
    boxes are in [x1, y1, x2, y2] format
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    # Intersection coordinates
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # left-top [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # right-bottom [N,M,2]
    
    # Width and height of intersection
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    
    # IoU
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss