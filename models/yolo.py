import torch
import torch.nn as nn
import torchvision

class YOLOBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=True):
        super(SimpleYOLO, self).__init__()
        
        # Use ResNet-34 as backbone
        backbone = torchvision.models.resnet34(pretrained=pretrained_backbone)
        
        # Use the first layers of ResNet as feature extractor
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        # Detection head
        self.num_classes = num_classes
        self.detection_layers = nn.Sequential(
            YOLOBlock(512, 256),
            YOLOBlock(256, 128),
            nn.Conv2d(128, num_classes + 5, kernel_size=1)  # x, y, w, h, conf, classes
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Detection head
        return self.detection_layers(features)

def get_yolo(num_classes, pretrained_backbone=True):
    """
    Get a simplified YOLO model
    
    Args:
        num_classes (int): Number of classes
        pretrained_backbone (bool): Whether to use pretrained backbone
        
    Returns:
        SimpleYOLO: The model
    """
    return SimpleYOLO(num_classes, pretrained_backbone)
