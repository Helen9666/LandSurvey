import torchvision
from torchvision.models.detection.retinanet import RetinaNetHead

def get_retinanet(num_classes, pretrained=True):
    """
    Create a RetinaNet model with ResNet-50-FPN backbone
    
    Args:
        num_classes (int): Number of classes including background
        pretrained (bool): Whether to use pretrained backbone
        
    Returns:
        RetinaNet: The model
    """
    # Get pre-trained model
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=pretrained)
    
    # Get number of input features for the classifier
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Replace the pre-trained head with a new one
    model.head = RetinaNetHead(
        in_features,
        num_anchors,
        num_classes
    )
    
    return model
