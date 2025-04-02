import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_resnet50_fpn(num_classes, pretrained=True):
    """
    Get Faster R-CNN model with ResNet-50-FPN backbone from torchvision
    
    Args:
        num_classes (int): Number of classes including background
        pretrained (bool): Whether to use pretrained backbone
        
    Returns:
        FasterRCNN: The model
    """
    # Load a pre-trained model for classification and return
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
