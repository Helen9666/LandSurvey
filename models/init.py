from .faster_rcnn import get_faster_rcnn_resnet50_fpn
from .retinanet import get_retinanet
from .yolo import get_yolo

def get_model(model_name, num_classes, pretrained=True):
    """
    Factory function to get a model by name
    
    Args:
        model_name (str): Name of the model (faster_rcnn, retinanet, yolo)
        num_classes (int): Number of classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: The model
    """
    if model_name == 'faster_rcnn':
        return get_faster_rcnn_resnet50_fpn(num_classes, pretrained)
    elif model_name == 'retinanet':
        return get_retinanet(num_classes, pretrained)
    elif model_name == 'yolo':
        return get_yolo(num_classes, pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from: faster_rcnn, retinanet, yolo")
