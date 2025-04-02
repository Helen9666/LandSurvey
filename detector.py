import torch
import numpy as np
import cv2
from PIL import Image
import transforms as T
from models import get_model

class ObjectDetector:
    def __init__(self, model_path=None, model_name=None, num_classes=None, class_names=None, gsd=0.05):
        """
        Initialize the object detector
        
        Args:
            model_path (str): Path to the model checkpoint
            model_name (str): Name of the model architecture (faster_rcnn, retinanet, yolo)
            num_classes (int): Number of classes including background
            class_names (list): List of class names
            gsd (float): Ground sample distance in meters per pixel
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
            
        # Load model
        if model_path is not None:
            # Load from checkpoint
            self.model = torch.load(model_path, map_location=self.device)
        else:
            # Create new model
            assert model_name is not None and num_classes is not None, "Must provide model_name and num_classes if model_path is None"
            self.model = get_model(model_name, num_classes)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Set class names
        self.class_names = class_names
        self.gsd = gsd  # Ground sample distance in meters per pixel
        
    def detect(self, image_path, conf_threshold=0.5):
        """
        Detect objects in an image
        
        Args:
            image_path (str): Path to the image
            conf_threshold (float): Confidence threshold
            
        Returns:
            list: List of detections
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Transform image
        transform = T.ToTensor()
        image_tensor, _ = transform(image, {})
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
        # Extract detections
        detections = []
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        
        # Process predictions based on model type
        if isinstance(outputs, list):
            # Torchvision detection models (Faster R-CNN, RetinaNet)
            prediction = outputs[0]
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            labels = prediction['labels'].cpu().numpy()
            
            # Filter by confidence
            indices = np.where(scores >= conf_threshold)[0]
            
            for i in indices:
                box = boxes[i].astype(np.int32)
                label = labels[i]
                score = scores[i]
                
                # Calculate physical dimensions
                width_pixels = box[2] - box[0]
                height_pixels = box[3] - box[1]
                width_meters = width_pixels * self.gsd
                height_meters = height_pixels * self.gsd
                area_meters = width_meters * height_meters
                
                # Create detection object
                detection = {
                    'class': self.class_names[label-1] if self.class_names else f"Class {label}",
                    'confidence': float(score),
                    'bbox': box.tolist(),
                    'dimensions': {
                        'width': round(width_meters, 2),
                        'height': round(height_meters, 2),
                        'area': round(area_meters, 2)
                    }
                }
                detections.append(detection)
            
        return detections
    
    def visualize(self, image_path, detections, output_path=None):
        """
        Visualize detections on an image
        
        Args:
            image_path (str): Path to the image
            detections (list): List of detections
            output_path (str): Path to save the output image
            
        Returns:
            numpy.ndarray: Image with visualized detections
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']
            width = det['dimensions']['width']
            height = det['dimensions']['height']
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{cls} ({conf:.2f}): {width}x{height}m"
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save result image
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
