"""Manages the CV model for object detection."""

import io
import os
from typing import Any, List, Dict
import numpy as np
from PIL import Image
from ultralytics import YOLO


class CVManager:
    def __init__(self, model_path: str = "best.pt"):
        """Initialize the YOLOv11 model for inference.
        
        Args:
            model_path: Path to the trained YOLOv11 weights file.
                        Defaults to "best.pt" in the current directory.
        """
        # Load the model
        self.model = self._load_model(model_path)
        
        # Load class names
        self.class_names = self.model.names
        
        # Set default confidence threshold
        self.conf_threshold = 0.25
        
        print(f"Model loaded successfully with {len(self.class_names)} classes")

    def _load_model(self, model_path: str) -> YOLO:
        """Load the YOLOv11 model.
        
        Args:
            model_path: Path to the model weights.
            
        Returns:
            Loaded YOLO model.
        """
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            # Try to find the model in common locations
            possible_paths = [
                "best.pt",
                "weights/best.pt",
                "cv/models/best.pt",
                "../weights/best.pt"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Found model at {model_path}")
                    break
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def cv(self, image: bytes) -> List[Dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of dictionaries containing predictions in the format:
            [
                {
                    "label": str,       # Class name
                    "confidence": float,  # Confidence score between 0 and 1
                    "x": int,           # Bounding box top-left x coordinate
                    "y": int,           # Bounding box top-left y coordinate
                    "width": int,       # Bounding box width
                    "height": int       # Bounding box height
                },
                ...
            ]
        """
        try:
            # Convert bytes to PIL Image
            image_pil = Image.open(io.BytesIO(image))
            
            # Convert PIL Image to numpy array (RGB format)
            image_np = np.array(image_pil)
            
            # Run inference
            results = self.model(image_np, conf=self.conf_threshold)
            
            # Process detection results
            predictions = []
            
            # Handle results
            for result in results:
                boxes = result.boxes
                
                # Extract detections
                for i, box in enumerate(boxes):
                    # Get box coordinates (convert to integer)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Calculate width and height
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Get class index and name
                    cls_idx = int(box.cls[0].item())
                    label = self.class_names[cls_idx]
                    
                    # Get confidence score
                    confidence = float(box.conf[0].item())
                    
                    # Create detection object
                    detection = {
                        "label": label,
                        "confidence": confidence,
                        "x": x1,
                        "y": y1,
                        "width": width,
                        "height": height
                    }
                    
                    predictions.append(detection)
            
            return predictions
            
        except Exception as e:
            print(f"Error during inference: {e}")
            # Return empty list if there's an error
            return []
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the confidence threshold for detections.
        
        Args:
            threshold: Confidence threshold between 0 and 1.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        self.conf_threshold = threshold