"""Manages the CV model for object detection."""

import io
import os
import base64
from typing import Any, List, Dict
import numpy as np
from PIL import Image
from ultralytics import YOLO


class CVManager:
    def __init__(self, model_path: str = "best.pt"):
        self.model = self._load_model(model_path)
        self.class_names = self.model.names
        self.conf_threshold = 0.25
        print(f"Model loaded with {len(self.class_names)} classes")

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

    def process_single_image(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """Process a single image and return predictions in the expected format.
        
        Args:
            image_bytes: The image file in bytes.
            
        Returns:
            A list of dictionaries containing predictions in the format:
            [
                {
                    "bbox": [x, y, w, h],
                    "category_id": category_id
                },
                ...
            ]
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            results = self.model(image_np, conf=self.conf_threshold)

            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                    w, h = x2 - x1, y2 - y1
                    class_id = int(box.cls[0].item())
                    detections.append({
                        "bbox": [x1, y1, w, h],
                        "category_id": class_id,
                    })
            return detections
        except Exception as e:
            print(f"[ERROR] process_single_image failed: {e}")
            return []

    def cv(self, image_bytes: bytes):
        """For FastAPI single image inference."""
        return self.process_single_image(image_bytes)
    