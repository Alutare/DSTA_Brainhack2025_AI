import base64
import json
from pathlib import Path
import requests
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

from cv_manager import CVManager


def test_cv_manager_locally():
    """Test the CV Manager directly without the server."""
    print("Testing CVManager locally...")
    
    # Initialize the manager
    manager = CVManager()
    
    # Load a test image
    test_image_path = "test_image.jpg"
    
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
    
    # Run inference
    print("Running inference...")
    detections = manager.cv(image_bytes)
    
    # Print results
    print(f"Found {len(detections)} objects:")
    for i, detection in enumerate(detections):
        print(f"{i+1}. {detection['label']} (conf: {detection['confidence']:.2f})")
        print(f"   Box: ({detection['x']}, {detection['y']}) - {detection['width']}x{detection['height']}")
    
    # Visualize results
    visualize_detections(test_image_path, detections)
    
    return detections


def test_cv_server_request():
    """Test sending a request to the CV server."""
    print("Testing CV server request...")
    
    # Load a test image
    test_image_path = "test_image.jpg"
    
    with open(test_image_path, "rb") as f:
        image_bytes = f.read()
    
    # Encode to base64
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Prepare request payload
    payload = {
        "instances": [
            {"b64": b64_image}
        ]
    }
    
    # Send request to server
    url = "http://localhost:8000/cv"  # Adjust port if necessary
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        predictions = result["predictions"][0]
        
        # Print results
        print(f"Server returned {len(predictions)} objects:")
        for i, detection in enumerate(predictions):
            print(f"{i+1}. {detection['label']} (conf: {detection['confidence']:.2f})")
            print(f"   Box: ({detection['x']}, {detection['y']}) - {detection['width']}x{detection['height']}")
        
        # Visualize results
        visualize_detections(test_image_path, predictions, output_path="server_result.jpg")
        
        return predictions
    
    except requests.exceptions.ConnectionError:
        print("Failed to connect to server. Is the server running?")
        return None
    except Exception as e:
        print(f"Error testing server: {e}")
        return None


def visualize_detections(image_path, detections, output_path=None):
    """Visualize detections on the image."""
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Define colors for different classes (up to 18 classes)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0),
        (64, 0, 64), (0, 64, 64)
    ]
    
    # Create font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw each detection
    for i, detection in enumerate(detections):
        x = detection["x"]
        y = detection["y"]
        w = detection["width"]
        h = detection["height"]
        label = detection["label"]
        conf = detection["confidence"]
        
        # Get color by class (cycling through the colors list)
        color = colors[hash(label) % len(colors)]
        
        # Draw rectangle
        draw.rectangle([(x, y), (x + w, y + h)], outline=color, width=3)
        
        # Draw label
        text = f"{label} {conf:.2f}"
        text_w, text_h = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else font.getsize(text)
        draw.rectangle([(x, y - text_h - 4), (x + text_w, y)], fill=color)
        draw.text((x, y - text_h - 2), text, fill=(255, 255, 255), font=font)
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(image))
    plt.axis('off')
    
    # Save if output path is provided
    if output_path:
        image.save(output_path)
        print(f"Visualization saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test the CV manager locally
    local_detections = test_cv_manager_locally()
    
    # Test sending a request to the server
    # Note: Server must be running for this to work
    print("\nTesting server connection (will fail if server is not running)...")
    server_detections = test_cv_server_request()
    
    if local_detections and server_detections:
        # Compare results to ensure consistency
        local_count = len(local_detections)
        server_count = len(server_detections)
        
        print(f"\nComparison: Local detected {local_count} objects, Server detected {server_count} objects")
        
        if local_count == server_count:
            print("✅ Detection counts match!")
        else:
            print("❌ Detection counts differ between local and server.")