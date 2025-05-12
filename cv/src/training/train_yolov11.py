from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

train_results = model.train(
    data="datasets/yolo_dataset/data.yaml", 
    epochs=30,  
    batch = 64,
    imgsz=640, 
    workers=0,
    device=0  
)

# Evaluate the model's performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("datasets/cv/images/0.jpg")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model