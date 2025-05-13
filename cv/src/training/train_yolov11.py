from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

train_results = model.train(
    data="datasets/yolo_dataset1/data.yaml", 
    epochs=30,  
    batch = -1,
    imgsz=640, 
    workers=0,
    augment = True,
    device=0  
)

# Evaluate the model's performance on the validation set
metrics = model.val()