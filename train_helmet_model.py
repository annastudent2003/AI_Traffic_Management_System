import os
from ultralytics import YOLO
import yaml

def prepare_dataset():
    """Prepare dataset in YOLO format"""
    # Create dataset structure
    os.makedirs("helmet_dataset/images/train", exist_ok=True)
    os.makedirs("helmet_dataset/images/val", exist_ok=True)
    os.makedirs("helmet_dataset/labels/train", exist_ok=True)
    os.makedirs("helmet_dataset/labels/val", exist_ok=True)
    
    # Create dataset.yaml
    dataset_config = {
        'path': './helmet_dataset',
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'helmet', 1: 'no_helmet', 2: 'person'},
        'nc': 3
    }
    
    with open('helmet_dataset/dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f)
    
    print("✅ Dataset structure prepared")

def train_model():
    """Train custom helmet detection model"""
    # Load model
    model = YOLO("yolov8s.pt")
    
    # Train the model
    results = model.train(
        data="helmet_dataset/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        patience=15,
        device="cpu",
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        weight_decay=0.0005,
        dropout=0.1,
        verbose=True
    )
    
    print("✅ Training completed!")
    return model

def evaluate_model(model):
    """Evaluate model performance"""
    # Evaluate on validation set
    metrics = model.val()
    print(f"📊 mAP50-95: {metrics.box.map:.3f}")
    print(f"📊 mAP50: {metrics.box.map50:.3f}")
    print(f"📊 Precision: {metrics.box.mp:.3f}")
    print(f"📊 Recall: {metrics.box.mr:.3f}")
    
    # Export to ONNX for better performance
    model.export(format="onnx")
    print("✅ Model exported to ONNX format")

if __name__ == "__main__":
    print("🚀 Starting helmet detection model training...")
    prepare_dataset()
    print("📝 Please manually organize your images and labels into:")
    print("   - helmet_dataset/images/train/")
    print("   - helmet_dataset/images/val/")
    print("   - helmet_dataset/labels/train/")
    print("   - helmet_dataset/labels/val/")
    print("📝 Then run this script again to start training")
    
    # Check if dataset exists and start training
    if os.path.exists("helmet_dataset/images/train") and len(os.listdir("helmet_dataset/images/train")) > 0:
        model = train_model()
        evaluate_model(model)
    else:
        print("❌ Dataset not ready. Please organize your data first.")