from ultralytics import YOLO
import os

print("🚀 Starting YOLOv8 model download...")
print("This may take a few minutes depending on your internet connection.")

os.makedirs("models", exist_ok=True)
print("✅ Created models directory")

# Download YOLOv8 models
models_to_download = {
    "yolov8n.pt": "Standard YOLOv8 Nano (for traffic detection)",
    "yolov8s.pt": "Standard YOLOv8 Small (for helmet detection)"
}

for model_name, description in models_to_download.items():
    try:
        print(f"\n📥 Downloading {model_name} - {description}")
        model = YOLO(model_name)
        print(f"✅ Successfully downloaded {model_name}")
        
        if os.path.exists(model_name):
            os.rename(model_name, f"models/{model_name}")
            print(f"📁 Moved {model_name} to models folder")
            
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")

print("\n" + "="*50)
print("🎉 All downloads completed!")
print("📁 Models are saved in: models/")
print("🚀 You can now run: python main.py")
print("="*50)