from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import json
import shutil
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets

app = Flask(__name__)
CORS(app)

# 📌 กำหนดโฟลเดอร์
UPLOAD_FOLDER = r"C:\งานตั้น\DIP\DIP_project\food_images_1\test_food"
STATIC_FOLDER = "static"
TRAINING_FOLDER = "food_images_1"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)

# 📌 ตรวจสอบ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 ใช้: {device}")

# 📌 โหลดโมเดลและข้อมูล
MODEL_PATH = "food_model_vit_best.pth"
NUTRITION_FILE = "food_nutrition.json"

# โหลด CLASS_NAMES จาก food_classes.json
with open("food_classes.json", "r") as f:
    CLASS_NAMES = json.load(f)
print(f"📋 CLASS_NAMES: {CLASS_NAMES} (จำนวน: {len(CLASS_NAMES)})")

# ตรวจสอบข้อมูลใน food_images_1
print("📂 ตรวจสอบ food_images_1:")
for folder in os.listdir(TRAINING_FOLDER):
    num_images = len(os.listdir(os.path.join(TRAINING_FOLDER, folder)))
    print(f"  - {folder}: {num_images} ภาพ")

# โหลดโมเดล
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        checkpoint_classes = checkpoint['head.weight'].shape[0]
        print(f"📦 โมเดลเก่ามี {checkpoint_classes} คลาส")
        if checkpoint_classes != len(CLASS_NAMES):
            print(f"⚠️ จำนวนคลาสไม่ตรง (โมเดล: {checkpoint_classes}, CLASS_NAMES: {len(CLASS_NAMES)})")
            model.head = torch.nn.Linear(model.head.in_features, len(CLASS_NAMES))
            state_dict = checkpoint
            new_state_dict = model.state_dict()
            for key in state_dict:
                if key in new_state_dict and new_state_dict[key].shape == state_dict[key].shape:
                    new_state_dict[key] = state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ ปรับโมเดลให้รองรับคลาสใหม่เรียบร้อย")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ โหลดโมเดลสำเร็จ จำนวนคลาส: {len(CLASS_NAMES)}")
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
        print("⚠️ ใช้โมเดลเริ่มต้นแทน")
else:
    print("⚠️ ไม่พบโมเดล เริ่มจากโมเดลใหม่")
model.to(device)
model.eval()

# โหลดข้อมูลโภชนาการ
try:
    with open(NUTRITION_FILE, "r", encoding="utf-8") as f:
        NUTRITION_DATA = json.load(f)
    print(f"✅ โหลดข้อมูลโภชนาการสำเร็จ: {list(NUTRITION_DATA.keys())}")
except FileNotFoundError:
    NUTRITION_DATA = {
        "แกงขี้เหล็ก": {"calories": 250, "protein": 5, "fat": 8, "carbs": 10},
        "ข้าวผัด": {"calories": 200, "protein": 6, "fat": 10, "carbs": 30}
    }
    print("⚠️ ไม่พบ food_nutrition.json ใช้ข้อมูลเริ่มต้น")

# 📌 Transform สำหรับการทำนาย
predict_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 Transform สำหรับการเทรน
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/food_list", methods=["GET"])
def get_food_list():
    return jsonify(CLASS_NAMES)  # ส่งรายการอาหารทั้งหมดกลับไป


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

def enhance_image(image):
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    enhanced_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
    enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_img_rgb)

def detect_edges(image):
    img_np = np.array(image)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def predict_image(image):
    enhanced_image = enhance_image(image)
    img = predict_transform(enhanced_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted_idx = torch.max(probs, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = conf.item() * 100
    nutrition = NUTRITION_DATA.get(predicted_class, {})
    
    edge_image = detect_edges(image)
    edge_buffer = BytesIO()
    edge_image.save(edge_buffer, format="JPEG")
    edge_data = base64.b64encode(edge_buffer.getvalue()).decode('utf-8')
    
    print(f"🔍 ทำนาย: {predicted_class} (Confidence: {confidence:.2f}%)")
    return predicted_class, confidence, nutrition, edge_data

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    
    food_name, confidence, nutrition, edge_data = predict_image(img)
    
    if confidence >= 70:
        class_folder = os.path.join(UPLOAD_FOLDER, food_name)
    else:
        class_folder = os.path.join(UPLOAD_FOLDER, "unknown")
    os.makedirs(class_folder, exist_ok=True)
    filename = get_next_filename(f"{food_name}.jpg", class_folder)
    file_path = os.path.join(class_folder, filename)
    img.save(file_path)
    
    response = {
        "food_name": food_name,
        "confidence": f"{confidence:.2f}%",
        "nutrition": nutrition,
        "saved_path": file_path,
        "needs_label": confidence < 70,
        "edge_image": f"data:image/jpeg;base64,{edge_data}"
    }
    print(f"📤 ส่งข้อมูลไป frontend: {response}")
    return jsonify(response)

@app.route("/update_label", methods=["POST"])
def update_label():
    global model, CLASS_NAMES

    data = request.get_json()
    print(f"📥 ได้รับค่า: {data}")  # ✅ DEBUG LOG เพื่อตรวจสอบค่าที่ส่งมา

    if not data or 'path' not in data or 'label' not in data:
        print("❌ ข้อมูลไม่ครบ")
        return jsonify({'error': '❌ ข้อมูลไม่ครบ'}), 400

    old_path = data['path']
    new_label = str(data['label']).strip()  # ✅ ป้องกันข้อมูลผิดประเภท

    print(f"📂 ค่าที่ได้รับ: path={old_path}, label={new_label}")

    if not old_path or not new_label:
        print("❌ ค่า path หรือ label ว่างเปล่า")
        return jsonify({'error': '❌ ค่า path หรือ label ว่างเปล่า'}), 400

    if not os.path.exists(old_path):
        print(f"❌ ไฟล์ไม่พบ: {old_path}")
        return jsonify({'error': f"❌ ไฟล์ไม่พบ: {old_path}"}), 400

    new_folder = os.path.join(TRAINING_FOLDER, new_label)
    os.makedirs(new_folder, exist_ok=True)
    new_path = os.path.join(new_folder, os.path.basename(old_path))

    shutil.move(old_path, new_path)
    print(f"📂 ย้ายไฟล์จาก {old_path} ไปยัง {new_path}")

    if not os.path.exists(new_path):
        print(f"❌ ย้ายไฟล์ไม่สำเร็จ: {new_path}")
        return jsonify({'error': f"❌ ย้ายไฟล์ไม่สำเร็จ: {new_path}"}), 400

    is_new_class = new_label not in CLASS_NAMES
    if is_new_class:
        CLASS_NAMES.append(new_label)
        with open("food_classes.json", "w") as f:
            json.dump(CLASS_NAMES, f)
        print(f"✅ เพิ่ม {new_label} ใน food_classes.json")

    num_images = len(os.listdir(new_folder))
    print(f"📸 จำนวนภาพใน {new_label}: {num_images}")

    if is_new_class or num_images >= 1:
        print("🔄 กำลังเรียก retrain_model()...")
        retrain_model(is_new_class)
        print("✅ Retrain โมเดลสำเร็จ!")

    return jsonify({"status": "success", "message": f"อัพเดทเป็น {new_label} และเริ่มฝึกโมเดลใหม่"})



def retrain_model(is_new_class):
    global model, CLASS_NAMES
    
    print(f"🚀 เริ่ม retrain โมเดล (ใหม่: {is_new_class})...")
    
    train_dataset = datasets.ImageFolder(TRAINING_FOLDER, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    if len(train_dataset) == 0:
        print("❌ ไม่มีข้อมูลใน food_images_1 ไม่สามารถเทรนได้")
        return
    
    new_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
    new_model.load_state_dict(model.state_dict(), strict=False)
    new_model.to(device)
    
    optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001 if is_new_class else 0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    epochs = 10 if is_new_class else 3
    new_model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = new_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    model = new_model
    model.eval()
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Retrain เสร็จสิ้น บันทึกโมเดลที่ {MODEL_PATH}")

def get_next_filename(filename, folder):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1
    return new_filename

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
