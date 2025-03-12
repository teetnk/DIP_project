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
import time
from threading import Lock

app = Flask(__name__)
CORS(app)

# 📌 กำหนดโฟลเดอร์
UPLOAD_FOLDER = r"C:\งานตั้น\DIP\DIP_project\uploaded_images"
STATIC_FOLDER = "static"
TRAINING_FOLDER = r"C:\งานตั้น\DIP\DIP_project\food_images_1"
MODEL_PATH = "food_model_vit_best.pth"
CLASS_FILE = "food_classes.json"
NUTRITION_FILE = "food_nutrition_fixed.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRAINING_FOLDER, exist_ok=True)

# 📌 ตัวแปรควบคุมการเทรน
is_training = False
training_lock = Lock()  # ล็อกเพื่อป้องกันการเข้าถึงพร้อมกัน
train_queue = []  # คิวสำหรับเก็บคลาสที่ต้องเทรน

# 📌 ตรวจสอบ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 ใช้: {device}")

# 📌 โหลด CLASS_NAMES
try:
    with open(CLASS_FILE, "r") as f:
        CLASS_NAMES = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    CLASS_NAMES = ["กุ้งอบวุ้นเส้น", "ขนมไหว้พระจันทร์"]
    with open(CLASS_FILE, "w") as f:
        json.dump(CLASS_NAMES, f)
    print("⚠️ ไม่พบหรือไฟล์ food_classes.json เสียหาย ใช้ค่าเริ่มต้น")
print(f"📋 CLASS_NAMES: {CLASS_NAMES} (จำนวน: {len(CLASS_NAMES)})")

# ตรวจสอบข้อมูลใน food_images_1
print("📂 ตรวจสอบ food_images_1:")
for folder in os.listdir(TRAINING_FOLDER):
    num_images = len(os.listdir(os.path.join(TRAINING_FOLDER, folder)))
    print(f"  - {folder}: {num_images} ภาพ")

# โหลดโมเดล
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=len(CLASS_NAMES))
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
        print("⚠️ ใช้โมเดล pretrained แทน")
else:
    print("⚠️ ไม่พบโมเดล เริ่มจากโมเดล pretrained")
model.to(device)
model.eval()

# 📌 โหลดข้อมูลโภชนาการจาก food_nutrition_fixed.json โดยตรง
try:
    with open(NUTRITION_FILE, "r", encoding="utf-8") as f:
        NUTRITION_DATA = json.load(f)
    if "foods" not in NUTRITION_DATA:
        raise ValueError("ไฟล์ food_nutrition_fixed.json ไม่มีคีย์ 'foods'")
    print(f"✅ โหลดข้อมูลโภชนาการสำเร็จ: {list(NUTRITION_DATA['foods'].keys())}")
except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
    print(f"❌ ไม่สามารถโหลด food_nutrition_fixed.json ได้: {e}")
    NUTRITION_DATA = None

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
    return jsonify(CLASS_NAMES)

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
    if NUTRITION_DATA is None:
        return None, 0, {}, ""
    
    enhanced_image = enhance_image(image)
    img = predict_transform(enhanced_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted_idx = torch.max(probs, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = conf.item() * 100
    nutrition = NUTRITION_DATA.get("foods", {}).get(predicted_class, {})
    
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
    if food_name is None:
        return jsonify({'error': 'ไม่สามารถโหลดข้อมูลโภชนาการได้ กรุณาตรวจสอบ food_nutrition_fixed.json'}), 500
    
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
    global model, CLASS_NAMES, is_training, train_queue

    data = request.get_json()
    print(f"📥 ได้รับค่า: {data}")
    
    if not data or 'path' not in data or 'label' not in data:
        print("❌ ข้อมูลไม่ครบ")
        return jsonify({'error': '❌ ข้อมูลไม่ครบ'}), 400
    
    old_path = data['path']
    new_label = str(data['label']).strip()
    nutrition = data.get('nutrition', {})
    
    print(f"📂 ค่าที่ได้รับ: path={old_path}, label={new_label}, nutrition={nutrition}")
    
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
    
    if NUTRITION_DATA is not None and new_label not in NUTRITION_DATA.get("foods", {}) and nutrition:
        NUTRITION_DATA.setdefault("foods", {})[new_label] = nutrition
        with open(NUTRITION_FILE, "w", encoding="utf-8") as f:
            json.dump(NUTRITION_DATA, f, ensure_ascii=False, indent=2)
        print(f"✅ เพิ่ม {new_label} ใน {NUTRITION_FILE} พร้อมข้อมูลโภชนาการ")
    
    is_new_class = new_label not in CLASS_NAMES
    if is_new_class:
        CLASS_NAMES.append(new_label)
        with open(CLASS_FILE, "w") as f:
            json.dump(CLASS_NAMES, f)
        print(f"✅ เพิ่ม {new_label} ใน {CLASS_FILE}")
    
    num_images = len(os.listdir(new_folder))
    print(f"📸 จำนวนภาพใน {new_label}: {num_images}")
    
    # เพิ่มคลาสหรือภาพในคิวเทรน
    with training_lock:
        if is_new_class or num_images >= 5:
            if new_label not in train_queue:
                train_queue.append(new_label)
            if not is_training:
                is_training = True
                # เริ่มเทรนหลังจากรอ 2 วินาทีเพื่อให้ request อื่นเข้ามา
                time.sleep(2)  # รอให้ request อื่นเพิ่มใน queue
                process_training()

    return jsonify({"status": "success", "message": f"อัพเดทเป็น {new_label} และเริ่มฝึกโมเดลใหม่"})

def process_training():
    global model, CLASS_NAMES, is_training, train_queue
    while train_queue:
        with training_lock:
            if not train_queue:
                is_training = False
                return
            label_to_train = train_queue.pop(0)
        
        print(f"🚀 เริ่ม retrain โมเดลสำหรับ {label_to_train}...")
        
        train_dataset = datasets.ImageFolder(TRAINING_FOLDER, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        if len(train_dataset) == 0:
            print("❌ ไม่มีข้อมูลใน food_images_1 ไม่สามารถเทรนได้")
            is_training = False
            return
        
        new_model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
        new_model.load_state_dict(model.state_dict(), strict=False)
        new_model.to(device)
        
        optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001 if label_to_train not in CLASS_NAMES else 0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        
        epochs = 5 if label_to_train not in CLASS_NAMES else 3  # ลด epoch เพื่อความเร็ว
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
        print(f"✅ Retrain เสร็จสิ้นสำหรับ {label_to_train}, บันทึกโมเดลที่ {MODEL_PATH}")
    
    with training_lock:
        is_training = False

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
