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

app = Flask(__name__)
CORS(app)

# 📌 กำหนดโฟลเดอร์
UPLOAD_FOLDER = r"C:\Users\uouku\Desktop\DIP_project_code\Test_Food"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 📌 ตรวจสอบ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 ใช้: {device}")

# 📌 โหลดโมเดลและข้อมูล
MODEL_PATH = "food_model_vit_best.pth"
CLASS_NAMES = sorted(os.listdir("food_images_1"))  # รีเซ็ตจาก dataset เดิม
NUTRITION_FILE = "food_nutrition.json"

# โหลดโมเดล
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ โหลดโมเดลสำเร็จ จำนวนคลาส: {len(CLASS_NAMES)}")
except Exception as e:
    print(f"❌ ไม่สามารถโหลดโมเดลได้: {e}")
    raise

# โหลดข้อมูลโภชนาการ
try:
    with open(NUTRITION_FILE, "r", encoding="utf-8") as f:
        NUTRITION_DATA = json.load(f)
except FileNotFoundError:
    NUTRITION_DATA = {
        "แกงขี้เหล็ก": {"calories": 150, "protein": 5, "fat": 8},
        "ข้าวผัด": {"calories": 200, "protein": 6, "fat": 10}
    }

# 📌 Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 เสิร์ฟหน้าเว็บ
@app.route("/")
def home():
    return render_template("index.html")

# 📌 เสิร์ฟ Static Files
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

# 📌 ฟังก์ชันพยากรณ์
def predict_image(image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted_idx = torch.max(probs, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = conf.item() * 100
    nutrition = NUTRITION_DATA.get(predicted_class, {})
    return predicted_class, confidence, nutrition

# 📌 API ทำนายอาหารและบันทึก
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    
    food_name, confidence, nutrition = predict_image(img)
    
    if confidence >= 70:
        class_folder = os.path.join(UPLOAD_FOLDER, food_name)
    else:
        class_folder = os.path.join(UPLOAD_FOLDER, "unknown")
    os.makedirs(class_folder, exist_ok=True)
    filename = get_next_filename(f"{food_name}.jpg", class_folder)
    file_path = os.path.join(class_folder, filename)
    img.save(file_path)
    
    return jsonify({
        "food_name": food_name,
        "confidence": f"{confidence:.2f}%",
        "nutrition": nutrition,
        "saved_path": file_path,
        "needs_label": confidence < 70
    })

# 📌 API อัพเดท label
@app.route("/update_label", methods=["POST"])
def update_label():
    data = request.get_json()
    if not data or 'path' not in data or 'label' not in data:
        return jsonify({'error': 'Missing path or label'}), 400
    
    old_path = data['path']
    new_label = data['label']
    new_folder = os.path.join(UPLOAD_FOLDER, new_label)
    os.makedirs(new_folder, exist_ok=True)
    
    new_path = os.path.join(new_folder, os.path.basename(old_path))
    shutil.move(old_path, new_path)
    
    return jsonify({"status": "success", "message": f"อัพเดทเป็น {new_label}"})

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