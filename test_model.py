import timm
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# 📌 ตั้งค่าพาธไฟล์โมเดล
MODEL_PATH = "food_model_vit.pth"
CLASS_NAMES = ["ก๋วยเตี๋ยวเรือ", "ข้าวขาหมู", "ข้าวมันไก่"]  # 🔹 แก้ไขตามคลาสของคุณ

# 📌 โหลดโมเดล Vision Transformer (ViT)
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# 📌 ตั้งค่า Transform สำหรับทดสอบภาพ
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 ฟังก์ชันทดสอบภาพ
def predict_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ ไม่พบไฟล์รูปภาพ: {img_path}")

    print(f"📸 กำลังวิเคราะห์ภาพ: {img_path}")

    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # เพิ่มมิติให้เป็น (1, 3, 224, 224)

    with torch.no_grad():
        outputs = model(img)
        _, predicted = outputs.max(1)
    
    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()

    print(f"✅ คำทำนาย: {predicted_class} (ความมั่นใจ: {confidence:.2f}%)")

# 📌 ทดสอบกับภาพตัวอย่าง
TEST_IMAGE_PATH = "test_image.jpg"  # 🔹 เปลี่ยนเป็นภาพที่ต้องการทดสอบ
predict_image(TEST_IMAGE_PATH)
