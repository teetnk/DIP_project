"""
Development by :
Thanakorn Sutakiatsakul
"""
import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import os

# 📌 ตรวจสอบว่ามี CUDA หรือไม่
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 ใช้: {device}")

# 📌 โหลดชื่อคลาส
CLASS_NAMES = sorted(os.listdir("food_images_1"))  # โฟลเดอร์ dataset

# 📌 โหลดโมเดลที่เทรนเสร็จ
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("food_model_vit_best.pth", map_location=device))
model.to(device)
model.eval()

# 📌 Transform สำหรับทดสอบ
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 ฟังก์ชันพยากรณ์จากรูปภาพ
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = test_transform(img).unsqueeze(0).to(device)  # เพิ่ม batch dimension

    with torch.no_grad():
        output = model(img)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted_idx = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence = conf.item() * 100

    return predicted_class, confidence

# 📌 ทดสอบพยากรณ์
test_image = "food_images_1/ไก่ทอดหาดใหญ่/download (4).jpg"  # เปลี่ยนเป็นรูปที่ต้องการทดสอบ
predicted_class, confidence = predict_image(test_image)
print(f"📸 ฉันคิดว่านี่คือ {predicted_class} 🍽️ (ความมั่นใจ: {confidence:.2f}%)")
