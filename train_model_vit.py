"""
Development by :
Thanakorn Sutakiatsakul
"""
import torch
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from PIL import Image
import os
import json

# 📌 ตรวจสอบว่าใช้ CPU หรือ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 กำลังใช้: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print(f"🖥️ GPU ที่ใช้: {torch.cuda.get_device_name(0)}")
    print(f"🧠 VRAM ทั้งหมด: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 📌 โหลดชื่อคลาส
dataset_path = "food_images_1"
CLASS_NAMES = [d for d in sorted(os.listdir(dataset_path)) if os.path.isdir(os.path.join(dataset_path, d)) and len(os.listdir(os.path.join(dataset_path, d))) > 0]
n_classes = len(CLASS_NAMES)

if n_classes == 0:
    raise ValueError("❌ ไม่พบโฟลเดอร์อาหารที่มีภาพใน dataset!")

print(f"🔍 จำนวนคลาสที่พบ: {n_classes} → {CLASS_NAMES}")

with open("food_classes.json", "w") as f:
    json.dump(CLASS_NAMES, f)

# 📌 โหลดโมเดล Vision Transformer
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=n_classes)
model.to(device)

# 📌 Loss Function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.00005)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 📌 Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 สร้าง FoodDataset
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and len(os.listdir(os.path.join(root_dir, d))) > 0])

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.images.append(img_path)
                    self.labels.append(label)
                except Exception:
                    print(f"❌ ข้ามไฟล์เสีย: {img_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# ✅ ป้องกัน RuntimeError ใน Windows
if __name__ == '__main__':
    full_dataset = FoodDataset(dataset_path, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    print(f"📌 จำนวนภาพ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print("✅ เริ่มสร้าง DataLoader ...")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    print("✅ DataLoader สร้างเสร็จ! กำลังเริ่มเทรน ...")

    # 📌 ใช้ Mixed Precision
   # ... (ส่วนอื่นเหมือนเดิม) ...

    # 📌 ใช้ Mixed Precision
    scaler = GradScaler()

    best_val_loss = float("inf")
    patience = 20  # ปรับเป็น 20
    patience_counter = 0

    for epoch in range(50):
        # Train Loop
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation Loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        print(f"📊 Epoch {epoch+1}/50, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Scheduler และ Early Stopping
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "food_model_vit_best.pth")
            print(f"✅ บันทึกโมเดลที่ดีที่สุดที่ Epoch {epoch+1}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹️ Early Stopping: หยุดการเทรนเนื่องจาก Val Loss ไม่ดีขึ้น")
                break

    print("✅ การเทรนเสร็จสิ้น! โมเดลที่ดีที่สุดถูกบันทึกที่ food_model_vit_best.pth")
