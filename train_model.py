import timm
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# 📌 ตั้งค่าพาธ Dataset
dataset_path = "food_images_1"

# 📌 โหลดโมเดล ViT (Vision Transformer)
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=3)

# 📌 กำหนดค่า Transform เพื่อปรับแต่งข้อมูล
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 📌 สร้างคลาสสำหรับโหลดข้อมูล
class FoodDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = os.listdir(root_dir)

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.images.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# 📌 โหลดข้อมูล
train_dataset = FoodDataset(os.path.join(dataset_path, "train"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 📌 กำหนด Loss และ Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 📌 เทรนโมเดล
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(20):  # 🔹 เทรน 20 รอบ
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f"Epoch {epoch+1}/20, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%")

# 📌 บันทึกโมเดลที่เทรนเสร็จ
torch.save(model.state_dict(), "food_model_vit.pth")
print("✅ โมเดล Vision Transformer เทรนเสร็จแล้ว บันทึกเป็น `food_model_vit.pth`")
