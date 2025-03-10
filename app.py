import os
import json
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, request, jsonify
from io import BytesIO
from tensorflow.keras.preprocessing import image
from PIL import Image
import boto3
import logging
import train_model

# ✅ ตั้งค่า LINE API
LINE_CHANNEL_ACCESS_TOKEN = "NpMQ7DSOlkaLp0/Q60f31LJER7OBd0rqvVPmg58ZqwDMd6ecU7OUiTviXf56u1YmSR+GuoRl+2hVC6kVAFAwtBSE9b07HLaKxSGYOBDUQjzHVigxujMyKEc35QDtv2NhHKRXacAJKNiy377Dnxcr0AdB04t89/1O/w1cDnyilFU="

# ✅ โหลดโมเดลที่เทรนเสร็จแล้ว
model = tf.keras.models.load_model("food_model_trained.h5")
class_names = ["ต้มยำกุ้ง", "กะเพราไก่", "ข้าวมันไก่"]  # 🔹 เปลี่ยนตาม Class ของโมเดล

app = Flask(__name__)

# ✅ ตั้งค่า S3 และ DynamoDB
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
table_name = os.getenv('LINE_BOT_TABLE', None)
table = dynamodb.Table(table_name)

# 📌 ฟังก์ชันทำนายเมนูอาหาร
def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence

# 📌 ฟังก์ชันส่งข้อความไปยัง LINE
def send_line_message(user_id, text):
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "to": user_id,
        "messages": [{"type": "text", "text": text}]
    }
    requests.post(url, headers=headers, json=data)

# 📌 Webhook API ที่รับรูปจาก LINE
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        if "events" in data:
            for event in data["events"]:
                if event.get("message", {}).get("type") == "image":
                    user_id = event["source"]["userId"]
                    message_id = event["message"]["id"]

                    # 📌 ดึงรูปจาก LINE
                    image_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
                    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
                    response = requests.get(image_url, headers=headers)

                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        predicted_class, confidence = predict_image(img)

                        response_text = f"📸 ฉันคิดว่านี่คือ {predicted_class} 🍽️\nความมั่นใจ: {confidence:.2f}%"
                        send_line_message(user_id, response_text)

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# 📌 Route สำหรับตรวจสอบว่าเซิร์ฟเวอร์ทำงานอยู่
@app.route("/", methods=["GET"])
def home():
    return "✅ Flask Server is Running!", 200

# 📌 Route สำหรับ Trigger เทรนโมเดลใหม่
@app.route("/train", methods=["POST"])
def train():
    try:
        train_model()
        return jsonify({"status": "ok", "message": "Training started!"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
