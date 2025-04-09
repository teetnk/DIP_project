/*
Development by :
Thanakorn Sutakiatsakul
Piyachok Ridsadaeng
*/
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # ✅ เปิดให้ API รับคำขอจากเว็บอื่นได้

# ✅ เสิร์ฟหน้าเว็บ
@app.route("/")
def home():
    return render_template("index.html")

UPLOAD_FOLDER = r"C:\Users\uouku\Desktop\DIP_project_code\Test_Food"
STATIC_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ เสิร์ฟ Static Files (CSS, JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(STATIC_FOLDER, filename)

def get_next_filename(filename, folder):
    """ สร้างชื่อไฟล์ใหม่โดยเพิ่มเลขต่อท้ายอัตโนมัติ """
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(folder, new_filename)):
        new_filename = f"{name}_{counter}{ext}"
        counter += 1

    return new_filename

@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        print("📥 ได้รับคำขออัปโหลดรูปภาพแล้ว!")

        # ✅ ตรวจสอบว่ามีไฟล์ถูกส่งมาหรือไม่
        if "image" not in request.files:
            print("❌ ไม่มีไฟล์ถูกส่งมา!")
            return jsonify({"status": "error", "message": "ไม่มีไฟล์อัปโหลด!"}), 400

        file = request.files["image"]

        # ✅ ตรวจสอบชื่อไฟล์
        if file.filename == "":
            print("❌ ไฟล์ไม่มีชื่อ!")
            return jsonify({"status": "error", "message": "ไฟล์ไม่มีชื่อ!"}), 400

        # ✅ กำหนดชื่อไฟล์ใหม่โดยเพิ่มเลขต่อท้ายถ้ามีไฟล์ซ้ำ
        filename = get_next_filename(file.filename, UPLOAD_FOLDER)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        print(f"✅ บันทึกไฟล์ที่: {file_path}")
        return jsonify({"status": "success", "message": "อัปโหลดสำเร็จ!", "path": file_path})
    except Exception as e:
        print("❌ เกิดข้อผิดพลาด:", e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
