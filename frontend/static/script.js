const fileInput = document.getElementById("fileInput");
const captureButton = document.getElementById("capture");
const video = document.getElementById("cameraPreview");
const canvas = document.getElementById("canvas");
const photo = document.getElementById("photo");

// ✅ ใช้ Ngrok URL เพื่อส่งภาพไปยัง Flask API
const BACKEND_URL = "https://6c3b-158-108-229-254.ngrok-free.app"; // 🔹 แก้ให้ตรงกับ URL ของคุณ

// 🚀 ฟังก์ชันส่งภาพไปยัง Backend
async function uploadImage(file) {
    const formData = new FormData();
    formData.append("image", file);

    console.log("📤 กำลังอัปโหลดไฟล์:", file.name);

    try {
        const response = await fetch(`${BACKEND_URL}/upload`, {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        console.log("✅ อัปโหลดสำเร็จ:", data);
        alert(data.message);
    } catch (error) {
        console.error("❌ อัปโหลดผิดพลาด", error);
        alert("อัปโหลดผิดพลาด กรุณาลองใหม่!");
    }
}



// 📸 เปิดกล้อง
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error("ไม่สามารถเปิดกล้อง:", error);
    }
}

// 📸 ถ่ายภาพและอัปโหลด
captureButton.addEventListener("click", () => {
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    photo.src = canvas.toDataURL("image/png");

    // แปลงเป็นไฟล์และอัปโหลด
    canvas.toBlob((blob) => {
        const file = new File([blob], "captured_image.png", { type: "image/png" });
        uploadImage(file);
    });
});

// 📂 อัปโหลดรูปจากแกลเลอรี่
fileInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            photo.src = e.target.result;
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }
});

// ✅ เรียกฟังก์ชันเปิดกล้องเมื่อโหลดหน้าเว็บ
startCamera();
