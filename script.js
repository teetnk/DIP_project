const video = document.getElementById('cameraPreview');
const canvas = document.getElementById('canvas');
const photo = document.getElementById('photo');
const captureButton = document.getElementById('capture');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const result = document.getElementById('result');
const nutritionInfo = document.getElementById('nutritionInfo');
const nutritionResult = document.getElementById('nutritionResult');
const labelInput = document.getElementById('labelInput');
const submitLabel = document.getElementById('submitLabel');

// เข้าถึงกล้อง
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        console.error("เกิดข้อผิดพลาดในการเข้าถึงกล้อง: ", err);
        result.textContent = "ไม่สามารถเข้าถึงกล้องได้";
    });

// ถ่ายภาพ
captureButton.addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');
    photo.src = imageData;
    photo.style.display = 'block';
    sendImageToBackend(imageData);
});

// คลิกปุ่มอัพโหลดเพื่อเลือกไฟล์
uploadButton.addEventListener('click', () => {
    fileInput.click();
});

// เลือกไฟล์
fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            photo.src = e.target.result;
            photo.style.display = 'block';
            sendImageToBackend(e.target.result);
        };
        reader.readAsDataURL(file);
    }
});

// ส่งรูปภาพไป backend และรับผลลัพธ์
async function sendImageToBackend(imageData) {
    result.textContent = "กำลังวิเคราะห์...";
    nutritionInfo.style.display = 'none';
    labelInput.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();
        
        if (data.error) {
            result.textContent = "เกิดข้อผิดพลาด: " + data.error;
        } else {
            result.textContent = `ผลลัพธ์: ${data.food_name} (มั่นใจ: ${data.confidence})`;
            if (data.nutrition && Object.keys(data.nutrition).length > 0) {
                nutritionResult.textContent = JSON.stringify(data.nutrition, null, 2);
                nutritionInfo.style.display = 'block';
            } else {
                nutritionResult.textContent = "ไม่มีข้อมูลโภชนาการสำหรับอาหารนี้";
                nutritionInfo.style.display = 'block';
            }
            console.log("บันทึกภาพที่: " + data.saved_path);

            // ถ้าโมเดลไม่มั่นใจ (confidence < 70%)
            if (data.needs_label) {
                labelInput.style.display = 'block';
                submitLabel.onclick = async () => {
                    const newLabel = document.getElementById('newLabel').value;
                    if (newLabel) {
                        const labelResponse = await fetch('/update_label', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ path: data.saved_path, label: newLabel })
                        });
                        const labelData = await labelResponse.json();
                        result.textContent = labelData.message;
                        labelInput.style.display = 'none';
                    }
                };
            }
        }
    } catch (error) {
        result.textContent = "เกิดข้อผิดพลาดในการเชื่อมต่อ: " + error.message;
    }
}