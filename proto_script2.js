const video = document.getElementById('cameraPreview');
const canvas = document.getElementById('canvas');
const photo = document.getElementById('photo');
const edgePhoto = document.getElementById('edgePhoto');
const captureButton = document.getElementById('capture');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const resetButton = document.getElementById('resetButton');
const foodName = document.getElementById('foodName');
const confidence = document.getElementById('confidence');
const nutritionInfo = document.getElementById('nutritionInfo');
const nutritionTable = document.getElementById('nutritionTable');
const labelInput = document.getElementById('labelInput');
const submitLabel = document.getElementById('submitLabel');
const trainingStatus = document.getElementById('trainingStatus');
const foodHistory = document.getElementById('foodHistory');
const foodList = document.getElementById("food-list");
const newLabel = document.getElementById("newLabel");

// เข้าถึงกล้อง
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => video.srcObject = stream)
    .catch(err => {
        console.error("เกิดข้อผิดพลาดในการเข้าถึงกล้อง: ", err);
        foodName.textContent = "ไม่สามารถเข้าถึงกล้องได้";
    });

// โหลดประวัติจาก localStorage
function loadHistory() {
    const history = JSON.parse(localStorage.getItem('foodHistory')) || [];
    foodHistory.innerHTML = '';
    history.forEach(item => {
        const li = document.createElement('li');
        li.textContent = `${item.name} (${item.calories || '-'} kcal)`;
        foodHistory.appendChild(li);
    });
}

async function loadFoodList() {
    try {
        const response = await fetch("/food_list");  // ดึงข้อมูลจาก Flask
        const foodNames = await response.json();

        foodList.innerHTML = "";  // ล้างค่าที่มีอยู่ก่อนหน้า
        foodNames.forEach(food => {
            let option = document.createElement("option");
            option.value = food;
            foodList.appendChild(option);
        });
    } catch (error) {
        console.error("❌ ไม่สามารถโหลดรายการอาหาร:", error);
    }
}

document.addEventListener("DOMContentLoaded", loadFoodList);

// บันทึกประวัติใน localStorage
function saveHistory(name, calories) {
    let history = JSON.parse(localStorage.getItem('foodHistory')) || [];
    history.unshift({ name, calories });
    if (history.length > 5) history.pop();
    localStorage.setItem('foodHistory', JSON.stringify(history));
    loadHistory();
}

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

// อัพโหลดไฟล์
uploadButton.addEventListener('click', () => fileInput.click());
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

// รีเซ็ต
resetButton.addEventListener('click', () => {
    photo.style.display = 'none';
    edgePhoto.style.display = 'none';
    foodName.textContent = "ชื่ออาหาร: -";
    confidence.textContent = "ความมั่นใจ: -";
    nutritionInfo.style.display = 'none';
    labelInput.style.display = 'none';
    trainingStatus.style.display = 'none';
});

let savedPath = null;  // 🔥 เพิ่มตัวแปรเก็บค่า path ของรูปภาพ

async function sendImageToBackend(imageData) {
    foodName.textContent = "ชื่ออาหาร: กำลังวิเคราะห์...";
    confidence.textContent = "ความมั่นใจ: -";
    nutritionInfo.style.display = 'none';
    labelInput.style.display = 'none';
    edgePhoto.style.display = 'none';
    trainingStatus.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });
        const data = await response.json();
        
        if (data.error) {
            foodName.textContent = "เกิดข้อผิดพลาด: " + data.error;
            confidence.textContent = "ความมั่นใจ: -";
        } else {
            foodName.textContent = `ชื่ออาหาร: ${data.food_name}`;
            confidence.innerHTML = `ความมั่นใจ: <span class="${parseFloat(data.confidence) >= 70 ? 'confidence-green' : 'confidence-orange'}">${data.confidence}</span>`;

            edgePhoto.src = data.edge_image;
            edgePhoto.style.display = 'block';

            savedPath = data.saved_path;  // ✅ บันทึก path ของรูปภาพที่ใช้ทำนาย
            console.log(`📂 บันทึก savedPath: ${savedPath}`);

            if (parseFloat(data.confidence) >= 70) {
                confirmFoodName(data.food_name, data.confidence);
            } else {
                labelInput.style.display = 'block';
            }

            if (data.needs_label) {
                alert(`โมเดลไม่แน่ใจ อาจเป็น "${data.food_name}" หรืออาหารอื่น กรุณาระบุชื่อที่ถูกต้อง`);
                labelInput.style.display = 'block';
            }
            submitLabel.onclick = async () => {
                const newLabelValue = newLabel.value.trim();  // ตัดช่องว่างออก
                console.log(`📤 ค่า input ที่ได้จาก index.html:`, newLabelValue);
            
                if (!newLabelValue) {
                    alert("❌ กรุณากรอกหรือเลือกชื่ออาหาร");
                    return;
                }
            
                if (!savedPath) {
                    alert("❌ ไม่พบ path ของรูปภาพ กรุณาถ่ายภาพใหม่");
                    return;
                }
            
                console.log(`📤 ส่งค่าไป update_label: path=${savedPath}, label=${newLabelValue}`);
            
                try {
                    const labelResponse = await fetch('/update_label', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            path: savedPath,
                            label: newLabelValue
                        })
                    });
            
                    if (!labelResponse.ok) throw new Error('การอัพเดทโมเดลล้มเหลว');
                    const labelData = await labelResponse.json();
                    console.log(`✅ ค่า response จาก Flask:`, labelData);
            
                    alert(labelData.message);
                    trainingStatus.style.display = 'none';
                    labelInput.style.display = 'none';
            
                    setTimeout(() => {
                        alert("🎉 การฝึกโมเดลเสร็จสิ้นแล้ว!");
                        location.reload();
                    }, 3000);
                } catch (error) {
                    alert(`❌ เกิดข้อผิดพลาด: ${error.message}`);
                    console.error("🚨 ERROR:", error);
                }
            };            
        }
    } catch (error) {
        console.error("❌ ERROR:", error);
        foodName.textContent = "เกิดข้อผิดพลาดในการเชื่อมต่อ: " + error.message;
    }
}

// ฟังก์ชันแสดงคำถามหากโมเดลมั่นใจ >= 70%
function confirmFoodName(foodName, confidence) {
    const confirmationContainer = document.createElement("div");
    confirmationContainer.id = "confirmationContainer";
    confirmationContainer.innerHTML = `
        <p>🍛 อาหารในรูปนี้คือ "<strong>${foodName}</strong>" ใช่ไหม?</p>
        <button id="confirmYes" class="btn-confirm">✔️ ใช่</button>
        <button id="confirmNo" class="btn-confirm">❌ ไม่ใช่</button>
    `;

    document.body.appendChild(confirmationContainer);

    document.getElementById("confirmYes").addEventListener("click", () => {
        document.body.removeChild(confirmationContainer);
        saveConfirmedFood(foodName);  // บันทึกโดยใช้ชื่อที่โมเดลทำนาย
    });

    document.getElementById("confirmNo").addEventListener("click", () => {
        document.body.removeChild(confirmationContainer);
        showLabelInput();  // แสดงช่องกรอกข้อมูลแทน
    });
}

// ฟังก์ชันเมื่อผู้ใช้กด "ใช่"
function saveConfirmedFood(foodName) {
    console.log(`✅ ผู้ใช้ยืนยันว่าเป็น "${foodName}"`);
    alert(`บันทึกเมนู: ${foodName} แล้ว!`);
    // สามารถเพิ่มโค้ดบันทึกลงฐานข้อมูลหรือเรียก API ได้ที่นี่
}

// ฟังก์ชันเมื่อผู้ใช้กด "ไม่ใช่"
function showLabelInput() {
    document.getElementById("labelInput").style.display = "block"; // แสดงช่องให้ผู้ใช้กรอกเอง
}


// โหลดประวัติเมื่อเริ่มต้น
loadHistory();
