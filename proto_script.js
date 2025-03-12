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

// ส่งภาพไป backend
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
            
            nutritionInfo.style.display = 'block';
            while (nutritionTable.rows.length > 1) {
                nutritionTable.deleteRow(1);
            }
            if (data.nutrition && Object.keys(data.nutrition).length > 0) {
                const units = { calories: "kcal", protein: "g", fat: "g", carbs: "g" };
                for (const [key, value] of Object.entries(data.nutrition)) {
                    const row = nutritionTable.insertRow();
                    const cell1 = row.insertCell(0);
                    const cell2 = row.insertCell(1);
                    cell1.textContent = key === "calories" ? "พลังงาน" : key === "protein" ? "โปรตีน" : key === "fat" ? "ไขมัน" : "คาร์โบไฮเดรต";
                    cell2.textContent = `${value} ${units[key] || ''}`;
                }
                saveHistory(data.food_name, data.nutrition.calories);
            } else {
                const row = nutritionTable.insertRow();
                const cell = row.insertCell(0);
                cell.colSpan = 2;
                cell.textContent = "ไม่มีข้อมูลโภชนาการสำหรับอาหารนี้";
                saveHistory(data.food_name, null);
            }

            if (data.needs_label) {
                alert(`โมเดลไม่แน่ใจ อาจเป็น "${data.food_name}" หรืออาหารอื่น กรุณาระบุชื่อที่ถูกต้อง`);
                labelInput.style.display = 'block';
                submitLabel.onclick = async () => {
                    const newLabel = document.getElementById('newLabel').value;
                    const calories = document.getElementById('calories').value;
                    if (newLabel) {
                        trainingStatus.style.display = 'block';
                        try {
                            const labelResponse = await fetch('/update_label', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                    path: data.saved_path,
                                    label: newLabel,
                                    nutrition: calories ? { calories: parseInt(calories) } : {}
                                })
                            });
                            if (!labelResponse.ok) throw new Error('การอัพเดทโมเดลล้มเหลว');
                            const labelData = await labelResponse.json();
                            foodName.textContent = labelData.message;
                            labelInput.style.display = 'none';
                            trainingStatus.style.display = 'none';
                            saveHistory(newLabel, calories ? parseInt(calories) : null);
                        } catch (error) {
                            foodName.textContent = `เกิดข้อผิดพลาด: ${error.message}`;
                            trainingStatus.style.display = 'none';
                            labelInput.style.display = 'none';
                        }
                    }
                };
            }
        }
    } catch (error) {
        foodName.textContent = "เกิดข้อผิดพลาดในการเชื่อมต่อ: " + error.message;
        confidence.textContent = "ความมั่นใจ: -";
    }
}

// โหลดประวัติเมื่อเริ่มต้น
loadHistory();
