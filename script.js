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
const switchCameraButton = document.getElementById('switchCamera');

let currentStream = null;
let facingMode = 'user';

function startCamera(facing) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }
    
    const constraints = {
        video: {
            facingMode: facing
        }
    };
    
    navigator.mediaDevices.getUserMedia(constraints)
        .then(stream => {
            video.srcObject = stream;
            currentStream = stream;
            console.log(`📸 ใช้กล้อง: ${facing}`);
        })
        .catch(err => {
            console.error("เกิดข้อผิดพลาดในการเข้าถึงกล้อง: ", err);
            foodName.textContent = "ไม่สามารถเข้าถึงกล้องได้: " + err.message;
            switchCameraButton.disabled = true;
        });
}

startCamera(facingMode);

switchCameraButton.addEventListener('click', () => {
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    startCamera(facingMode);
    switchCameraButton.textContent = facingMode === 'user' ? '📷 สลับเป็นกล้องหลัง' : '📷 สลับเป็นกล้องหน้า';
});

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('foodHistory')) || [];
    foodHistory.innerHTML = '';
    history.forEach(item => {
        const li = document.createElement('li');
        const link = document.createElement('a');
        link.href = item.path; // ใช้ saved_path เป็น URL
        link.textContent = `${item.name} (${item.calories || '-'} kcal)`;
        link.target = '_blank'; // เปิดในแท็บใหม่
        link.style.textDecoration = 'none'; // ลบเส้นใต้ของลิงก์
        link.style.color = '#2d5a50'; // สีลิงก์ให้เข้ากับโทน
        link.addEventListener('mouseover', () => {
            link.style.color = '#219653'; // เปลี่ยนสีเมื่อ hover
        });
        link.addEventListener('mouseout', () => {
            link.style.color = '#2d5a50';
        });
        li.appendChild(link);
        foodHistory.appendChild(li);
    });
}

async function loadFoodList() {
    try {
        const response = await fetch("/food_list");
        const foodNames = await response.json();
        foodList.innerHTML = "";
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

function saveHistory(name, calories, path) {
    let history = JSON.parse(localStorage.getItem('foodHistory')) || [];
    history.unshift({ name, calories, path }); // เพิ่ม saved_path
    if (history.length > 5) history.pop();
    localStorage.setItem('foodHistory', JSON.stringify(history));
    loadHistory();
}

captureButton.addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/jpeg');
    photo.src = imageData;
    photo.style.display = 'block';
    sendImageToBackend(imageData);
});

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

resetButton.addEventListener('click', () => {
    photo.style.display = 'none';
    edgePhoto.style.display = 'none';
    foodName.textContent = "ชื่ออาหาร: -";
    confidence.textContent = "ความมั่นใจ: -";
    nutritionInfo.style.display = 'none';
    labelInput.style.display = 'none';
    trainingStatus.style.display = 'none';
});

let savedPath = null;

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

            savedPath = data.saved_path;
            console.log(`📂 บันทึก savedPath: ${savedPath}`);

            nutritionInfo.style.display = 'block';
            while (nutritionTable.rows.length > 1) nutritionTable.deleteRow(1);
            if (data.nutrition && Object.keys(data.nutrition).length > 0) {
                const units = { calories: "kcal" };
                for (const [key, value] of Object.entries(data.nutrition)) {
                    if (key !== "ingredients") {
                        const row = nutritionTable.insertRow();
                        row.insertCell(0).textContent = key === "calories" ? "พลังงาน" : key;
                        row.insertCell(1).textContent = `${value} ${units[key] || ''}`;
                    }
                }
                const row = nutritionTable.insertRow();
                row.insertCell(0).textContent = "วัตถุดิบ";
                row.insertCell(1).textContent = data.nutrition.ingredients ? data.nutrition.ingredients.join(", ") : "ไม่มีข้อมูล";
            } else {
                const row = nutritionTable.insertRow();
                row.insertCell(0).colSpan = 2;
                row.insertCell(0).textContent = "ไม่มีข้อมูลโภชนาการ";
            }

            saveHistory(data.food_name, data.nutrition.calories || null, savedPath); // เก็บ saved_path

            if (data.needs_label) {
                alert(`โมเดลไม่แน่ใจ อาจเป็น "${data.food_name}" หรืออาหารอื่น กรุณาระบุชื่อที่ถูกต้อง`);
                labelInput.style.display = 'block';
            }
            submitLabel.onclick = async () => {
                const newLabelValue = newLabel.value.trim();
                const calories = document.getElementById('calories').value;
                console.log(`📤 ค่า input: label=${newLabelValue}, calories=${calories}`);
            
                if (!newLabelValue) {
                    alert("❌ กรุณากรอกหรือเลือกชื่ออาหาร");
                    return;
                }
            
                if (!savedPath) {
                    alert("❌ ไม่พบ path ของรูปภาพ กรุณาถ่ายภาพใหม่");
                    return;
                }
            
                try {
                    const labelResponse = await fetch('/update_label', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            path: savedPath,
                            label: newLabelValue,
                            nutrition: calories ? { calories: parseInt(calories), ingredients: ["ไม่มีข้อมูล"] } : {}
                        })
                    });
            
                    if (!labelResponse.ok) throw new Error('การอัพเดทโมเดลล้มเหลว');
                    const labelData = await labelResponse.json();
                    console.log(`✅ Response:`, labelData);
            
                    alert(labelData.message);
                    trainingStatus.style.display = 'block';
                    labelInput.style.display = 'none';
            
                    setTimeout(() => {
                        trainingStatus.style.display = 'none';
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

loadHistory();
