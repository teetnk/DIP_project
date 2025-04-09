/*
Development by :
Thanakorn Sutakiatsakul
Piyachok Ridsadaeng
*/
/* ==================== พื้นฐานของเว็บ ==================== */
body {
    font-family: 'Kanit', Arial, sans-serif;
    background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); /* ปรับสีพื้นหลังให้สดใสขึ้นเล็กน้อย */
    text-align: center;
    margin: 0;
    padding: 30px; /* เพิ่ม padding เพื่อให้มีระยะห่างจากขอบ */
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center; /* จัดให้เนื้อหาอยู่กึ่งกลางแนวตั้ง */
    overflow-x: hidden; /* ป้องกัน scroll bar แนวนอน */
}

/* ==================== หัวข้อ ==================== */
h1 {
    color: #1a3c34; /* เปลี่ยนสีให้เข้มขึ้นและเข้ากับโทน */
    font-size: 2.2em; /* ลดขนาดเล็กน้อยให้สมดุล */
    margin-bottom: 25px;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.05); /* ลดความเข้มของเงา */
}

h2 {
    color: #2d5a50; /* ปรับสีให้เข้ากับ h1 */
    font-size: 1.4em;
    margin: 25px 0;
}

/* ==================== ตารางข้อมูล ==================== */
table {
    border-collapse: collapse;
    width: 85%; /* ปรับให้กว้างขึ้นเล็กน้อย */
    max-width: 600px;
    margin: 25px auto;
    border-radius: 12px;
    overflow: hidden;
    background-color: rgba(255, 255, 255, 0.95); /* เพิ่มความโปร่งใสเล็กน้อย */
    backdrop-filter: blur(5px); /* เพิ่มความทันสมัย */
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08); /* ปรับเงาให้ดูนุ่มนวล */
}

th, td {
    border: 1px solid #e5e5e5;
    padding: 12px 15px;
    font-size: 1em; /* ลดขนาดฟอนต์ให้สมดุล */
}

th {
    background-color: #2d5a50; /* เปลี่ยนสีให้เข้ากับโทน h1, h2 */
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}

td {
    color: #333;
    text-align: left;
}

td:nth-child(2) { /* ปรับคอลัมน์ "ปริมาณ" ให้อยู่กึ่งกลาง */
    text-align: center;
}

tr:nth-child(even) {
    background-color: #f8fafc;
}

tr:hover {
    background-color: #e9ecef;
    transition: background-color 0.3s ease;
}

/* ==================== ส่วนแสดงผลข้อมูล ==================== */
.result-container {
    margin-top: 30px;
    background-color: rgba(255, 255, 255, 0.92);
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
    width: 85%;
    max-width: 600px;
    backdrop-filter: blur(5px); /* เพิ่มความทันสมัย */
}

.confidence-green {
    color: #219653; /* สีเขียวที่ทันสมัยขึ้น */
    font-weight: 600;
}

.confidence-orange {
    color: #f39c12; /* สีส้มที่สดใสขึ้น */
    font-weight: 600;
}

/* ==================== ปุ่มและกล้อง ==================== */
button {
    margin: 12px;
    padding: 12px 25px;
    font-size: 1em;
    cursor: pointer;
    border: none;
    border-radius: 30px; /* ปรับให้โค้งมนมากขึ้น */
    color: white;
    min-width: 120px; /* เพิ่มขนาดขั้นต่ำให้ปุ่มสมดุล */
    transition: background-color 0.3s, transform 0.2s, box-shadow 0.2s;
}

button:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
    filter: brightness(1.1); /* เพิ่มความสว่างเมื่อ hover */
}

button:active {
    transform: translateY(0);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

#capture {
    background-color: #007bff;
}

#uploadButton {
    background-color: #28a745;
}

#resetButton {
    background-color: #dc3545;
}

#switchCamera {
    background-color: #6c757d; /* สีเทาสำหรับปุ่มสลับกล้อง */
}

#submitLabel {
    background-color: #f39c12; /* ปรับสีให้เข้ากับ confidence-orange */
}

video, img#photo {
    max-width: 90%;
    width: 100%;
    border: 2px solid #e0e0e0;
    border-radius: 15px;
    margin-bottom: 25px;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
}

#edgePhoto {
    max-width: 90%;
    width: 100%;
    border: 2px solid #3498db;
    border-radius: 15px;
    margin-top: 25px;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.05);
}

/* ==================== ส่วนกรอกชื่ออาหาร ==================== */
#labelInput {
    display: none;
    margin-top: 25px;
    background-color: rgba(255, 245, 245, 0.95); /* ปรับสีให้อ่อนลง */
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.06);
    width: 85%;
    max-width: 500px;
    border-left: 5px solid #e74c3c;
    backdrop-filter: blur(5px);
}

#newLabel, #calories {
    padding: 10px;
    font-size: 1em;
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    width: 70%;
    margin: 10px 5px;
    background-color: #f8fafc; /* เพิ่มสีพื้นหลังอ่อน ๆ */
    transition: border-color 0.3s, box-shadow 0.3s;
}

#newLabel:focus, #calories:focus {
    border-color: #f39c12;
    box-shadow: 0 0 5px rgba(243, 156, 18, 0.3);
    outline: none;
}

/* ==================== Training Status และ History ==================== */
#trainingStatus {
    display: none;
    margin-top: 25px;
    padding: 12px;
    background-color: rgba(52, 152, 219, 0.15);
    border-radius: 8px;
    color: #1e90ff;
    font-weight: 500;
    border: 1px solid rgba(52, 152, 219, 0.3);
}

/* ... (โค้ดเดิมด้านบนจนถึง #historyContainer) ... */

#historyContainer {
    margin-top: 30px;
    width: 85%;
    max-width: 600px;
}

#foodHistory li {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 12px 15px;
    margin: 8px 0;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    text-align: left;
    transition: transform 0.2s;
}

#foodHistory li:hover {
    transform: translateX(5px);
}

#foodHistory a {
    text-decoration: none;
    color: #2d5a50;
    font-weight: 500;
    transition: color 0.3s;
}

#foodHistory a:hover {
    color: #219653;
}

/* ... (โค้ดเดิมด้านล่างจนถึง @media และ animation) ... */

/* ==================== Responsive Design ==================== */
@media (max-width: 768px) {
    body {
        padding: 20px;
    }

    table, .result-container, #labelInput, #historyContainer {
        width: 95%;
    }

    button {
        padding: 10px 20px;
        font-size: 0.9em;
        min-width: 100px;
    }

    h1 {
        font-size: 1.8em;
    }

    h2 {
        font-size: 1.2em;
    }

    #newLabel, #calories {
        width: 100%; /* ปรับให้เต็มความกว้างบนหน้าจอเล็ก */
        margin: 8px 0;
    }
}

/* ==================== Animation ==================== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-container, #nutritionInfo, #labelInput, #edgePhoto, #trainingStatus {
    animation: fadeIn 0.6s ease-out;
}

/* 📌 สไตล์ของกล่องยืนยันอาหาร */
#confirmationContainer {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 9999;
    text-align: center;
    width: 90%;
    max-width: 400px;
}

/* 📌 สไตล์ของปุ่มยืนยัน */
.btn-confirm {
    background: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    margin: 5px;
    cursor: pointer;
    border-radius: 5px;
    font-size: 16px;
    transition: all 0.3s ease;
}

.btn-confirm:hover {
    opacity: 0.8;
}

/* 📌 ปรับแต่ง input ของ labelInput */
#labelInput {
    display: none;
    text-align: center;
    margin-top: 15px;
}

#labelInput input {
    padding: 8px;
    width: 80%;
    border-radius: 5px;
    border: 1px solid #ccc;
}

#labelInput button {
    background: #2ecc71;
    color: white;
    border: none;
    padding: 8px 15px;
    margin-left: 5px;
    cursor: pointer;
    border-radius: 5px;
    font-size: 14px;
    transition: all 0.3s ease;
}

#labelInput button:hover {
    opacity: 0.8;
}
