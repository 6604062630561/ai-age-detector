# 🧠 IS Project AI Age Detector
ระบบปัญญาประดิษฐ์สำหรับคาดเดาช่วงอายุของบุคคลจากภาพใบหน้า  
พัฒนาโดยใช้เทคนิค Deep Learning และแสดงผลผ่านเว็บแอปพลิเคชัน

---

# 📌 หลักการทำงานของระบบ

ระบบนี้ใช้ **Deep Learning (Convolutional Neural Network)** ในการวิเคราะห์ภาพใบหน้า  
เพื่อคาดเดาช่วงอายุของบุคคลจากลักษณะของใบหน้า

ขั้นตอนการทำงานของระบบมีดังนี้

1️⃣ ผู้ใช้ทำการอัปโหลดภาพใบหน้าเข้าสู่ระบบ  

2️⃣ ระบบจะใช้ **Face Detection** เพื่อตรวจจับตำแหน่งของใบหน้าในภาพ  

3️⃣ ระบบจะทำการ **Crop และปรับขนาดภาพใบหน้า** ให้เหมาะสมสำหรับโมเดล AI  

4️⃣ โมเดล Deep Learning ที่ผ่านการฝึกแล้วจะทำการ **Predict ช่วงอายุ**

5️⃣ ระบบจะแสดงผลลัพธ์ดังนี้

- ช่วงอายุ (Age Group)
- อายุโดยประมาณ (Estimated Age)
- ความมั่นใจของโมเดล (Confidence)
- กราฟความน่าจะเป็นของแต่ละช่วงอายุ

---

# 🎯 ช่วงอายุที่ระบบสามารถทำนายได้

ระบบแบ่งช่วงอายุออกเป็น 3 กลุ่ม

| ช่วงอายุ | คำอธิบาย |
|--------|--------|
| Young | อายุประมาณ 0 - 20 ปี |
| Middle Age | อายุประมาณ 21 - 50 ปี |
| Old | อายุ 51 ปีขึ้นไป |

---

# 🧠 เทคโนโลยีที่ใช้

โปรเจคนี้พัฒนาด้วยเทคโนโลยีดังต่อไปนี้

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe (Face Detection)
- Streamlit (Web Application)
- Matplotlib (Visualization)


จัดทำโดย:
นาย อชิตพล แทนโป 6604062630561<br>
นาย จุมพลภัทร์ สาเกกูล 6604062630111

Credit:<br>
Dataset : https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset#<br>
: https://www.kaggle.com/datasets/dataturks/face-detection-in-images#<br>
feature:<br>
Input:
Image → pixel values (numeric features จำนวนมาก)<br>
Output:
Class → YOUNG / MIDDLE / OLD

---


## 1️⃣ Clone โปรเจคจาก GitHub

```bash
git clone https://github.com/6604062630561/ai-age-detector.git

