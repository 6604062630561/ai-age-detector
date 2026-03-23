# 🧠 IS Project: AI Age & MBTI Detector  
ระบบปัญญาประดิษฐ์สำหรับ  
- 🔹 คาดเดาอายุจากใบหน้า (Face Age Detection)  
- 🔹 วิเคราะห์บุคลิกภาพ MBTI จากข้อความ (Personality Prediction)  

พัฒนาโดยใช้ **Deep Learning + Machine Learning** และแสดงผลผ่านเว็บแอปพลิเคชัน

---

# 📌 หลักการทำงานของระบบ

## 🧠 1. Face Age Detection (Deep Learning)

ระบบใช้ **Convolutional Neural Network (CNN)** วิเคราะห์ภาพใบหน้า

### ขั้นตอน:
1️⃣ ผู้ใช้อัปโหลดภาพใบหน้า  
2️⃣ ระบบใช้ **Face Detection (MediaPipe)** ตรวจจับใบหน้า  
3️⃣ ทำการ **Crop + Resize + Normalize**  
4️⃣ โมเดล CNN ทำการ **Predict ช่วงอายุ**  
5️⃣ แสดงผล:

- ช่วงอายุ (Age Group)  
- อายุโดยประมาณ (Estimated Age)  
- ความมั่นใจ (Confidence)  
- กราฟ Probability  

---

## 🧩 2. MBTI Personality Prediction (Machine Learning)

ระบบใช้ **Natural Language Processing (NLP)** วิเคราะห์ข้อความ

### ขั้นตอน:
1️⃣ ผู้ใช้ป้อนข้อความ (เช่น ความคิดเห็น / โพสต์)  
2️⃣ ระบบทำ **Text Cleaning**
- lowercase  
- remove URL  
- remove special characters  

3️⃣ แปลงข้อความเป็นตัวเลขด้วย **TF-IDF**  
4️⃣ ใช้ **Logistic Regression (4 models)** ทำนาย  
5️⃣ รวมผลเป็น MBTI

---

# 🎯 MBTI ที่ระบบทำนาย

ระบบแบ่งเป็น 4 แกน (Binary Classification)

| Dimension | Meaning |
|----------|--------|
| I / E | Introvert / Extrovert |
| N / S | Intuition / Sensing |
| T / F | Thinking / Feeling |
| J / P | Judging / Perceiving |

👉 รวมกันเป็น 16 Personality Types เช่น  
- INTJ  
- ENFP  
- ISTP  

---

# 🎯 ช่วงอายุที่ระบบสามารถทำนายได้

| ช่วงอายุ | คำอธิบาย |
|--------|--------|
| Young | อายุประมาณ 0 - 20 ปี |
| Middle Age | อายุประมาณ 21 - 50 ปี |
| Old | อายุ 51 ปีขึ้นไป |

---

# 🧠 เทคโนโลยีที่ใช้

## 🔹 Deep Learning
- TensorFlow / Keras  
- CNN (Image Classification)

## 🔹 Machine Learning
- Scikit-learn  
- Logistic Regression  
- TF-IDF Vectorizer  

## 🔹 Computer Vision
- OpenCV  
- MediaPipe  

## 🔹 Web Application
- Streamlit  

## 🔹 Visualization
- Matplotlib  

---

# ⚙️ Feature ของระบบ

## 🔹 Input
- ภาพใบหน้า (Image)
- ข้อความ (Text)

## 🔹 Output

### Face Age:
- Age Group  
- Estimated Age  
- Confidence  
- Probability Graph  

### MBTI:
- Personality Type (เช่น INTJ)  
- Probability ของแต่ละแกน  
- กราฟแสดงผล  

---

# 📊 โครงสร้างโมเดล

## 🔹 Age Model (CNN)
- Input: Image (128x128x3)  
- Output: 3 classes (Young, Middle, Old)  

## 🔹 MBTI Model (ML)
- Input: Text → TF-IDF Vector  
- Model: Logistic Regression (4 ตัว)  
- Output: I/E, N/S, T/F, J/P  

---

# 👨‍💻 จัดทำโดย

- นาย อชิตพล แทนโป 6604062630561  
- นาย จุมพลภัทร์ สาเกกูล 6604062630111  

---

# 📚 Credit

Dataset:

- Age Detection  
https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset  
https://www.kaggle.com/datasets/dataturks/face-detection-in-images  

- MBTI Dataset  
https://www.kaggle.com/datasets/datasnaek/mbti-type  

---

# 🚀 วิธีใช้งาน

## 1️⃣ Clone โปรเจค

```bash
git clone https://github.com/6604062630561/ai-age-detector.git
cd ai-age-detector
