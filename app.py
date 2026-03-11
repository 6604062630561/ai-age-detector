import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import mediapipe as mp

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="AI Age Detector",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# STYLE
# ----------------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#141e30,#243b55);
color:#ffffff;
font-family:'Segoe UI',sans-serif;
}

.big-title{
font-size:64px;
text-align:center;
font-weight:800;
color:white;
text-shadow:0px 0px 25px rgba(0,150,255,0.9);
}

.subtitle{
text-align:center;
font-size:22px;
color:#cfd9df;
margin-bottom:30px;
}

.card{
background: rgba(255,255,255,0.07);
padding:25px;
border-radius:20px;
backdrop-filter: blur(12px);
box-shadow:0px 0px 25px rgba(0,0,0,0.6);
}

.age-box{
background: linear-gradient(135deg,#00c6ff,#0072ff);
padding:40px;
border-radius:25px;
text-align:center;
color:white;
box-shadow:0px 0px 40px rgba(0,200,255,0.8);
margin-bottom:20px;
}

.age-number{
font-size:85px;
font-weight:800;
}

.age-label{
font-size:24px;
color:#eaf6ff;
}

/* FIX AI CONFIDENCE COLOR */

[data-testid="stMetricValue"]{
color:#00e5ff !important;
font-size:34px;
font-weight:bold;
}

[data-testid="stMetricLabel"]{
color:#ffffff !important;
font-size:18px;
}

[data-testid="stSidebar"]{
background: linear-gradient(180deg,#0f2027,#203a43);
}

.stProgress > div > div > div > div{
background-image: linear-gradient(90deg,#00c6ff,#0072ff);
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER
# ----------------------------

st.markdown('<p class="big-title">🧠 AI AGE DETECTOR</p>', unsafe_allow_html=True)

st.markdown(
'<p class="subtitle">Deep Learning Age Prediction System</p>',
unsafe_allow_html=True
)

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_ai():
    return load_model("best_age_model.h5")

with st.spinner("Loading AI Model..."):
    model = load_ai()

classes = [
"Middle Age (21-50)",
"Old (51+)",
"Young (0-20)"
]

age_values = [35,65,10]

# ----------------------------
# MEDIAPIPE FACE DETECTION
# ----------------------------

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(
model_selection=1,
min_detection_confidence=0.6
)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("⚙️ System Info")

st.sidebar.success("AI Model Loaded")

st.sidebar.write("Age Classes")

st.sidebar.write(classes)

# ----------------------------
# UPLOAD IMAGE
# ----------------------------

uploaded_file = st.file_uploader(
"Upload face image",
type=["jpg","jpeg","png"]
)

# ----------------------------
# MAIN PROCESS
# ----------------------------

if uploaded_file:

    col1,col2 = st.columns(2)

    image = Image.open(uploaded_file)

    img = np.array(image)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = detector.process(img_rgb)

    if not results.detections:

        st.warning("No face detected")

    else:

        h,w,_ = img.shape

        st.success(f"Detected {len(results.detections)} face(s)")

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            cv2.rectangle(img,(x,y),(x+bw,y+bh),(0,255,0),3)

            face = img[y:y+bh, x:x+bw]

            face = cv2.resize(face,(128,128))

            face = face / 255.0

            face = np.reshape(face,(1,128,128,3))

            prediction = model.predict(face)

            predicted_class = classes[np.argmax(prediction)]

            confidence = np.max(prediction) * 100

            estimated_age = int(
                prediction[0][0]*age_values[0] +
                prediction[0][1]*age_values[1] +
                prediction[0][2]*age_values[2]
            )

            # IMAGE DISPLAY

            with col1:

                st.image(img,caption="Detected Face",use_column_width=True)

            # RESULT PANEL

            with col2:

                st.markdown("### 🤖 AI Prediction")

                st.markdown(f"""
                <div class="age-box">
                <div class="age-label">Estimated Age</div>
                <div class="age-number">{estimated_age}</div>
                <div class="age-label">Years Old</div>
                </div>
                """, unsafe_allow_html=True)

                st.write(f"**Age Group:** {predicted_class}")

                st.metric("AI Confidence", f"{confidence:.2f}%")

                st.progress(int(confidence))

                fig = plt.figure()

                bars = plt.bar(classes,prediction[0])

                for bar in bars:
                    bar.set_color("#00c6ff")

                plt.title("AI Prediction Probability",color="white")
                plt.ylabel("Confidence",color="white")
                plt.xticks(color="white")
                plt.yticks(color="white")

                fig.patch.set_alpha(0)

                st.pyplot(fig)

# ----------------------------
# ABOUT
# ----------------------------

st.markdown("---")

st.markdown("## 📖 About This AI")

st.write("""

This project demonstrates how Artificial Intelligence can estimate age groups from facial images.

Workflow

1️⃣ Upload image  
2️⃣ Detect face using MediaPipe  
3️⃣ Extract facial features  
4️⃣ Predict age group using CNN model  

The model was trained using Transfer Learning with MobileNetV2.

""")

# ----------------------------
# CREDIT
# ----------------------------

st.markdown("---")

st.markdown("""

### 👨‍💻 Project Creators

Achitphon Theanpo  6604062630561
Jumponpat Sakekun  6604062630111

### ☁️ Model Training

Trained using Google Colab GPU

### 📂 Dataset

Dataset from Kaggle Face Age Dataset

""")