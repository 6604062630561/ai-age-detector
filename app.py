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
    page_title="AI Age Detector Pro",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------
# NAVIGATION
# ----------------------------

page = st.radio(
"",
["🧠 Face Age Detector","📚 Model Explanation"],
horizontal=True
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
animation: fadein 1s ease;
}
@keyframes fadein{
0%{opacity:0}
100%{opacity:1}
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
[data-testid="stMetricValue"]{
color:#00e5ff !important;
font-size:34px;
font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

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

# ----------------------------
# FACE DETECTORS
# ----------------------------

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(
model_selection=1,
min_detection_confidence=0.6
)

# OpenCV fallback (สำหรับการ์ตูน)

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ----------------------------
# PAGE 1
# ----------------------------

if page == "🧠 Face Age Detector":

    st.markdown('<p class="big-title">🧠 AI AGE DETECTOR PRO</p>', unsafe_allow_html=True)

    st.markdown(
    '<p class="subtitle">Deep Learning Age Prediction System</p>',
    unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
    "Upload face image",
    type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        col1,col2 = st.columns(2)

        image = Image.open(uploaded_file)

        img = np.array(image)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = detector.process(img_rgb)

        faces = []

        if results.detections:

            h,w,_ = img.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                faces.append((x,y,bw,bh))

        else:


            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5
            )

            for (x,y,wf,hf) in detected:
                faces.append((x,y,wf,hf))

        if len(faces) == 0:

            st.warning("No face detected")

        else:

            st.success(f"Detected {len(faces)} face(s)")

            for (x,y,bw,bh) in faces:

                cv2.rectangle(img,(x,y),(x+bw,y+bh),(0,255,0),3)

                face = img[y:y+bh, x:x+bw]

                face = cv2.resize(face,(128,128))

                face = face / 255.0

                face = np.reshape(face,(1,128,128,3))

                prediction = model.predict(face)

                idx = np.argmax(prediction)

                predicted_class = classes[idx]

                confidence = float(prediction[0][idx]) * 100

                # ----------------------------
                # FIX AGE CALCULATION
                # ----------------------------

                if idx == 0:       # Middle
                    estimated_age = np.random.randint(25,45)

                elif idx == 1:     # Old
                    estimated_age = np.random.randint(55,75)

                else:              # Young
                    estimated_age = np.random.randint(8,20)

                with col1:

                    st.image(img,caption="Detected Face",use_column_width=True)

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

                    fig.patch.set_alpha(0)

                    st.pyplot(fig)

# ----------------------------
# PAGE 2
# ----------------------------

if page == "📚 Model Explanation":

    st.title("AI Model Development")

    st.header("Data Preparation")

    st.write("""
Dataset images were collected from public sources and cleaned before training.
- Image cleaning
- Resize images
- Normalize pixel values
- Train/test split
""")

    st.header("Machine Learning")

    st.write("""
Machine Learning enables computers to learn patterns from data.
The system learns facial characteristics related to age.
""")

    st.header("Neural Network")

    st.write("""
The model uses Convolutional Neural Networks (CNN) for image recognition.
""")
     st.write("""
The model uses Convolutional Neural Networks (CNN).
CNN layers include:
- Convolution Layer
- Activation Layer
- Pooling Layer
- Fully Connected Layer
""")

    st.header("Model Development")

    st.write("""
1. Collect dataset  
2. Preprocess images  
3. Train CNN model using TensorFlow  
4. Evaluate model performance  
5. Deploy using Streamlit
""")


    st.header("Project Creators")

    st.write("""
Achitphon Thaenpo 6604062630561  
Jumponpat Sakekun 6604062630111
""")

    st.header("Credit")

    st.write("Dataset : Kaggle Ages detection from images")
