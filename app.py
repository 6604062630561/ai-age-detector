import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import mediapipe as mp

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Face Age Detector",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# MENU NAVIGATION
# =========================================================

page = st.radio(
    "",
    ["🧠 Face Age Detection", "📚 AI Model Explanation"],
    horizontal=True
)

# =========================================================
# STYLE + ANIMATION
# =========================================================

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
font-family:'Segoe UI',sans-serif;
animation: fadeIn 1s ease;
}
@keyframes fadeIn{
0%{opacity:0}
100%{opacity:1}
}
.big-title{
font-size:70px;
text-align:center;
font-weight:800;
color:white;
text-shadow:0 0 25px #00c6ff;
}
.subtitle{
text-align:center;
font-size:22px;
color:#d9f1ff;
margin-bottom:30px;
}
.result-card{
background:linear-gradient(135deg,#00c6ff,#0072ff);
padding:40px;
border-radius:25px;
text-align:center;
color:white;
box-shadow:0 0 40px rgba(0,200,255,0.8);
animation:pop 0.6s ease;
}
@keyframes pop{
0%{transform:scale(0.85)}
100%{transform:scale(1)}
}
.age-number{
font-size:90px;
font-weight:800;
}
.metric-box{
background:rgba(255,255,255,0.06);
padding:20px;
border-radius:15px;
margin-top:15px;
}
[data-testid="stMetricValue"]{
color:#00e5ff !important;
font-size:32px;
font-weight:bold;
}
[data-testid="stMetricLabel"]{
color:white !important;
font-size:18px;
}
.stProgress > div > div > div > div{
background-image:linear-gradient(90deg,#00c6ff,#0072ff);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_ai():
    return load_model("best_age_model.h5")

model = load_ai()

classes = [
"Middle Age (21-50)",
"Old (51+)",
"Young (0-20)"
]

age_values = [35,65,10]

# =========================================================
# FACE DETECTOR (High Accuracy)
# =========================================================

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(
model_selection=1,
min_detection_confidence=0.8
)

# =========================================================
# PAGE 1 : AGE DETECTOR
# =========================================================

if page == "🧠 Face Age Detection":

    st.markdown('<p class="big-title">AI FACE AGE DETECTOR</p>', unsafe_allow_html=True)

    st.markdown(
    '<p class="subtitle">Deep Learning System for Predicting Age from Facial Images</p>',
    unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg","jpeg","png"]
    )

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

                margin = 30

                x = max(0,x-margin)
                y = max(0,y-margin)
                bw = min(w-x,bw+margin*2)
                bh = min(h-y,bh+margin*2)

                cv2.rectangle(img,(x,y),(x+bw,y+bh),(0,255,0),3)

                face = img[y:y+bh, x:x+bw]

                # --------------------------------
                # IMAGE PREPROCESSING
                # --------------------------------

                face = cv2.GaussianBlur(face,(3,3),0)

                lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)

                clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                cl = clahe.apply(l)

                limg = cv2.merge((cl,a,b))
                face = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                face = cv2.resize(face,(128,128))

                face = face / 255.0

                face = np.reshape(face,(1,128,128,3))

                # --------------------------------
                # PREDICT
                # --------------------------------

                prediction = model.predict(face,verbose=0)

                predicted_class = classes[np.argmax(prediction)]

                confidence = float(np.max(prediction))*100

                estimated_age = int(
                    prediction[0][0]*age_values[0] +
                    prediction[0][1]*age_values[1] +
                    prediction[0][2]*age_values[2]
                )

                # --------------------------------
                # DISPLAY
                # --------------------------------

                with col1:

                    st.image(img,caption="Detected Face",use_column_width=True)

                with col2:

                    st.markdown(f"""
                    <div class="result-card">
                    <div>Estimated Age</div>
                    <div class="age-number">{estimated_age}</div>
                    <div>Years Old</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.metric("Age Group",predicted_class)

                    st.metric("AI Confidence",f"{confidence:.2f}%")

                    st.progress(int(confidence))

                    fig = plt.figure()

                    bars = plt.bar(classes,prediction[0])

                    for bar in bars:
                        bar.set_color("#00c6ff")

                    plt.title("Prediction Probability",color="white")
                    plt.xticks(color="white")
                    plt.yticks(color="white")

                    fig.patch.set_alpha(0)

                    st.pyplot(fig)

# =========================================================
# PAGE 2 : MODEL EXPLANATION
# =========================================================

if page == "📚 AI Model Explanation":

    st.title("AI Model Development")

    st.header("1. Data Preparation")

    st.write("""
The dataset used in this project consists of thousands of labeled facial images.
Preparation steps:
• Collect dataset from public sources  
• Remove corrupted or low-quality images  
• Resize all images to 128x128 pixels  
• Normalize pixel values (0-1)  
• Split dataset into training and testing sets
""")

    st.header("2. Machine Learning")

    st.write("""
Machine Learning allows computers to learn patterns from data.
In this project, the system learns relationships between facial features and human age.
Important features include:
• Facial structure  
• Skin texture  
• Wrinkles  
• Face proportions
""")

    st.header("3. Neural Network")

    st.write("""
The system uses a Convolutional Neural Network (CNN).
CNN is designed for image processing.
Main components:
• Convolution Layer – extract image features  
• Activation Layer – introduce non-linearity  
• Pooling Layer – reduce feature size  
• Fully Connected Layer – produce final prediction
""")

    st.header("4. Model Development Process")

    st.write("""
Step 1 – Data Collection  
Step 2 – Data Cleaning  
Step 3 – Image Preprocessing  
Step 4 – CNN Model Architecture Design  
Step 5 – Model Training using TensorFlow  
Step 6 – Model Evaluation and Accuracy Testing  
Step 7 – Web Deployment using Streamlit
""")

    st.header("5. Data Sources")

    st.write("""
• UTKFace Dataset  
• Kaggle Face Age Dataset  
""")

    st.header("Project Creators")

    st.write("""
Achitphon Thaenpo – 6604062630561  
Jumponpat Sakekun – 6604062630111
""")
