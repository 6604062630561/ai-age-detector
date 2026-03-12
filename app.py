import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import mediapipe as mp
import time
import platform
import tensorflow as tf

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Age Detector Pro",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#141e30,#243b55);
color:white;
font-family:'Segoe UI';
animation: fadeIn 1.2s ease;
}

/* FADE ANIMATION */

@keyframes fadeIn{
0%{opacity:0;}
100%{opacity:1;}
}

/* TITLE */

.big-title{
font-size:64px;
text-align:center;
font-weight:800;
text-shadow:0px 0px 30px rgba(0,200,255,0.8);
animation: glow 2s ease-in-out infinite alternate;
}

/* TITLE GLOW */

@keyframes glow{
from{
text-shadow:0 0 20px #00c6ff;
}
to{
text-shadow:0 0 45px #00c6ff,0 0 60px #0072ff;
}
}

/* SUBTITLE */

.subtitle{
text-align:center;
font-size:22px;
margin-bottom:25px;
color:#d0e6ff;
animation: slideDown 1s ease;
}

@keyframes slideDown{
0%{
opacity:0;
transform:translateY(-20px);
}
100%{
opacity:1;
transform:translateY(0);
}
}

/* AGE BOX */

.age-box{
background: linear-gradient(135deg,#00c6ff,#0072ff);
padding:40px;
border-radius:20px;
text-align:center;
box-shadow:0px 0px 30px rgba(0,200,255,0.7);
animation: floatBox 3s ease-in-out infinite;
transition: transform 0.3s;
}

/* HOVER EFFECT */

.age-box:hover{
transform:scale(1.05);
}

/* FLOAT */

@keyframes floatBox{
0%{transform:translateY(0px);}
50%{transform:translateY(-10px);}
100%{transform:translateY(0px);}
}

/* AGE NUMBER */

.age-number{
font-size:80px;
font-weight:800;
animation: pulse 2s infinite;
}

/* PULSE */

@keyframes pulse{
0%{transform:scale(1);}
50%{transform:scale(1.05);}
100%{transform:scale(1);}
}

/* LABEL */

.age-label{
font-size:22px;
}

/* SIDEBAR */

section[data-testid="stSidebar"]{
background:#0f172a;
}

/* BUTTON */

.stButton>button{
background:linear-gradient(90deg,#00c6ff,#0072ff);
border:none;
color:white;
font-weight:600;
border-radius:10px;
padding:10px 20px;
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
box-shadow:0 0 20px #00c6ff;
}

/* PROGRESS BAR */

.stProgress > div > div > div{
background-image: linear-gradient(90deg,#00c6ff,#0072ff);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# NAVIGATION
# =====================================================

page = st.radio(
"",
["🧠 Face Age Detector","📚 Model Explanation"],
horizontal=True
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_ai():

    model = load_model("best_age_model.h5")

    # warmup model (ลด lag ครั้งแรก)
    dummy = np.zeros((1,128,128,3))
    model.predict(dummy,verbose=0)

    return model


with st.spinner("Loading AI Model..."):
    model = load_ai()

# =====================================================
# CLASS LABELS
# =====================================================

classes = [
"Middle Age (21-50)",
"Old (51+)",
"Young (0-20)"
]

# =====================================================
# MEDIAPIPE FACE DETECTOR
# =====================================================

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(
model_selection=1,
min_detection_confidence=0.65
)

# =====================================================
# OPENCV FALLBACK DETECTOR
# =====================================================

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================================================
# IMAGE PREPROCESSING
# =====================================================

def preprocess_face(face):

    face = cv2.resize(face,(128,128))

    face = face.astype("float32")/255.0

    face = np.reshape(face,(1,128,128,3))

    return face

# =====================================================
# AGE ESTIMATION LOGIC (IMPROVED)
# =====================================================

def estimate_age(index,confidence):

    if index == 0:
        base = 35

    elif index == 1:
        base = 65

    else:
        base = 15

    noise = int((1-confidence/100)*12)

    age = base + np.random.randint(-noise,noise+1)

    return max(age,1)

# =====================================================
# SYSTEM INFO
# =====================================================

def show_system_info():

    st.sidebar.title("System Info")

    st.sidebar.write("Python:", platform.python_version())
    st.sidebar.write("TensorFlow:", tf.__version__)
    st.sidebar.write("Platform:", platform.system())
    st.sidebar.write("Processor:", platform.processor())

show_system_info()

# =====================================================
# PAGE 1 : DETECTOR
# =====================================================

if page == "🧠 Face Age Detector":

    st.markdown('<p class="big-title">AI AGE DETECTOR PRO</p>', unsafe_allow_html=True)

    st.markdown('<p class="subtitle">Deep Learning Face Age Prediction</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload face image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        start_time = time.time()

        # FIX RGB ERROR
        image = Image.open(uploaded_file).convert("RGB")

        img = np.array(image)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = detector.process(img_rgb)

        faces = []

        h,w,_ = img.shape

        if results.detections:

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # FACE MARGIN (เพิ่มความแม่น)
                margin = int(bw*0.2)

                x = max(0,x-margin)
                y = max(0,y-margin)

                bw = min(w-x,bw+margin*2)
                bh = min(h-y,bh+margin*2)

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

            col1,col2 = st.columns(2)

            for (x,y,bw,bh) in faces:

                cv2.rectangle(img,(x,y),(x+bw,y+bh),(0,255,0),3)

                face = img[y:y+bh, x:x+bw]

                face_input = preprocess_face(face)

                prediction = model.predict(face_input,verbose=0)

                idx = np.argmax(prediction)

                predicted_class = classes[idx]

                confidence = float(prediction[0][idx]) * 100

                estimated_age = estimate_age(idx,confidence)

                with col1:

                    st.image(img, caption="Detected Face", use_column_width=True)

                with col2:

                    st.markdown("### AI Prediction")

                    st.markdown(f"""
                    <div class="age-box">
                    <div class="age-label">Estimated Age</div>
                    <div class="age-number">{estimated_age}</div>
                    <div class="age-label">Years Old</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("Age Group:", predicted_class)

                    st.metric("AI Confidence", f"{confidence:.2f}%")

                    st.progress(int(confidence))

                    fig = plt.figure()

                    bars = plt.bar(classes,prediction[0])

                    for bar in bars:
                        bar.set_color("#00c6ff")

                    plt.title("Prediction Probability")

                    st.pyplot(fig)

        end_time = time.time()

        st.write("Processing Time:", round(end_time-start_time,2),"seconds")

# =====================================================
# PAGE 2 : EXPLANATION
# =====================================================

if page == "📚 Model Explanation":

    st.title("AI Model Development")

    st.header("Data Preparation")

    st.write("""
Dataset images were collected from public sources and cleaned before training.
Steps include:

• Image cleaning  
• Face alignment  
• Resize images  
• Pixel normalization  
• Train/test split
""")

    st.header("Machine Learning")

    st.write("""
Machine Learning enables computers to learn patterns from data.

In this project the system learns relationships between:

• Facial texture  
• Wrinkles  
• Skin smoothness  
• Face structure
""")

    st.header("Neural Network")

    st.write("""
The model uses Convolutional Neural Networks (CNN).
CNN layers include:

• Convolution Layer  
• Activation Layer  
• Pooling Layer  
• Fully Connected Layer
""")

    st.header("Model Development")

    st.write("""
1 Collect dataset  
2 Preprocess images  
3 Train CNN model using TensorFlow  
4 Evaluate model performance  
5 Deploy using Streamlit
""")

    st.header("Project Creators")

    st.write("""
Achitphon Thaenpo 6604062630561  
Jumponpat Sakekun 6604062630111
""")

    st.header("Credit")

    st.write("Dataset : Kaggle Age Detection Dataset")
