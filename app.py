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
    page_title="AI Age Detector",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#141e30,#243b55);
color:#E8F6FF;
font-family:'Segoe UI';
}

h1,h2,h3,h4,p,label,span{
color:#E8F6FF !important;
}

.big-title{
font-size:64px;
text-align:center;
font-weight:800;
}

.subtitle{
text-align:center;
font-size:22px;
margin-bottom:25px;
color:#bde6ff;
}

.age-box{
background: linear-gradient(135deg,#00c6ff,#0072ff);
padding:30px;
border-radius:20px;
text-align:center;
box-shadow:0px 0px 30px rgba(0,200,255,0.7);
margin-bottom:10px;
}

.age-number{
font-size:70px;
font-weight:800;
color:white;
}

/* FIX: ทำให้ Face Results table มองเห็น */

[data-testid="stTable"]{
color:#FFFFFF !important;
}

[data-testid="stTable"] *{
color:#FFFFFF !important;
}

.stDataFrame{
color:white !important;
}

[data-testid="stSidebar"]{
background:#101a2b;
}

[data-testid="stSidebar"] *{
color:#E8F6FF !important;
}


/* CREATOR GRID */

.creator-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:30px;
margin-top:20px;
}

.creator-card{
background:linear-gradient(135deg,#00c6ff,#0072ff);
padding:25px;
border-radius:18px;
text-align:center;
box-shadow:0px 0px 20px rgba(0,200,255,0.5);
}

.creator-card h3{
color:white !important;
font-size:22px;
margin-bottom:8px;
}

.creator-card p{
color:white !important;
font-size:18px;
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

    dummy = np.zeros((1,128,128,3))
    model.predict(dummy,verbose=0)

    return model

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
# FACE DETECTOR
# =====================================================

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(
model_selection=1,
min_detection_confidence=0.65
)

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =====================================================
# PREPROCESS
# =====================================================

def preprocess_face(face):

    face = cv2.resize(face,(128,128))

    kernel = np.array([
    [0,-1,0],
    [-1,5,-1],
    [0,-1,0]
    ])

    face = cv2.filter2D(face,-1,kernel)

    # เพิ่มตรงนี้
    face = cv2.bilateralFilter(face,5,50,50)

    face = face / 255.0

    face = np.expand_dims(face,axis=0)

    return face

# =====================================================
# AGE ESTIMATION
# =====================================================

def estimate_age(pred):

    young = pred[2]
    middle = pred[0]
    old = pred[1]

    age = (
    young * 8 +
    middle * 32 +
    old * 65
    )

    return int(age)

# =====================================================
# SYSTEM INFO
# =====================================================

st.sidebar.title("System Info")
st.sidebar.write("Python:", platform.python_version())
st.sidebar.write("TensorFlow:", tf.__version__)
st.sidebar.write("Platform:", platform.system())

# =====================================================
# PAGE 1
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

        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)

        img_rgb = img.copy()
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

                faces.append((x,y,bw,bh))

        if len(faces)==0:

            st.warning("No face detected")

        else:

            results_list = []
            probability_data = []

            for i,(x,y,bw,bh) in enumerate(faces):

                face = img[y:y+bh, x:x+bw]

                face_input = preprocess_face(face)

                p1 = model.predict(face_input,verbose=0)[0]

                flip = np.expand_dims(cv2.flip(face_input[0],1),0)
                p2 = model.predict(flip,verbose=0)[0]

                bright = np.clip(face_input*1.2,0,1)
                p3 = model.predict(bright,verbose=0)[0]

                dark = np.clip(face_input*0.8,0,1)
                p4 = model.predict(dark,verbose=0)[0]

                blur = np.expand_dims(cv2.GaussianBlur(face_input[0],(3,3),0),0)
                p5 = model.predict(blur,verbose=0)[0]

                # FIX: smoothing ใหม่
                prediction = (p1 + p2 + p3 + p4 + p5 + p1) / 6

                probability_data.append(prediction)

                # FIX: bias correction
                if prediction[2] > 0.65:
                    idx = 2
                elif prediction[1] > 0.70:
                    idx = 1
                else:
                    idx = np.argmax(prediction)

                predicted_class = classes[idx]

                confidence = float(prediction[idx]) * 100

                estimated_age = estimate_age(prediction)

                results_list.append({
                    "Face": i+1,
                    "Age": estimated_age,
                    "Group": predicted_class,
                    "Confidence": round(confidence,2)
                })

                label = f"Face {i+1} | {estimated_age} yrs"

                cv2.rectangle(img,(x,y),(x+bw,y+bh),(0,255,0),3)

                cv2.putText(
                    img,
                    label,
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2
                )

            col1,col2 = st.columns(2)

            with col1:
                st.image(img, caption="Detected Faces", use_column_width=True)

            with col2:

                st.markdown("### Face Results")
                st.table(results_list)

                st.markdown("### Estimated Age")

                cols = st.columns(len(results_list))

                for i, face in enumerate(results_list):

                    with cols[i]:

                        st.markdown(f"""
                        <div class="age-box">
                        <div>Face {face['Face']}</div>
                        <div class="age-number">{face['Age']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("### Age Prediction Probability")

                labels = ["Middle Age","Old","Young"]

                for i, prob in enumerate(probability_data):

                    fig, ax = plt.subplots()

                    ax.bar(labels, prob)

                    ax.set_ylim(0,1)

                    for j,v in enumerate(prob):
                        ax.text(j,v+0.02,f"{v:.2f}",ha="center")

                    st.pyplot(fig)

        end_time = time.time()

        st.write("Processing Time:", round(end_time-start_time,2),"seconds")

# =====================================================
# PAGE 2
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

The system analyzes:

• Facial texture  
• Wrinkles  
• Skin smoothness  
• Face structure
""")

    st.header("Neural Network")

    st.write("""
The model uses Convolutional Neural Networks (CNN).

Layers include:

• Convolution Layer  
• Activation Layer  
• Pooling Layer  
• Fully Connected Layer
""")

    st.header("Model Development")

    st.write("""
1 Collect dataset  
2 Preprocess images  
3 Train CNN model  
4 Evaluate performance  
5 Deploy with Streamlit
""")

    st.header("Credit")

    st.write("Dataset : Kaggle Age Detection from Images")

    st.markdown("""

<div class="creator-grid">

<div class="creator-card">
<h3>Achitphon Thaenpo</h3>
<p>6604062630561</p>
</div>

<div class="creator-card">
<h3>Jumponpat Sakekun</h3>
<p>6604062630111</p>
</div>

</div>

""", unsafe_allow_html=True)
