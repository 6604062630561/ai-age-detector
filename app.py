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
import itertools

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Age /MBTI Detector",
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
["🧠 Face Age Detector","📚 Face Neural Network","🧩 MBTI Predictor","📊 MBTI ML"],
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
@st.cache_resource
def load_mbti():

    import joblib

    vectorizer = joblib.load("vectorizer.pkl")
    models = joblib.load("mbti_models.pkl")

    return vectorizer, models

mbti_vectorizer, mbti_models = load_mbti()

# =====================================================
# TEXT PREPROCESS (MBTI)
# =====================================================
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


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

if page == "📚 Face Neural Network":

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

if page == "🧩 MBTI Predictor":

    st.markdown('<p class="big-title">MBTI PERSONALITY AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict Personality from Text</p>', unsafe_allow_html=True)

    text = st.text_area("Enter your thoughts / posts", height=200)

    def predict_mbti_topk(text, k=5):
        X = mbti_vectorizer.transform([clean_text(text)])
        probs = [model.predict_proba(X)[0][1] for model in mbti_models]

        # trait options
        trait_pairs = [
            ("I","E"),
            ("N","S"),
            ("T","F"),
            ("J","P")
        ]

        # probability สำหรับ trait ฝั่งแรก
        trait_probs = [
            [p, 1-p] for p in probs
        ]

        # สร้าง MBTI ทั้ง 16 แบบ
        all_mbti = list(itertools.product(*trait_pairs))

        mbti_scores = []
        for mbti in all_mbti:
            score = 1.0
            for i, t in enumerate(mbti):
                idx = 0 if t == trait_pairs[i][0] else 1
                score *= trait_probs[i][idx]
            mbti_scores.append(("".join(mbti), score))

        # เรียงและเลือก Top-k
        top_mbti = sorted(mbti_scores, key=lambda x: x[1], reverse=True)[:k]

        return top_mbti, probs

    if st.button("Analyze Personality"):

        if text.strip() == "":
            st.warning("Please enter text")
        else:

            top5, probs = predict_mbti_topk(text)

            # แสดง MBTI อันดับ 1
            st.markdown(f"""
            <div class="age-box">
            <div>Your MBTI</div>
            <div class="age-number">{top5[0][0]}</div>
            </div>
            """, unsafe_allow_html=True)

            # แสดง Top-5 MBTI
            st.markdown("### Top 5 MBTI Predictions")
            for mbti, score in top5:
                st.write(f"{mbti} → {score:.2%}")

            # แสดง probability ของแต่ละ trait
            st.markdown("### Trait Probabilities")

            labels = ["I vs E","N vs S","T vs F","J vs P"]
            left_probs = [p if p > 0.5 else 1-p for p in probs]
            right_probs = [1-p for p in left_probs]

            # horizontal stacked bar
            fig, ax = plt.subplots(figsize=(7,3))
            ax.barh(labels, left_probs, color="#4CAF50", label="Dominant Trait")
            ax.barh(labels, right_probs, left=left_probs, color="#FFC107", label="Opposite Trait")

            for i,(l,r) in enumerate(zip(left_probs,right_probs)):
                ax.text(l/2, i, f"{l:.0%}", va='center', ha='center', color='white', fontweight='bold')
                ax.text(l+r/2, i, f"{r:.0%}", va='center', ha='center', color='black', fontweight='bold')

            ax.set_xlim(0,1)
            ax.set_xlabel("Probability")
            ax.legend(loc="lower right")
            ax.invert_yaxis()  # ให้ trait อยู่ top-down
            st.pyplot(fig)

if page == "📊 MBTI ML":

    st.title("MBTI Model Explanation")

    st.header("Dataset")

    st.write("""
The dataset contains text posts labeled with MBTI personality types.

Each sample includes:

• Personality type (16 classes)  
• Multiple text posts combined  

Example:

"I enjoy deep thinking and abstract ideas"
""")

    st.header("Preprocessing")

    st.write("""
Text data is cleaned before training:

• Lowercase conversion  
• Remove URLs  
• Remove special characters  
• Tokenization  
""")

    st.header("Feature Extraction")

    st.write("""
TF-IDF (Term Frequency - Inverse Document Frequency) is used.

It converts text into numerical vectors based on word importance.

Key idea:

• Important words → higher weight  
• Common words → lower weight  
""")

    st.header("Machine Learning Model")

    st.write("""
The system uses Logistic Regression classifiers.

Instead of predicting 16 classes directly, it splits into 4 binary tasks:

• I vs E  
• N vs S  
• T vs F  
• J vs P  

Each model learns patterns independently.
""")

    st.header("Prediction Process")

    st.write("""
1 Input text  
2 Convert to TF-IDF vector  
3 Predict probability for each trait  
4 Combine results into MBTI type  

Example:

Text → [0.8, 0.7, 0.6, 0.9]  
Result → INTJ
""")

    st.header("Model Strength")

    st.write("""
• Fast prediction  
• Lightweight model  
• Easy deployment (pickle format)  
• Works well with text data  
""")

    st.header("Limitations")

    st.write("""
• Depends heavily on text quality  
• MBTI is not 100% scientifically accurate  
• Context understanding is limited compared to deep learning  
""")
