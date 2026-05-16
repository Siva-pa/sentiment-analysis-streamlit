import streamlit as st
import pickle
import re
import string
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title='Review Detective Machine!',
    page_icon='🕵️‍♂️',
    layout='wide'
)

# =========================
# LOAD MODEL & VECTORIZER
# =========================
@st.cache_resource
def load_assets():
    with open('naive_bayes_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

try:
    model, vectorizer = load_assets()
except Exception as e:
    st.error(f"Oh no! The robot's gears are stuck. Error: {e}")
    st.stop()

# =========================
# STOPWORDS CONFIG
# =========================
if hasattr(vectorizer, 'get_stop_words') and vectorizer.get_stop_words() is not None:
    stop_words = set(vectorizer.get_stop_words())
else:
    stop_words = set()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# =========================
# FUN & BRIGHT CSS
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Quicksand:wght=500;700&display=swap');
    
    /* Fun, bright backgrounds and comic-book fonts */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Quicksand', sans-serif;
        background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%) !important;
        color: #2C3E50 !important;
    }
    
    h1, h2, h3, .big-title {
        font-family: 'Fredoka One', cursive !important;
    }

    /* Pop-out Cartoon Cards */
    .kid-card {
        background: white;
        border: 4px solid #2C3E50;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 8px 8px 0px #2C3E50;
        margin-bottom: 25px;
    }

    /* Big Custom Textarea */
    textarea {
        background-color: #F8FAFC !important;
        border: 3px solid #2C3E50 !important;
        color: #2C3E50 !important;
        border-radius: 12px !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }

    /* Big, clickable Bouncy Button */
    div.stButton > button {
        background-color: #FF4757;
        color: white;
        border: 3px solid #2C3E50;
        padding: 15px;
        border-radius: 15px;
        font-size: 22px;
        font-family: 'Fredoka One', cursive;
        box-shadow: 0px 6px 0px #2C3E50;
        transition: all 0.1s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #FF6B81;
        color: white;
        transform: translateY(2px);
        box-shadow: 0px 4px 0px #2C3E50;
    }
    div.stButton > button:active {
        transform: translateY(6px);
        box-shadow: 0px 0px 0px #2C3E50;
    }

    /* Fun Result Boxes */
    .mood-box {
        border: 4px solid #2C3E50;
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 8px 8px 0px #2C3E50;
        margin-top: 20px;
    }
    .mood-happy {
        background-color: #2ED573;
        color: white;
    }
    .mood-sad {
        background-color: #FF4757;
        color: white;
    }
    .mood-title {
        font-family: 'Fredoka One', cursive;
        font-size: 32px;
        margin-bottom: 10px;
        text-shadow: 2px 2px #2C3E50;
    }
    .mood-text {
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR (THE INSTRUCTION MANUAL)
# =========================
with st.sidebar:
    st.markdown("## 🎒 Detective Kit")
    st.markdown("Welcome, Junior Detective! This app uses **Artificial Intelligence (AI)** to read words and guess how a person is feeling.")
    
    st.markdown("---")
    st.markdown("### 🤖 Robot Secret Stats")
    st.info("🧠 **Brain Type:** Naïve Bayes Game Engine\n\n🎯 **Language Skill:** CountVectorizer")
    
    st.markdown("---")
    st.caption("Made to make learning Machine Learning fun! ✨")

# =========================
# MAIN GAME HERO HEADER
# =========================
st.markdown('<h1 style="text-align: center; color: #2C3E50; font-size: 48px; margin-bottom:0;">🕵️‍♂️ THE MOOD READER MACHINE!</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #475569; font-size: 20px; font-weight: bold; margin-bottom: 40px;">Can our smart AI code crack the secret mood behind the review?</p>', unsafe_allow_html=True)

# LAYOUT SPACING
col_game, col_guide = st.columns([1.6, 1.4], gap="large")

with col_game:
    st.markdown('<div class="kid-card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; color:#2C3E50;'>✍️ Write a Restaurant Review below:</h3>", unsafe_allow_html=True)
    
    review_input = st.text_area(
        "Label hidden for style",
        label_visibility="collapsed",
        height=180,
        placeholder="Type here! Like: 'The ice cream was huge and delicious!' or 'The soup was cold and yucky...'"
    )
    
    trigger_game = st.button("🔍 ACTIVATE MOOD RADAR!")
    st.markdown('</div>', unsafe_allow_html=True)

with col_guide:
    st.markdown('<div class="kid-card" style="height: 100%;">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0; color:#2C3E50;'>✨ How the Robot Thinks:</h3>", unsafe_allow_html=True)
    
    st.markdown("1. **Lowercasing:** It changes ALL capital letters to small letters so it doesn't get confused! 🔡")
    st.markdown("2. **Scrubbing:** It throws away boring words like 'the', 'is', and 'and' to find the juicy describing words! 🧼")
    st.markdown("3. **Math Magic:** It counts up the happy and sad words to make its final guess! 📊")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# GAME ANIMATION & RESULTS
# =========================
if trigger_game:
    if not review_input.strip():
        st.toast("Oops! You forgot to type a review first! 📝", icon="🤓")
    else:
        # Fun cartoon loading steps
        status_placeholder = st.empty()
        
        with status_placeholder.container():
            st.markdown("### 🧹 Putting on detective glasses... (Cleaning Text)")
            cleaned_text = clean_text(review_input)
            time.sleep(0.5)
            
            st.markdown("### 🔢 Counting words on our fingers... (Vectorizing)")
            vectorized_matrix = vectorizer.transform([cleaned_text])
            time.sleep(0.5)
            
            st.markdown("### 🤖 Asking the Robot Brain for the answer... (Predicting)")
            prediction_output = model.predict(vectorized_matrix)[0]
            probability_matrix = model.predict_proba(vectorized_matrix)
            time.sleep(0.4)
            
        status_placeholder.empty() # Clear loading text

        # Show Results with explicit Positive/Negative designations
        if prediction_output == 1:
            score = probability_matrix[0][1] * 100
            
            st.markdown(
                f'''
                <div class="mood-box mood-happy">
                    <div class="mood-title">🎉 POSITIVE REVIEW: HAPPY MOOD! 🍕</div>
                    <div class="mood-text">The robot is {score:.0f}% sure this person loved their food! YAY!</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.balloons()
            
        else:
            score = probability_matrix[0][0] * 100
            
            st.markdown(
                f'''
                <div class="mood-box mood-sad">
                    <div class="mood-title">⚠️ NEGATIVE REVIEW: TUMMY ACHES! 💔</div>
                    <div class="mood-text">The robot is {score:.0f}% sure this customer left an unhappy review! Oh no!</div>
                </div>
                ''',
                unsafe_allow_html=True
            )
            st.snow()
