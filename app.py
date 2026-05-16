import streamlit as st
import pickle
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# DOWNLOAD NLTK DATA
# =========================

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# =========================
# LOAD MODEL & VECTORIZER
# =========================

with open('naive_bayes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# =========================
# NLP TOOLS
# =========================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# =========================
# TEXT CLEANING FUNCTION
# =========================

def clean_text(text):

    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title='AI Sentiment Analyzer',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded'
)

# =========================
# CUSTOM CSS
# =========================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .title {
        font-size: 52px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #B0B3B8;
        font-size: 20px;
        margin-bottom: 40px;
    }

    .card {
        background-color: #1E1E1E;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    }

    .positive {
        background-color: #0F5132;
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }

    .negative {
        background-color: #842029;
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }

    textarea {
        font-size: 18px !important;
    }

    div.stButton > button {
        background-color: #4F46E5;
        color: white;
        height: 3em;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #4338CA;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title('📊 Project Information')

    st.write("""
    ### Technologies Used

    ✅ NLP

    ✅ Bernoulli Naïve Bayes

    ✅ CountVectorizer

    ✅ Streamlit

    ✅ Scikit-learn
    """)

    st.markdown('---')

    st.write("Built using Python & Machine Learning")

# =========================
# MAIN TITLE
# =========================

st.markdown(
    '<div class="title">🧠 AI Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Restaurant Review Sentiment Prediction using Machine Learning</div>',
    unsafe_allow_html=True
)

# =========================
# LAYOUT
# =========================

left_col, right_col = st.columns([2, 1])

# =========================
# LEFT COLUMN
# =========================

with left_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    review = st.text_area(
        "✍️ Enter Restaurant Review",
        height=220,
        placeholder="Example: The food was delicious and the service was amazing..."
    )

    predict_button = st.button(
        "🚀 Predict Sentiment",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# RIGHT COLUMN
# =========================

with right_col:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("✨ Features")

    st.write("✔ Real-time Prediction")
    st.write("✔ Confidence Score")
    st.write("✔ Text Cleaning")
    st.write("✔ Stopword Removal")
    st.write("✔ Lemmatization")

    st.markdown('---')

    st.subheader("🤖 Model")

    st.write("Model: BernoulliNB")
    st.write("Vectorizer: CountVectorizer")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION SECTION
# =========================

if predict_button:

    if review.strip() == "":

        st.warning("⚠ Please enter a review.")

    else:

        cleaned_review = clean_text(review)

        vectorized_review = vectorizer.transform(
            [cleaned_review]
        )

        prediction = model.predict(
            vectorized_review
        )[0]

        probability = model.predict_proba(
            vectorized_review
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # POSITIVE
        if prediction == 1:

            confidence = probability[0][1] * 100

            st.markdown(
                f"""
                <div class="positive">
                    ✅ POSITIVE REVIEW
                    <br><br>
                    Confidence Score: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.balloons()

        # NEGATIVE
        else:

            confidence = probability[0][0] * 100

            st.markdown(
                f"""
                <div class="negative">
                    ❌ NEGATIVE REVIEW
                    <br><br>
                    Confidence Score: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

# =========================
# FOOTER
# =========================

st.markdown('---')

st.markdown(
    """
    <center>
        <p style='color:gray;'>
            Built with ❤️ using Streamlit, NLP, and Machine Learning
        </p>
    </center>
    """,
    unsafe_allow_html=True
)