import streamlit as st
import pickle
import re
import string

# =========================
# LOAD MODEL & VECTORIZER
# =========================

with open('naive_bayes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# =========================
# STOPWORDS
# =========================

# Getting stop words directly from your trained vectorizer is safer 
# and prevents deprecation issues with sklearn's ENGLISH_STOP_WORDS.
if hasattr(vectorizer, 'get_stop_words') and vectorizer.get_stop_words() is not None:
    stop_words = set(vectorizer.get_stop_words())
else:
    # Fallback to an empty set if the vectorizer didn't use stop words
    stop_words = set()

# =========================
# TEXT CLEANING FUNCTION
# =========================

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return ' '.join(words)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title='AI Sentiment Analyzer',
    page_icon='🧠',
    layout='wide'
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
        margin-bottom: 20px;
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

# LEFT SIDE
with left_col:
    # Open the custom card container
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
    
    # Securely close the card container layout
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("✨ Features")

    st.write("✔ Real-time Prediction")
    st.write("✔ Confidence Score")
    st.write("✔ Text Cleaning")
    st.write("✔ Stopword Removal")

    st.markdown('---')

    st.subheader("🤖 Model")

    st.write("Model: BernoulliNB")
    st.write("Vectorizer: CountVectorizer")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION
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

        # Positive
        if prediction == 1:

            confidence = probability[0][1] * 100

            st.markdown(
                f'''
                <div class="positive">
                    ✅ POSITIVE REVIEW
                    <br><br>
                    Confidence Score: {confidence:.2f}%
                </div>
                ''',
                unsafe_allow_html=True
            )

            st.balloons()

        # Negative
        else:

            confidence = probability[0][0] * 100

            st.markdown(
                f'''
                <div class="negative">
                    ❌ NEGATIVE REVIEW
                    <br><br>
                    Confidence Score: {confidence:.2f}%
                </div>
                ''',
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
            Built with ❤️ using Streamlit & Machine Learning
        </p>
    </center>
    """,
    unsafe_allow_html=True
)