# 🧠 AI Sentiment Analysis Web App

A professional Machine Learning web application built using **Python**, **NLP**, **Naïve Bayes**, and **Streamlit** to predict whether a restaurant review is **Positive** or **Negative**.

---

# 🚀 Live Features

✅ Restaurant Review Sentiment Prediction  
✅ Natural Language Processing (NLP)  
✅ Text Cleaning & Preprocessing  
✅ Bernoulli Naïve Bayes Model  
✅ CountVectorizer Feature Extraction  
✅ Real-time Predictions  
✅ Confidence Score Display  
✅ Professional Streamlit UI  
✅ Deployable on Streamlit Cloud  

---

# 📌 Project Overview

This project performs **Sentiment Analysis** on restaurant reviews using Machine Learning.

The application takes a user review as input and predicts whether the review sentiment is:

- ✅ Positive
- ❌ Negative

---

# 📂 Project Structure

```bash
sentiment-analysis-streamlit/
│
├── app.py
├── output.csv
├── notebook.ipynb
├── naive_bayes_model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
```

---

# 🛠 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Handling |
| NumPy | Numerical Operations |
| NLTK | NLP Preprocessing |
| Scikit-learn | Machine Learning |
| BernoulliNB | Classification Model |
| CountVectorizer | Text Vectorization |
| Streamlit | Web Application |
| Matplotlib | Visualization |
| Seaborn | Visualization |

---

# 📊 Dataset Information

Dataset contains:

| Column | Description |
|---|---|
| Review | Customer Review Text |
| Liked | Sentiment Label |

### Sentiment Labels

- `1` → Positive Review
- `0` → Negative Review

---

# 🔍 Machine Learning Workflow

## 1️⃣ Data Loading

- Load dataset using Pandas
- Explore dataset structure

## 2️⃣ Data Preprocessing

The following preprocessing steps are performed:

- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Remove extra spaces
- Remove stopwords
- Lemmatization

## 3️⃣ Feature Extraction

Text is converted into numerical format using:

- CountVectorizer
- TF-IDF (optional)

## 4️⃣ Model Training

Model used:

```python
BernoulliNB()
```

## 5️⃣ Model Evaluation

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 🧪 Example Inputs

## Positive Review

```text
The food was amazing and the staff were very friendly.
```

## Negative Review

```text
Worst restaurant experience ever.
```

---

# 📈 Example Output

```text
✅ POSITIVE REVIEW
Confidence Score: 98.45%
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/sentiment-analysis-streamlit.git
```

---

## Open Project Folder

```bash
cd sentiment-analysis-streamlit
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# 🌐 Streamlit Cloud Deployment

Deploy easily using:

:contentReference[oaicite:0]{index=0}

### Steps

1. Push project to GitHub
2. Login to Streamlit Cloud
3. Click **New App**
4. Select repository
5. Select `app.py`
6. Deploy

---

# 📦 Requirements

```txt
streamlit
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
```

---

# 🧠 Model Used

## Bernoulli Naïve Bayes

BernoulliNB works well for:

- Binary features
- Text classification
- Short review sentiment analysis

---

# 📸 Application UI

Features of the professional UI:

- Dark Theme
- Responsive Layout
- Sidebar Information Panel
- Confidence Score
- Styled Prediction Cards
- Interactive Design

---

# 📚 Learning Outcomes

This project demonstrates:

- NLP Preprocessing
- Sentiment Analysis
- Feature Engineering
- Text Vectorization
- Machine Learning Classification
- Streamlit Deployment
- Professional UI Design

---

# 👨‍💻 Author

Developed using ❤️ with Python, NLP, and Machine Learning.

---

# ⭐ Support

If you like this project:

⭐ Star the repository  
🍴 Fork the repository  
📢 Share with others  

---

# 📜 License

This project is open-source and available under the MIT License.
