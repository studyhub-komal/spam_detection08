# 📩 Spam Message Detection Using CountVectorizer

A machine learning project in Python that classifies text messages as **spam** or **not spam**. Built with Pandas and Scikit-learn, it covers text preprocessing, bag-of-words feature extraction, model training, evaluation, and a Streamlit web app with a detector and a dashboard.

---

## ✨ Features

- Classifies any message as spam or not spam with a confidence score
- Bag-of-words feature extraction using `CountVectorizer`
- `MultinomialNB` (Naive Bayes) classifier
- Streamlit app with two tabs:
  - **Detector**: paste a message and get an instant prediction
  - **Dashboard**: dataset overview, accuracy / precision / recall / F1, confusion matrix, message length comparison, and top words for spam vs not spam

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📁 Project Structure

```
spam-detector/
├── app.py                 # Streamlit app (Detector + Dashboard)
├── spam_model.pkl         # Trained MultinomialNB model
├── transform.pkl          # Fitted CountVectorizer
├── spam_data_csv.csv      # Dataset (5000 messages)
├── requirements.txt
└── README.md
```

## 📊 Dataset

- 5000 messages with two columns: `text` and `label`
- Balanced: 2500 spam and 2500 not spam
- Label mapping used in this project:

| Label | Meaning  |
|-------|----------|
| 0     | Spam     |
| 1     | Not spam |

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   ```

2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac / Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. Open `http://localhost:8501` in your browser.

## 🧠 How It Works

1. The text is converted into word counts using `CountVectorizer`.
2. The counts are passed to a trained `MultinomialNB` model.
3. The model predicts the class and the probability of each class.
4. The app shows the result along with a spam probability bar.

> Scaling is not used because `MultinomialNB` works directly on word counts and does not accept negative values.

## ⚠️ Limitations

- The dataset has repeated sentence patterns, so accuracy on this data is very high but may be lower on real-world SMS or WhatsApp spam.
- The model only understands words seen during training (vocabulary of about 1,346 words).
- Adding a larger and more varied dataset would improve real-world performance.

## 🔮 Future Improvements

- Try `TfidfVectorizer`, Logistic Regression, or SVM
- Train on a larger real-world dataset
- Deploy the app on Streamlit Community Cloud

## 👤 Author

**YOUR NAME**

GitHub: [github.com/YOUR_USERNAME](https://github.com/studyhub-komal)


---

⭐ If you found this project useful, consider giving it a star!
