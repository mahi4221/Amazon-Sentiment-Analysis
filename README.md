# 🛍️ Amazon Product Review Sentiment Analysis

An end-to-end **Natural Language Processing (NLP) and Machine Learning** project that analyzes Amazon product reviews and classifies them as **Positive** or **Negative** sentiment.

The project includes data preprocessing, sentiment labeling, TF-IDF feature extraction, multiple machine learning models, model comparison, evaluation, and a Streamlit web application for testing new reviews.

---

## 📌 Project Overview

Customer reviews contain valuable information about user satisfaction and product quality. Manually analyzing hundreds of thousands of reviews is difficult, so this project uses NLP and Machine Learning to automatically determine the sentiment expressed in a review.

The system follows this workflow:

**Amazon Reviews → Data Cleaning → Sentiment Labeling → Text Preprocessing → TF-IDF → Machine Learning → Sentiment Prediction**

---

## 🎯 Objective

The main objective is to build a machine learning system that can:

- Process large-scale Amazon product reviews
- Convert star ratings into binary sentiment labels
- Clean and normalize review text
- Extract useful textual features using TF-IDF
- Train and compare multiple classification algorithms
- Evaluate models using standard classification metrics
- Predict the sentiment of new, unseen reviews through a Streamlit interface

---

## 📊 Dataset

The project uses the **Amazon Fine Food Reviews** dataset, containing Amazon customer reviews with information such as:

- Review score
- Review summary
- Review text
- Product information
- User information
- Helpfulness information

For sentiment classification, the project uses the following fields:

- `Score`
- `Summary`
- `Text`

### Sentiment Mapping

| Star Rating | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Removed / Neutral |
| 4–5 | Positive |

Three-star reviews are excluded so that the task becomes a clear binary classification problem.

---

## 🧠 Machine Learning Models

Three supervised learning algorithms are trained and compared:

### 1. Naive Bayes

`MultinomialNB` is used because it works effectively with sparse text features such as TF-IDF representations.

### 2. Logistic Regression

Logistic Regression is used as a strong linear baseline for binary sentiment classification.

### 3. Linear SVM

`LinearSVC` is used because Support Vector Machines are highly effective for high-dimensional text classification problems.

The best-performing model is automatically identified and saved for use by the Streamlit application.

---

## 🔤 NLP Pipeline

### 1. Data Loading

The Amazon review dataset is loaded using Pandas.

### 2. Column Selection

Only the `Score`, `Summary`, and `Text` columns are required for sentiment analysis.

### 3. Missing Value Handling

Reviews without required text or score information are removed, while missing summaries are replaced with empty text.

### 4. Sentiment Creation

Ratings are converted into binary labels:

```text
1, 2 → Negative
4, 5 → Positive
3    → Removed
```

### 5. Combining Review Text

The review summary and review body are combined into a single text field.

### 6. Duplicate Removal

Duplicate reviews are removed before model training.

### 7. Text Cleaning

The preprocessing stage includes:

- Lowercasing
- HTML tag removal
- URL removal
- Removal of unnecessary characters
- Whitespace normalization

Important sentiment-bearing words are retained.

### 8. Train-Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

A stratified split is used to preserve the class distribution.

### 9. TF-IDF Feature Extraction

TF-IDF converts review text into numerical features.

The project uses:

- Unigrams and bigrams
- Up to 150,000 features
- Minimum document frequency of 2
- Maximum document frequency of 95%
- Sublinear TF scaling
- Unicode accent stripping

### 10. Model Training

Naive Bayes, Logistic Regression, and Linear SVM are trained using the TF-IDF features.

### 11. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

---

## 🏆 Model Selection

After training, the models are compared using their evaluation metrics. The model with the best **F1 Score** is selected automatically and its name is stored in:

```text
models/best_model_name.pkl
```

The complete comparison is stored in:

```text
models/model_results.csv
```

---

## 📁 Project Structure

```text
Amazon-Sentiment-Analysis/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
└── models/
    ├── tfidf_vectorizer.pkl
    ├── naive_bayes.pkl
    ├── logistic_regression.pkl
    ├── linear_svm.pkl
    ├── best_model_name.pkl
    ├── model_results.csv
    ├── cleaned_reviews.csv
    ├── test_sample.csv
    └── confusion matrix images
```

> The large `Reviews.csv` dataset is used locally for training and is not required to be committed to GitHub.

---

## 🛠️ Technologies Used

- **Python**
- **Pandas** – Data processing
- **NumPy** – Numerical operations
- **Scikit-learn** – Machine Learning and NLP
- **TF-IDF** – Text feature extraction
- **Matplotlib** – Confusion matrix visualization
- **Joblib / Pickle** – Model serialization
- **Streamlit** – Interactive web application

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/mahi4221/Amazon-Sentiment-Analysis.git
```

### 2. Open the project folder

```bash
cd Amazon-Sentiment-Analysis
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment on Windows

```powershell
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Models

Place the Amazon review dataset at the path configured in `train_model.py` and run:

```bash
python train_model.py
```

The training script will:

1. Load the dataset
2. Clean the data
3. Create sentiment labels
4. Combine review summary and text
5. Remove duplicate reviews
6. Clean the review text
7. Split the data
8. Generate TF-IDF features
9. Train three machine learning models
10. Evaluate the models
11. Save the trained models and evaluation results
12. Select the best model

---

## 🌐 Running the Streamlit Application

After the models have been generated, run:

```bash
python -m streamlit run app.py
```

The application provides an interactive interface where users can enter their own Amazon product review and receive a predicted sentiment.

### Example Input

```text
This product is excellent. The quality is amazing and I am very happy with my purchase.
```

### Example Output

```text
Positive
```

Another example:

```text
The product stopped working after a few days. Very poor quality and I am disappointed.
```

Possible output:

```text
Negative
```

---

## 📈 Evaluation Metrics

The project evaluates each model using four main numerical metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Measures correctness of positive predictions |
| Recall | Measures how many actual positive cases are detected |
| F1 Score | Harmonic mean of precision and recall |

Confusion matrices are also generated for each trained model.

---

## 💡 Key Features

- ✅ Large-scale Amazon review processing
- ✅ Binary sentiment classification
- ✅ 1–2 star reviews classified as Negative
- ✅ 4–5 star reviews classified as Positive
- ✅ 3-star reviews removed
- ✅ Text cleaning and normalization
- ✅ Duplicate review removal
- ✅ TF-IDF with unigram and bigram features
- ✅ Naive Bayes classification
- ✅ Logistic Regression classification
- ✅ Linear SVM classification
- ✅ Automatic best-model selection
- ✅ Confusion matrix generation
- ✅ Saved ML models for reuse
- ✅ Interactive Streamlit application
- ✅ External review prediction

---

## 🔮 Future Improvements

Possible improvements include:

- Add deep learning models such as LSTM or BERT
- Add sentiment probability calibration
- Add review rating prediction as a multi-class problem
- Add aspect-based sentiment analysis
- Add interactive model-performance charts
- Deploy the Streamlit application online
- Add multilingual review support
- Add explainable AI features for individual predictions

---

## ⚠️ Disclaimer

This project is developed for **educational and demonstration purposes**. The predictions are generated by machine learning models trained on historical review data and should not be treated as a definitive representation of customer opinion.

---

## 👩‍💻 Author

**Maruthi Bonela**

B.Tech – Computer Science and Engineering (AI/ML)

GitHub: https://github.com/mahi4221

---

## ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub!

---

**Amazon Product Review Sentiment Analysis | NLP + TF-IDF + Machine Learning + Streamlit**
