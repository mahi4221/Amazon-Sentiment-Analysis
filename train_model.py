# ============================================================
# AMAZON PRODUCT REVIEW SENTIMENT ANALYSIS
# MODEL TRAINING PIPELINE
# ============================================================

import os
import re
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Users\Maruthi B\Downloads\Reviews.csv"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("AMAZON PRODUCT REVIEW SENTIMENT ANALYSIS")
print("=" * 70)

print("\n[1/10] Loading dataset...")

df = pd.read_csv(r"C:\Users\Maruthi B\Downloads\Amazon-Sentiment-Analysis\Reviews.csv")

print(f"\nDataset loaded successfully!")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. SELECT REQUIRED COLUMNS
# ============================================================

print("\n[2/10] Selecting required columns...")

df = df[["Score", "Summary", "Text"]].copy()

print("Selected columns:")
print(["Score", "Summary", "Text"])


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

print("\n[3/10] Removing missing values...")

before_missing = len(df)

df.dropna(subset=["Score", "Text"], inplace=True)

# Summary can be missing, so replace it with empty text
df["Summary"] = df["Summary"].fillna("")

after_missing = len(df)

print(f"Removed rows : {before_missing - after_missing:,}")
print(f"Remaining    : {after_missing:,}")


# ============================================================
# 5. REMOVE 3-STAR REVIEWS
# ============================================================

print("\n[4/10] Creating sentiment labels...")

print("\nOriginal Score Distribution:")
print(df["Score"].value_counts().sort_index())

# Keep only 1, 2, 4 and 5 star reviews
df = df[df["Score"].isin([1, 2, 4, 5])].copy()

# Create binary sentiment
df["sentiment"] = df["Score"].apply(
    lambda x: 0 if x in [1, 2] else 1
)

# Human-readable label
df["sentiment_label"] = df["sentiment"].map({
    0: "Negative",
    1: "Positive"
})

print("\nAfter removing 3-star reviews:")
print(df["Score"].value_counts().sort_index())

print("\nSentiment Distribution:")
print(df["sentiment_label"].value_counts())


# ============================================================
# 6. COMBINE SUMMARY + TEXT
# ============================================================

print("\n[5/10] Combining review summary and review text...")

df["review"] = (
    df["Summary"].astype(str)
    + " "
    + df["Text"].astype(str)
)

# Keep only required columns
df = df[["review", "sentiment", "sentiment_label"]]


# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

print("\n[6/10] Removing duplicate reviews...")

before_duplicates = len(df)

df.drop_duplicates(subset=["review"], inplace=True)

after_duplicates = len(df)

print(f"Duplicate reviews removed : "
      f"{before_duplicates - after_duplicates:,}")

print(f"Remaining reviews         : "
      f"{after_duplicates:,}")


# ============================================================
# 8. TEXT CLEANING
# ============================================================

print("\n[7/10] Cleaning review text...")
print("This may take some time because the dataset is large.")


def clean_text(text):
    """
    Clean Amazon review text for NLP.
    """

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Keep letters and apostrophes
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["review"] = df["review"].apply(clean_text)

# Remove empty reviews
df = df[df["review"].str.len() > 0].copy()

print(f"Reviews after cleaning: {len(df):,}")


# ============================================================
# 9. CHECK FINAL CLASS BALANCE
# ============================================================

print("\nFinal Class Distribution:")
print(df["sentiment_label"].value_counts())

print("\nFinal Class Percentages:")
print(
    (df["sentiment_label"].value_counts(normalize=True) * 100)
    .round(2)
)


# Save cleaned dataset
cleaned_path = os.path.join(MODEL_DIR, "cleaned_reviews.csv")

df.to_csv(cleaned_path, index=False)

print(f"\nCleaned dataset saved to:")
print(cleaned_path)


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

print("\n[8/10] Splitting dataset...")

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing samples  : {len(X_test):,}")


# ============================================================
# 11. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\n[9/10] Creating TF-IDF features...")
print("This may take several minutes.")

tfidf = TfidfVectorizer(
    max_features=150000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    strip_accents="unicode"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF completed!")

print(f"Training TF-IDF shape : {X_train_tfidf.shape}")
print(f"Testing TF-IDF shape  : {X_test_tfidf.shape}")

print(f"Vocabulary size       : {len(tfidf.vocabulary_):,}")


# ============================================================
# 12. SAVE TF-IDF VECTORIZER
# ============================================================

with open(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
    "wb"
) as file:

    pickle.dump(tfidf, file)

print("\nTF-IDF vectorizer saved.")


# ============================================================
# 13. DEFINE MODELS
# ============================================================

models = {

    "Naive Bayes": MultinomialNB(
        alpha=0.5
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=2.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    "Linear SVM": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 14. TRAIN MODELS
# ============================================================

print("\n[10/10] Training machine learning models...")

results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {model_name}")
    print("-" * 70)

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[model_name] = model

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Negative", "Positive"],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"]
    )

    plt.figure(figsize=(6, 5))
    disp.plot(values_format="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")

    cm_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}_confusion_matrix.png"
    )

    plt.savefig(cm_path, dpi=150)
    plt.close()

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"{safe_name}.pkl"
    )

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved: {model_path}")


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

print("\n" + "=" * 70)
print(f"BEST MODEL: {best_model_name}")
print("=" * 70)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "model_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

with open(
    os.path.join(MODEL_DIR, "best_model_name.pkl"),
    "wb"
) as file:

    pickle.dump(best_model_name, file)


# ============================================================
# 18. SAVE TEST SAMPLE FOR UI
# ============================================================

# Save a small sample instead of the complete test dataset
sample_size = min(5000, len(X_test))

sample_df = pd.DataFrame({
    "review": X_test.iloc[:sample_size].values,
    "actual_sentiment": y_test.iloc[:sample_size].values
})

sample_df.to_csv(
    os.path.join(MODEL_DIR, "test_sample.csv"),
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nGenerated files:")

for file_name in os.listdir(MODEL_DIR):
    print("  ✓", file_name)

print("\nYou can now create the Streamlit frontend.")
print("=" * 70)