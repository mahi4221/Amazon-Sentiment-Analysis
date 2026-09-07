import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Amazon Sentiment AI",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_DIR = Path("models")

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
NB_PATH = MODEL_DIR / "naive_bayes.pkl"
LR_PATH = MODEL_DIR / "logistic_regression.pkl"
SVM_PATH = MODEL_DIR / "linear_svm.pkl"
RESULTS_PATH = MODEL_DIR / "model_results.csv"


# ============================================================
# VIOLET + WHITE DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #faf5ff 50%,
            #f3e8ff 100%
        );
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #3b0764 0%,
            #5b21b6 50%,
            #7c3aed 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    h1 {
        color: #3b0764 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #4c1d95 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #5b21b6 !important;
        font-weight: 700 !important;
    }

    .main-title {
        font-size: 42px;
        font-weight: 850;
        color: #3b0764;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 20px;
    }

    .external-box {
        background: white;
        border: 2px solid #c4b5fd;
        border-radius: 20px;
        padding: 28px;
        margin-top: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(91, 33, 182, 0.12);
    }

    .external-title {
        font-size: 30px;
        font-weight: 800;
        color: #4c1d95;
    }

    .external-description {
        font-size: 17px;
        color: #6b7280;
    }

    .positive-result {
        background: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .positive-result h1 {
        color: #15803d !important;
    }

    .negative-result {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        margin-top: 20px;
    }

    .negative-result h1 {
        color: #b91c1c !important;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #ddd6fe;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(91, 33, 182, 0.08);
    }

    textarea {
        border: 2px solid #c4b5fd !important;
        border-radius: 12px !important;
    }

    textarea:focus {
        border: 2px solid #7c3aed !important;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        padding: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # Remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep letters and apostrophes
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(NB_PATH, "rb") as f:
        naive_bayes = pickle.load(f)

    with open(LR_PATH, "rb") as f:
        logistic_regression = pickle.load(f)

    with open(SVM_PATH, "rb") as f:
        linear_svm = pickle.load(f)

    return (
        vectorizer,
        naive_bayes,
        logistic_regression,
        linear_svm
    )


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    VECTORIZER_PATH,
    NB_PATH,
    LR_PATH,
    SVM_PATH
]

missing_files = [
    file.name
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("⚠️ Required model files are missing.")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

(
    vectorizer,
    naive_bayes,
    logistic_regression,
    linear_svm
) = load_models()


# ============================================================
# LOAD RESULTS
# ============================================================

if RESULTS_PATH.exists():

    results_df = pd.read_csv(RESULTS_PATH)

else:

    results_df = pd.DataFrame()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="color:white !important;">
        🛒 Amazon AI
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="color:#ddd6fe;">
        Product Review Sentiment Analysis
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Sentiment Analyzer",
            "📊 Model Performance",
            "🧠 About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 🏆 Best Model")

    st.success("⚡ Linear SVM")

    st.markdown("### 🎯 Accuracy")

    st.info("95.75%")

    st.markdown("### ⭐ F1 Score")

    st.info("97.46%")


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">🛒 Amazon Product Review Sentiment Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered sentiment classification using NLP, TF-IDF and Machine Learning</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📦 Dataset",
            "568K+",
            "Reviews"
        )

    with col2:
        st.metric(
            "🎯 Accuracy",
            "95.75%",
            "Linear SVM"
        )

    with col3:
        st.metric(
            "⭐ F1 Score",
            "97.46%",
            "Linear SVM"
        )

    with col4:
        st.metric(
            "🤖 Models",
            "3",
            "Compared"
        )

    st.divider()

    # ========================================================
    # ⭐ EXTERNAL INPUT SECTION
    # ========================================================

    st.markdown(
        """
        <div class="external-box">

        <div class="external-title">
        🔮 Test Your Own Review
        </div>

        <div class="external-description">
        Enter any new customer review below and test the trained
        Machine Learning model.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        """
        💡 **External Input:**  
        This is a completely new review entered by the user.
        It does not have to be present in the training dataset.
        The trained **Linear SVM** model will predict whether
        the review is Positive or Negative.
        """
    )

    # --------------------------------------------------------
    # EXAMPLE BUTTONS
    # --------------------------------------------------------

    st.markdown("### 💡 Try an Example")

    example1, example2, example3 = st.columns(3)

    if "external_review" not in st.session_state:
        st.session_state.external_review = ""

    with example1:

        if st.button(
            "😊 Positive Example",
            use_container_width=True
        ):

            st.session_state.external_review = (
                "I absolutely love this product. "
                "The quality is excellent and it works perfectly."
            )

    with example2:

        if st.button(
            "😞 Negative Example",
            use_container_width=True
        ):

            st.session_state.external_review = (
                "This product is terrible. "
                "It stopped working after a few days. "
                "I completely regret buying it."
            )

    with example3:

        if st.button(
            "⭐ Excellent Example",
            use_container_width=True
        ):

            st.session_state.external_review = (
                "Amazing product! Excellent quality, "
                "fast delivery and definitely worth the money."
            )

    # --------------------------------------------------------
    # EXTERNAL TEXT INPUT
    # --------------------------------------------------------

    external_review = st.text_area(
        "📝 Enter External Customer Review",
        value=st.session_state.external_review,
        height=180,
        placeholder=(
            "Type any new Amazon review here...\n\n"
            "Example: I really love this product. "
            "The quality is amazing and I am very happy "
            "with my purchase."
        )
    )

    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict = st.button(
        "🔮 PREDICT SENTIMENT",
        type="primary",
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict:

        if not external_review.strip():

            st.warning(
                "⚠️ Please enter a customer review first."
            )

        else:

            # Step 1: Clean input
            cleaned_review = clean_text(
                external_review
            )

            # Step 2: TF-IDF
            review_vector = vectorizer.transform(
                [cleaned_review]
            )

            # Step 3: Linear SVM prediction
            prediction = linear_svm.predict(
                review_vector
            )[0]

            # Step 4: Decision score
            decision_score = linear_svm.decision_function(
                review_vector
            )[0]

            # Confidence-like score
            confidence = (
                1 /
                (
                    1 +
                    np.exp(-abs(decision_score))
                )
            ) * 100

            st.divider()

            # =================================================
            # POSITIVE
            # =================================================

            if prediction == 1:

                st.markdown(
                    """
                    <div class="positive-result">

                    <h1>😊 POSITIVE SENTIMENT</h1>

                    <p style="font-size:18px;">
                    The AI predicts that this customer review
                    expresses a positive opinion.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # =================================================
            # NEGATIVE
            # =================================================

            else:

                st.markdown(
                    """
                    <div class="negative-result">

                    <h1>😞 NEGATIVE SENTIMENT</h1>

                    <p style="font-size:18px;">
                    The AI predicts that this customer review
                    expresses a negative opinion.
                    </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("")

            # ------------------------------------------------
            # RESULT DETAILS
            # ------------------------------------------------

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "🎯 Prediction",
                    "Positive"
                    if prediction == 1
                    else "Negative"
                )

            with r2:

                st.metric(
                    "🤖 Model",
                    "Linear SVM"
                )

            with r3:

                st.metric(
                    "📊 Confidence",
                    f"{confidence:.2f}%"
                )

            st.divider()

            # ------------------------------------------------
            # PIPELINE
            # ------------------------------------------------

            st.markdown("### 🔄 Prediction Pipeline")

            p1, p2, p3, p4 = st.columns(4)

            with p1:

                st.info(
                    "📝\n\n**External Input**\n\nYour new review"
                )

            with p2:

                st.info(
                    "🧹\n\n**Cleaning**\n\nText preprocessing"
                )

            with p3:

                st.info(
                    "🔢\n\n**TF-IDF**\n\nFeature extraction"
                )

            with p4:

                st.info(
                    "⚡\n\n**Linear SVM**\n\nPrediction"
                )

            st.divider()

            # ------------------------------------------------
            # USER INPUT DISPLAY
            # ------------------------------------------------

            st.markdown("### 📝 Review Analyzed")

            with st.container(border=True):

                st.write(external_review)


# ============================================================
# SENTIMENT ANALYZER PAGE
# ============================================================

elif page == "🔍 Sentiment Analyzer":

    st.markdown(
        '<div class="main-title">🔍 Sentiment Analyzer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Analyze any new customer review.</div>',
        unsafe_allow_html=True
    )

    st.info(
        "You can also use the **Test Your Own Review** section "
        "directly from the Home page."
    )

    review = st.text_area(
        "Enter Customer Review",
        height=200,
        placeholder="Type your review here..."
    )

    if st.button(
        "🚀 Analyze",
        type="primary",
        use_container_width=True
    ):

        if not review.strip():

            st.warning("Please enter a review.")

        else:

            cleaned = clean_text(review)

            vector = vectorizer.transform(
                [cleaned]
            )

            prediction = linear_svm.predict(
                vector
            )[0]

            score = linear_svm.decision_function(
                vector
            )[0]

            confidence = (
                1 /
                (
                    1 +
                    np.exp(-abs(score))
                )
            ) * 100

            if prediction == 1:

                st.success(
                    "😊 POSITIVE SENTIMENT"
                )

            else:

                st.error(
                    "😞 NEGATIVE SENTIMENT"
                )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Prediction",
                    "Positive"
                    if prediction == 1
                    else "Negative"
                )

            with c2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="main-title">📊 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Comparison of trained Machine Learning models.</div>',
        unsafe_allow_html=True
    )

    st.divider()

    if results_df.empty:

        st.warning(
            "model_results.csv was not found."
        )

    else:

        st.markdown("### 📋 Model Comparison")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Find model column
        model_column = None

        for column in [
            "Model",
            "model",
            "Model Name",
            "model_name"
        ]:

            if column in results_df.columns:

                model_column = column
                break

        # Find accuracy column
        accuracy_column = None

        for column in [
            "Accuracy",
            "accuracy",
            "accuracy_score"
        ]:

            if column in results_df.columns:

                accuracy_column = column
                break

        if model_column and accuracy_column:

            st.markdown("### 📈 Accuracy Comparison")

            chart_data = results_df[
                [model_column, accuracy_column]
            ].copy()

            chart_data = chart_data.set_index(
                model_column
            )

            st.bar_chart(
                chart_data
            )

        st.divider()

        st.markdown("### 🏆 Best Model")

        c1, c2 = st.columns(2)

        with c1:

            st.success(
                """
                ## ⚡ Linear SVM

                Best performing model
                """
            )

        with c2:

            st.metric(
                "Accuracy",
                "95.75%"
            )

            st.metric(
                "F1 Score",
                "97.46%"
            )

        st.divider()

        st.markdown("### 🔲 Confusion Matrices")

        cm1, cm2, cm3 = st.columns(3)

        matrices = [
            (
                cm1,
                "Naive Bayes",
                MODEL_DIR / "confusion_matrix_naive_bayes.png"
            ),
            (
                cm2,
                "Logistic Regression",
                MODEL_DIR / "confusion_matrix_logistic_regression.png"
            ),
            (
                cm3,
                "Linear SVM",
                MODEL_DIR / "confusion_matrix_linear_svm.png"
            )
        ]

        for column, title, image_path in matrices:

            with column:

                st.markdown(
                    f"### {title}"
                )

                if image_path.exists():

                    st.image(
                        str(image_path),
                        width="stretch"
                    )

                else:

                    st.warning(
                        "Image not found."
                    )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "🧠 About Project":

    st.markdown(
        '<div class="main-title">🧠 About Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Amazon Product Review Sentiment Analysis using Machine Learning</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### 🎯 Objective")

    st.write(
        """
        The objective of this project is to classify Amazon customer
        reviews into Positive and Negative sentiment using Natural
        Language Processing and Machine Learning.
        """
    )

    st.markdown("### 📦 Dataset")

    st.write(
        """
        The dataset contains more than 568,000 Amazon product reviews.
        """
    )

    st.markdown("### ⭐ Sentiment Mapping")

    sentiment_table = pd.DataFrame(
        {
            "Score": [
                "1 Star",
                "2 Stars",
                "3 Stars",
                "4 Stars",
                "5 Stars"
            ],
            "Sentiment": [
                "Negative",
                "Negative",
                "Removed",
                "Positive",
                "Positive"
            ],
            "Label": [
                "0",
                "0",
                "-",
                "1",
                "1"
            ]
        }
    )

    st.dataframe(
        sentiment_table,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "3-star reviews are removed because they are considered neutral/mixed."
    )

    st.markdown("### 🔄 NLP Pipeline")

    pipeline = [
        "Load Dataset",
        "Remove Missing Values",
        "Remove 3-Star Reviews",
        "Create Binary Labels",
        "Combine Summary + Text",
        "Remove Duplicates",
        "Clean Text",
        "Train-Test Split",
        "TF-IDF",
        "Train Models",
        "Evaluate Models",
        "Save Models"
    ]

    for i, step in enumerate(
        pipeline,
        start=1
    ):

        st.write(
            f"**{i}.** {step}"
        )

    st.markdown("### 🛠️ Technologies")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 🐍 Python")
        st.write("Programming")

    with c2:
        st.markdown("#### 📊 Pandas")
        st.write("Data processing")

    with c3:
        st.markdown("#### 🤖 Scikit-learn")
        st.write("Machine Learning")

    c4, c5, c6 = st.columns(3)

    with c4:
        st.markdown("#### 🌐 Streamlit")
        st.write("Web application")

    with c5:
        st.markdown("#### 🔢 NumPy")
        st.write("Numerical computation")

    with c6:
        st.markdown("#### 📈 Matplotlib")
        st.write("Visualization")

    st.divider()

    st.success(
        """
        🏆 **Best Model: Linear SVM**

        Accuracy: **95.75%**

        F1 Score: **97.46%**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">

    🛒 <b>Amazon Product Review Sentiment Analysis</b>

    <br><br>

    Python • NLP • TF-IDF • Machine Learning • Streamlit

    </div>
    """,
    unsafe_allow_html=True
)