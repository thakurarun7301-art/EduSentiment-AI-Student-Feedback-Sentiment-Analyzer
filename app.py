import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from preprocess import clean_text


MODEL_DIR = "models"


st.set_page_config(
    page_title="EduSentiment AI",
    page_icon="🧠",
    layout="wide"
)


st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 30px;
    }

    .info-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #ddd;
        text-align: center;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #ddd;
        margin-top: 20px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 700;
    }

    .confidence {
        font-size: 20px;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    '<div class="main-title">🧠 EduSentiment AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Intelligent Student Feedback Sentiment Analyzer</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:
    st.markdown(
        '<div class="info-card">📝<br><b>NLP</b><br>Text Processing</div>',
        unsafe_allow_html=True
    )


with col2:
    st.markdown(
        '<div class="info-card">🧠<br><b>LSTM</b><br>Deep Learning</div>',
        unsafe_allow_html=True
    )


with col3:
    st.markdown(
        '<div class="info-card">⚡<br><b>GRU</b><br>Deep Learning</div>',
        unsafe_allow_html=True
    )


st.write("")


lstm_path = os.path.join(
    MODEL_DIR,
    "lstm_sentiment.keras"
)

gru_path = os.path.join(
    MODEL_DIR,
    "gru_sentiment.keras"
)

tokenizer_path = os.path.join(
    MODEL_DIR,
    "tokenizer.json"
)

config_path = os.path.join(
    MODEL_DIR,
    "config.json"
)


required_files = [
    lstm_path,
    gru_path,
    tokenizer_path,
    config_path
]


if not all(
    os.path.exists(path)
    for path in required_files
):

    st.warning(
        "Models are not available yet."
    )

    st.info(
        "Please run: python train.py"
    )

    st.stop()


@st.cache_resource
def load_project():

    lstm = tf.keras.models.load_model(
        lstm_path
    )

    gru = tf.keras.models.load_model(
        gru_path
    )

    with open(
        tokenizer_path,
        "r",
        encoding="utf-8"
    ) as f:

        tokenizer = tokenizer_from_json(
            f.read()
        )

    with open(
        config_path,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    return lstm, gru, tokenizer, config


lstm, gru, tokenizer, config = load_project()


st.markdown(
    "## 🔍 Analyze Student Feedback"
)


model_name = st.selectbox(
    "Choose Deep Learning Model",
    ["LSTM", "GRU"]
)


st.markdown(
    "### 💬 Single Feedback Analysis"
)


st.write(
    "Try a sample feedback:"
)


sample_col1, sample_col2, sample_col3 = st.columns(3)


if "sample_feedback" not in st.session_state:

    st.session_state.sample_feedback = ""


with sample_col1:

    if st.button(
        "😊 Positive",
        use_container_width=True
    ):

        st.session_state.sample_feedback = (
            "The teacher explains concepts very clearly "
            "and the classes are very interesting."
        )


with sample_col2:

    if st.button(
        "😐 Neutral",
        use_container_width=True
    ):

        st.session_state.sample_feedback = (
            "The course content is okay and "
            "the classes are conducted regularly."
        )


with sample_col3:

    if st.button(
        "😞 Negative",
        use_container_width=True
    ):

        st.session_state.sample_feedback = (
            "The teacher does not explain concepts clearly "
            "and the classes are boring."
        )


feedback = st.text_area(
    "Write your feedback below:",
    value=st.session_state.sample_feedback,
    height=150,
    placeholder="Example: The teacher explains concepts very clearly."
)


if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not feedback.strip():

        st.error(
            "Please enter some feedback."
        )

    else:

        cleaned = clean_text(
            feedback
        )


        sequence = tokenizer.texts_to_sequences(
            [cleaned]
        )


        padded = pad_sequences(
            sequence,
            maxlen=config["max_len"],
            padding="post",
            truncating="post"
        )


        if model_name == "LSTM":

            model = lstm

        else:

            model = gru


        probabilities = model.predict(
            padded,
            verbose=0
        )[0]


        class_names = config["classes"]


        index = int(
            np.argmax(probabilities)
        )


        sentiment = class_names[index]


        confidence = (
            float(probabilities[index]) * 100
        )


        st.divider()


        st.markdown(
            "## 📊 Analysis Result"
        )


        if sentiment == "positive":

            emoji = "😊"
            title = "Positive Sentiment"

        elif sentiment == "negative":

            emoji = "😞"
            title = "Negative Sentiment"

        else:

            emoji = "😐"
            title = "Neutral Sentiment"


        result_html = f"""
<div class="result-card">
<div class="result-title">{emoji} {title}</div>
<div class="confidence">Confidence: <b>{confidence:.2f}%</b></div>
<div>Model Used: <b>{model_name}</b></div>
</div>
"""


        st.markdown(
            result_html,
            unsafe_allow_html=True
        )


        st.markdown(
            "### 📈 Sentiment Probabilities"
        )


        negative = (
            float(probabilities[0]) * 100
        )

        neutral = (
            float(probabilities[1]) * 100
        )

        positive = (
            float(probabilities[2]) * 100
        )


        chart_data = pd.DataFrame(
            {
                "Sentiment": [
                    "Negative",
                    "Neutral",
                    "Positive"
                ],
                "Probability": [
                    negative,
                    neutral,
                    positive
                ]
            }
        )


        st.bar_chart(
            chart_data,
            x="Sentiment",
            y="Probability"
        )


        st.write(
            f"😞 Negative: {negative:.2f}%"
        )

        st.progress(
            float(probabilities[0])
        )


        st.write(
            f"😐 Neutral: {neutral:.2f}%"
        )

        st.progress(
            float(probabilities[1])
        )


        st.write(
            f"😊 Positive: {positive:.2f}%"
        )

        st.progress(
            float(probabilities[2])
        )


        st.markdown(
            "### 🔬 NLP Processing"
        )


        st.write(
            "**Original Feedback:**"
        )

        st.info(
            feedback
        )


        st.write(
            "**Cleaned Text:**"
        )

        st.code(
            cleaned
        )


        st.write(
            "**Selected Deep Learning Model:**"
        )

        st.success(
            model_name
        )


st.divider()


st.markdown(
    "## 📁 Bulk Feedback Analysis"
)


st.write(
    "Upload a CSV file containing multiple student feedback records."
)


uploaded_file = st.file_uploader(
    "Upload Feedback CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )


        if "text" not in df.columns:

            st.error(
                "CSV file must contain a column named 'text'."
            )


        else:

            st.success(
                f"{len(df)} feedback records loaded successfully."
            )


            st.dataframe(
                df.head(),
                use_container_width=True
            )


            if st.button(
                "🚀 Analyze All Feedback",
                use_container_width=True
            ):


                if len(df) == 0:

                    st.warning(
                        "The CSV file does not contain any feedback."
                    )


                else:

                    results = []


                    progress = st.progress(
                        0
                    )


                    if model_name == "LSTM":

                        model = lstm

                    else:

                        model = gru


                    for i, text in enumerate(
                        df["text"].fillna("")
                    ):


                        cleaned = clean_text(
                            text
                        )


                        sequence = tokenizer.texts_to_sequences(
                            [cleaned]
                        )


                        padded = pad_sequences(
                            sequence,
                            maxlen=config["max_len"],
                            padding="post",
                            truncating="post"
                        )


                        probabilities = model.predict(
                            padded,
                            verbose=0
                        )[0]


                        index = int(
                            np.argmax(probabilities)
                        )


                        sentiment = config["classes"][index]


                        confidence = (
                            float(
                                probabilities[index]
                            ) * 100
                        )


                        results.append(
                            {
                                "Feedback": text,
                                "Sentiment": sentiment,
                                "Confidence": round(
                                    confidence,
                                    2
                                )
                            }
                        )


                        progress.progress(
                            (i + 1) / len(df)
                        )


                    result_df = pd.DataFrame(
                        results
                    )


                    st.markdown(
                        "### 📊 Bulk Analysis Results"
                    )


                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )


                    st.markdown(
                        "### 📌 Sentiment Summary"
                    )


                    summary = (
                        result_df["Sentiment"]
                        .value_counts()
                    )


                    metric1, metric2, metric3 = st.columns(3)


                    with metric1:

                        st.metric(
                            "😊 Positive",
                            int(
                                summary.get(
                                    "positive",
                                    0
                                )
                            )
                        )


                    with metric2:

                        st.metric(
                            "😐 Neutral",
                            int(
                                summary.get(
                                    "neutral",
                                    0
                                )
                            )
                        )


                    with metric3:

                        st.metric(
                            "😞 Negative",
                            int(
                                summary.get(
                                    "negative",
                                    0
                                )
                            )
                        )


                    st.markdown(
                        "### 📈 Overall Sentiment"
                    )


                    chart = pd.DataFrame(
                        {
                            "Sentiment": [
                                "Negative",
                                "Neutral",
                                "Positive"
                            ],
                            "Count": [
                                int(
                                    summary.get(
                                        "negative",
                                        0
                                    )
                                ),
                                int(
                                    summary.get(
                                        "neutral",
                                        0
                                    )
                                ),
                                int(
                                    summary.get(
                                        "positive",
                                        0
                                    )
                                )
                            ]
                        }
                    )


                    st.bar_chart(
                        chart,
                        x="Sentiment",
                        y="Count"
                    )


                    st.markdown(
                        "### 📊 Sentiment Percentages"
                    )


                    total = len(
                        result_df
                    )


                    positive_count = int(
                        summary.get(
                            "positive",
                            0
                        )
                    )


                    neutral_count = int(
                        summary.get(
                            "neutral",
                            0
                        )
                    )


                    negative_count = int(
                        summary.get(
                            "negative",
                            0
                        )
                    )


                    positive_percent = (
                        positive_count / total
                    )


                    neutral_percent = (
                        neutral_count / total
                    )


                    negative_percent = (
                        negative_count / total
                    )


                    st.write(
                        f"😊 Positive: {positive_percent * 100:.2f}%"
                    )

                    st.progress(
                        positive_percent
                    )


                    st.write(
                        f"😐 Neutral: {neutral_percent * 100:.2f}%"
                    )

                    st.progress(
                        neutral_percent
                    )


                    st.write(
                        f"😞 Negative: {negative_percent * 100:.2f}%"
                    )

                    st.progress(
                        negative_percent
                    )


                    csv_data = result_df.to_csv(
                        index=False
                    )


                    st.download_button(
                        "⬇️ Download Analysis Results",
                        csv_data,
                        "sentiment_results.csv",
                        "text/csv",
                        use_container_width=True
                    )


                    st.divider()


                    st.markdown(
                        "## 📊 EduSentiment Dashboard"
                    )


                    st.markdown(
                        "### 📌 Feedback Overview"
                    )


                    dashboard_col1, dashboard_col2, dashboard_col3, dashboard_col4 = st.columns(4)


                    with dashboard_col1:

                        st.metric(
                            "📋 Total Feedback",
                            total
                        )


                    with dashboard_col2:

                        st.metric(
                            "😊 Positive",
                            positive_count,
                            f"{positive_percent * 100:.1f}%"
                        )


                    with dashboard_col3:

                        st.metric(
                            "😐 Neutral",
                            neutral_count,
                            f"{neutral_percent * 100:.1f}%"
                        )


                    with dashboard_col4:

                        st.metric(
                            "😞 Negative",
                            negative_count,
                            f"{negative_percent * 100:.1f}%"
                        )


                    st.markdown(
                        "### 📈 Sentiment Distribution"
                    )


                    dashboard_chart = pd.DataFrame(
                        {
                            "Sentiment": [
                                "Positive",
                                "Neutral",
                                "Negative"
                            ],
                            "Count": [
                                positive_count,
                                neutral_count,
                                negative_count
                            ]
                        }
                    )


                    st.bar_chart(
                        dashboard_chart,
                        x="Sentiment",
                        y="Count"
                    )


                    st.markdown(
                        "### 📊 Sentiment Percentage"
                    )


                    percentage_chart = pd.DataFrame(
                        {
                            "Sentiment": [
                                "Positive",
                                "Neutral",
                                "Negative"
                            ],
                            "Percentage": [
                                positive_percent * 100,
                                neutral_percent * 100,
                                negative_percent * 100
                            ]
                        }
                    )


                    st.bar_chart(
                        percentage_chart,
                        x="Sentiment",
                        y="Percentage"
                    )


                    st.markdown(
                        "### 📋 Feedback Analysis"
                    )


                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )


    except Exception as e:

        st.error(
            f"Error while processing CSV: {e}"
        )