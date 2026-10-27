# EduSentiment AI

Student Feedback Sentiment Analysis using NLP, LSTM and GRU.

## Project objective
Analyze student feedback and classify it as:
- Positive
- Neutral
- Negative

## Concepts covered
- NLP text preprocessing
- Tokenization
- Stop-word removal
- Lemmatization
- TF-IDF
- Word Embedding
- LSTM
- GRU
- Model evaluation
- Streamlit deployment

## Folder structure

```text
EduSentiment-AI/
├── data/
│   └── feedback.csv
├── models/
├── app.py
├── train.py
├── preprocess.py
├── requirements.txt
└── README.md
```

## Setup in VS Code

Open the project folder in VS Code.

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install packages:

```powershell
python -m pip install -r requirements.txt
```

Train the LSTM and GRU models:

```powershell
python train.py
```

Run the Streamlit application:

```powershell
python -m streamlit run app.py
```

The browser will open the EduSentiment AI dashboard.

## Important
The included dataset is a small educational/demo dataset. For a final academic project, replace it with a larger real student-feedback dataset and retrain the models.
