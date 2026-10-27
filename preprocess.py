import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for package in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{package}")
    except LookupError:
        nltk.download(package, quiet=True)

stop_words = set(stopwords.words("english"))

# Important words ko remove nahi karna
important_words = {
    "not", "no", "never", "very", "too",
    "bad", "good", "poor", "excellent",
    "difficult", "easy", "boring", "helpful"
}

stop_words = stop_words - important_words

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)