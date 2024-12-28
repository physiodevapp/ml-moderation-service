from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import re
from nltk.corpus import stopwords

def preprocess_message(message):
    stop_words = set(stopwords.words('english'))
    message = message.lower()
    message = re.sub(r'[^\w\s]', '', message)
    message = " ".join(word for word in message.split() if word not in stop_words)
    return message

data = [
    ("I love this!", 0),
    ("This is terrible!", 1),
    ("What a great day!", 0),
    ("You are a fool!", 1),
    ("I hate this!", 1),
    ("You are stupid!", 1),
    ("This is fantastic!", 0),
    ("What a beautiful moment!", 0),
    ("You are amazing!", 0),
    ("This behavior is unacceptable!", 1),
    ("Get lost, idiot!", 1),
    ("You're brilliant!", 0),
    ("This is nonsense!", 1),
    ("I appreciate this!", 0),
    ("How rude of you!", 1)
]


messages, labels = zip(*data)
messages = [preprocess_message(msg) for msg in messages]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

with open("model.pkl", "wb") as f:
  pickle.dump((vectorizer, model), f)

print("Model trained and saved to model.pkl")