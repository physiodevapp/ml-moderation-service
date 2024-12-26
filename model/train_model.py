from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample data
data = [
  ("I love this!", 0), # 0 = Not offensive
  ("This is terrible!", 1), # 1 = Offensive
  ("What a great day!", 0), # 0 = Not offensive
  ("You are a fool!", 1) # 1 = Offensive
]

messages, labels = zip(*data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)

with open("model.pkl", "wb") as f:
  pickle.dump((vectorizer, model), f)

print("Model trained and saved to model.pkl")