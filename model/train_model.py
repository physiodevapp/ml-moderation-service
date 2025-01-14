import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import pickle
import re
from collections import Counter

# Text preprocessing function
def preprocess_message(message):
    message = message.lower()  # Convertir a minúsculas
    message = re.sub(r'[^\w\s]', '', message)  # Eliminar caracteres no alfanuméricos
    return message

# Load the Hate Speech dataset
df = pd.read_csv('labeled_data.csv')

# Use only the relevant columns
df = df[['tweet', 'class']]

# Rename classes to simplify (0 = not offensive, 1 = offensive)
df['class'] = df['class'].apply(lambda x: 1 if x in [0, 1] else 0)

# Preprocess the messages
df['tweet'] = df['tweet'].apply(preprocess_message)

# Split data into messages and labels
messages = df['tweet']
labels = df['class']

# Vectorize the text using bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5)
X = vectorizer.fit_transform(messages)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Apply oversampling to the training data
ros = RandomOverSampler(random_state=42)  # Initialize the oversampling method
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)  # Rebalance the dataset

# Check the new class distribution
print(f"Distribución original: {Counter(y_train)}")
print(f"Distribución después de oversampling: {Counter(y_resampled)}")

# Create the Logistic Regression model
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# Perform cross-validation to evaluate the model
print("\nValidación cruzada:")
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='f1_macro')
print(f"F1-Score en cada fold: {cv_scores}")
print(f"F1-Score promedio: {cv_scores.mean():.4f}")

# Train the model with the balanced data
model.fit(X_resampled, y_resampled)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print("\nClassification Report (evaluación final):")
print(classification_report(y_test, y_pred))

# Confusion matrix to analyze errors
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# Save the model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("\nModel trained and saved to model.pkl")
