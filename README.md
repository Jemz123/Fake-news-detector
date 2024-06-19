import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import numpy as np

# Load the dataset (assuming Fake.csv has already been loaded and processed)
data = pd.read_csv('FakeRED.csv')

# Drop rows with missing 'text' values
data = data.dropna(subset=['text'])

# Split into features and target
x, y = data['text'], data['label']

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
x_vectorized = vectorizer.fit_transform(x)

# Initialize and train the model (assuming the model has already been trained)
clf = LinearSVC()
clf.fit(x_vectorized, y)

# Save the trained model for later use
joblib.dump(clf, 'linear_svc_model.pkl')

# Load the trained model
clf = joblib.load('linear_svc_model.pkl')

# Read text from mytext.txt
with open("mytext.txt", "r", encoding="utf-8") as f:
    text_samples = f.readlines()

# Strip whitespace and newline characters from each line
text_samples = [text.strip() for text in text_samples if text.strip()]

if text_samples:
    for i, text_sample in enumerate(text_samples):
        # Vectorize the text sample
        vectorized_text = vectorizer.transform([text_sample])

        # Predict the label using the trained model
        predicted_label = clf.predict(vectorized_text)[0]

        # Print the predicted label
        if predicted_label == 0:
            print(f"Predicted label {i + 1}: 'REAL'")
        elif predicted_label == 1:
            print(f"Predicted label {i + 1}: 'FAKE'")
        else:
            print(f"Predicted label {i + 1}: Invalid prediction: {predicted_label}. Expected 0 (REAL) or 1 (FAKE).")
else:
    print("No valid text found in mytext.txt. Please provide valid text.")
