"""
Demo 4: Naïve Bayes Classification
- Uses the 20 Newsgroups text dataset (subset for speed)
- Applies TF-IDF vectorization + Multinomial Naïve Bayes
- Evaluates accuracy, confusion matrix, and classification report
- Saves images and summary to introai4 folder
Run: python demo4_naivebayes.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# ---------------------------
# 1) Output folder
# ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "introai4")
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving to:", OUT_DIR)

# ---------------------------
# 2) Data
# ---------------------------
categories = ["rec.sport.baseball", "rec.sport.hockey", "sci.space", "talk.politics.mideast"]
newsgroups = fetch_20newsgroups(subset="all", categories=categories, remove=("headers","footers","quotes"))

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target, test_size=0.25, random_state=42
)

# ---------------------------
# 3) Pipeline
# ---------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# ---------------------------
# 4) Evaluation
# ---------------------------
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)

print("Test Accuracy:", round(acc, 3))
print("\nClassification Report:\n", report)

# ---------------------------
# 5) Save Confusion Matrix
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=newsgroups.target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
plt.close()

# ---------------------------
# 6) Save Summary
# ---------------------------
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write("Demo 4: Naïve Bayes Classification\n")
    f.write(f"Test Accuracy: {acc:.3f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("Saved images to:", OUT_DIR)
print(" - confusion_matrix.png")
print("Summary saved to summary.txt")
