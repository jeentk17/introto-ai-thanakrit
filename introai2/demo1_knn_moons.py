"""
Demo 1: k-NN Classification on a Synthetic Dataset (make_moons)
- Saves outputs in folder: introai2 (next to the script)
Run: python demo1_knn_moons.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === output folder relative to script ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "introai1")
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving to:", OUT_DIR)

# 1) Data
X, y = make_moons(n_samples=500, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2) Visualize dataset
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], s=15)
plt.title("Dataset: Two Interleaving Moons")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dataset.png"), dpi=160)
plt.close()

# 3) Model selection over k
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
cv_scores = []
for k in k_values:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_idx = int(np.argmax(cv_scores))
best_k = k_values[best_idx]

plt.figure(figsize=(6, 5))
plt.plot(k_values, cv_scores, marker="o")
plt.title("Model Selection: CV Accuracy vs k")
plt.xlabel("k (neighbors)")
plt.ylabel("Cross-validated Accuracy")
plt.grid(True, linewidth=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_selection.png"), dpi=160)
plt.close()

# 4) Train best model
best_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])
best_pipe.fit(X_train, y_train)

# 5) Evaluate
y_pred = best_pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# 6) Decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = best_pipe.predict(grid).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.25)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=15)
plt.title(f"Decision Boundary (k={best_k})")
plt.xlabel("x1")
plt.ylabel("x2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "decision_boundary.png"), dpi=160)
plt.close()

# 7) Confusion matrix plot
plt.figure(figsize=(5.8, 5.2))
im = plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha='center', va='center')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=160)
plt.close()

# 8) Print and save a summary
summary = f"""Best k: {best_k}
Test Accuracy: {acc:.3f}

Classification Report:
{classification_report(y_test, y_pred)}
"""
print(summary)
with open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
    f.write(summary)
