# demo2_svm.py
# Demo 2: Support Vector Machine (SVM) Classification with Hyperparameter Tuning
# Mirrors Demo 1 structure: data prep -> CV search -> train -> evaluate -> print -> save plots

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
)
from sklearn.decomposition import PCA

# ---------- 0) Output folder (ALWAYS beside this script) ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "introai2")
os.makedirs(OUT_DIR, exist_ok=True)
print("Saving to:", OUT_DIR)

# ---------- 1) Generate/reproducible data (binary, moderately separable) ----------
X, y = make_classification(
    n_samples=500,
    n_features=20,
    n_informative=6,
    n_redundant=2,
    n_clusters_per_class=2,
    class_sep=1.2,
    flip_y=0.02,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ---------- 2) Define pipeline ----------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(probability=True))
])

# ---------- 3) Hyperparameter grid + CV search ----------
param_grid = [
    {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10, 100]},
    {"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]},
]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True, verbose=0)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_

# ---------- 4) Evaluate ----------
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print(f"Best Params: {gs.best_params_}")
print(f"Test Accuracy: {acc:.3f}")
print("\nClassification Report:")
print(report)

# ---------- 5) Confusion Matrix Plot ----------
cm = confusion_matrix(y_test, y_pred)
fig = plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (SVM)")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, [0,1])
plt.yticks(tick_marks, [0,1])
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close(fig)

# ---------- 6) Decision Boundary via PCA(2D) ----------
# Project features to 2D for visualization only
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(StandardScaler().fit_transform(X_train))
X_test_2d = pca.transform(StandardScaler().fit_transform(X_test))

# Train a new simple model on 2D space for plotting boundary
plot_pipe = Pipeline([
    ("clf", SVC(kernel=best_model.get_params()["clf__kernel"],
                C=best_model.get_params()["clf__C"],
                gamma=best_model.get_params().get("clf__gamma", "scale"),
                probability=True))
])
plot_pipe.fit(X_train_2d, y_train)

# Meshgrid
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = plot_pipe.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig = plt.figure(figsize=(6,5))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, s=18, edgecolor="k", alpha=0.7)
plt.title("SVM Decision Regions (PCA 2D projection of features)")
plt.xlabel("PC1")
plt.ylabel("PC2")
db_path = os.path.join(OUT_DIR, "decision_boundary_pca2d.png")
plt.tight_layout()
plt.savefig(db_path, dpi=200)
plt.close(fig)

# ---------- 7) ROC Curve (binary) ----------
fig = plt.figure(figsize=(5,4))
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve (SVM)")
roc_path = os.path.join(OUT_DIR, "roc_curve.png")
plt.tight_layout()
plt.savefig(roc_path, dpi=200)
plt.close(fig)

# ---------- 8) Learning Curve ----------
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy"
)
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

fig = plt.figure(figsize=(6,5))
plt.plot(train_sizes, train_mean, marker="o", label="Training score")
plt.plot(train_sizes, test_mean, marker="s", label="Cross-val score")
plt.title("Learning Curve (SVM)")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.grid(True, alpha=0.3)
plt.legend()
lc_path = os.path.join(OUT_DIR, "learning_curve.png")
plt.tight_layout()
plt.savefig(lc_path, dpi=200)
plt.close(fig)

# ---------- 9) Save a text summary for easy copy/paste ----------
with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best Params: {gs.best_params_}\n")
    f.write(f"Test Accuracy: {acc:.3f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nSaved images to: {OUT_DIR}")
print(" - confusion_matrix.png")
print(" - decision_boundary_pca2d.png")
print(" - roc_curve.png")
print(" - learning_curve.png")
print("Summary saved to summary.txt")
