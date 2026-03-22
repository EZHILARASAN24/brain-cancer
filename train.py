import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import matplotlib
matplotlib.use("Agg")   # ✅ Add this line
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATASET_PATH = "Dataset"
IMG_SIZE = (128, 128)
CATEGORIES = ["Normal", "Stroke"]

# Load dataset
def load_data():
    X, y = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATASET_PATH, category)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMG_SIZE)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(gray.flatten())
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)

print("📂 Loading dataset...")
X, y = load_data()

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models
models = {
    "DecisionTree": DecisionTreeClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "ExtraTrees": ExtraTreesClassifier(),
    "Bagging": BaggingClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

os.makedirs("saved_models", exist_ok=True)
os.makedirs("results", exist_ok=True)

results = {}

for name, model in models.items():
    print(f"🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    # Save model
    joblib.dump(model, f"saved_models/{name}.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f"{name} - Confusion Matrix")
    plt.savefig(f"results/{name}_confusion_matrix.png")
    plt.close()

print("✅ Training complete! Models saved in 'saved_models/'")
print("📊 Results:", results)
