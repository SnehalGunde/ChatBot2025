import os
import sys
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Set base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.json")

# Load dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [entry["user_input"] for entry in data]
labels = [entry["intent"] for entry in data]

# Label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=1)
X_vec = vectorizer.fit_transform(texts)

# Handle class imbalance using SMOTE
class_counts = Counter(y)
min_class_samples = min(class_counts.values())
k_neighbors = min(5, min_class_samples - 1) if min_class_samples > 1 else 1
smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_vec, y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# Define hyperparameters
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5],
}
xgb_params = {
    "n_estimators": [100, 200],
    "learning_rate": [0.1, 0.3],
    "max_depth": [3, 6],
}
logreg_params = {
    "C": [0.1, 1.0],#, 10.0],
    "penalty": ["l2"],
    "solver": ["liblinear"],#, "saga"],
    "max_iter": [200],
}

print("\n🔍 Grid Search: RandomForest (degraded)...")
rf_params = {
    'n_estimators': [10, 20],
    'max_depth': [2, 4],
    'min_samples_split': [5, 10]
}
rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params,
    cv=2,
    scoring='precision_macro',
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("\n🔍 Grid Search: XGBoost (degraded)...")
xgb_params = {
    'n_estimators': [10, 30],
    'max_depth': [2, 3],
    'learning_rate': [0.2, 0.3],
    'subsample': [0.5]
}
xgb = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    xgb_params,
    cv=2,
    scoring='precision_macro',
    n_jobs=-1
)
xgb.fit(X_train, y_train)

print("\n🔍 Grid Search: Logistic Regression (degraded)...")
logreg_params = {
    'C': [0.001, 0.01],
    'penalty': ['l2'],
    'solver': ['liblinear']
}
logreg = GridSearchCV(
    LogisticRegression(),
    logreg_params,
    cv=2,
    scoring='precision_macro',
    n_jobs=-1
)
logreg.fit(X_train, y_train)


# Store all models
models = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "Logistic Regression": logreg,
}

# Evaluate and plot
print("\n📊 Model Performance Summary:")
metrics = {}
for name, model_cv in models.items():
    print(f"\nEvaluating model {name}...")
    model = model_cv.best_estimator_
    y_pred = model.predict(X_test)

    print(f"\n🧠 Model: {name}")
    print(f"Best Parameters: {model_cv.best_params_}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    metrics[name] = f1

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(ticks=np.arange(len(label_encoder.classes_)), labels=label_encoder.classes_, rotation=45, ha='right')
    plt.yticks(ticks=np.arange(len(label_encoder.classes_)), labels=label_encoder.classes_)
    
    # Numbers inside boxes
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    plt.colorbar()
    plt.tight_layout()
    cm_filename = os.path.join(BASE_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_filename)
    print(f"📁 Confusion matrix saved: {cm_filename}")
    plt.close()

# Save the best model
best_model_name = max(metrics, key=metrics.get)
final_model = models[best_model_name].best_estimator_
model_path = os.path.join(BASE_DIR, "models")
os.makedirs(model_path, exist_ok=True)

joblib.dump(final_model, os.path.join(model_path, "best_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_path, "vectorizer.pkl"))
joblib.dump(label_encoder, os.path.join(model_path, "label_encoder.pkl"))

print(f"\n✅ Saved best model: {best_model_name}")
print("📦 Artifacts saved in /models/")
