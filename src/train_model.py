"""Train machine learning models for student performance prediction."""

import os
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLEANED_FILE = os.path.join(DATA_DIR, "student_analytics_cleaned.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")

FEATURE_COLUMNS = [
    "overall_attendance",
    "study_hours",
    "video_completion_rate",
    "average_quiz_accuracy",
    "average_quiz_score",
    "mock_test_average",
    "average_assignment_score",
    "assignment_delay_average",
    "engagement_score",
    "weekly_videos_watched",
    "weekly_forum_posts",
    "weekly_study_sessions",
]


def ensure_model_dir() -> None:
    """Create model directory if it does not exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load cleaned data for model training."""
    return pd.read_csv(CLEANED_FILE)


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, LabelEncoder, StandardScaler]:
    """Prepare features and labels for training and scaling."""
    df = df.copy()
    df = df.dropna(subset=FEATURE_COLUMNS + ["performance"])
    X = df[FEATURE_COLUMNS].values
    y = df["performance"].astype(str).values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_encoded, encoder, scaler


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """Train several models and return them in a dictionary."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, solver="liblinear"),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained {name}")
    return models


def evaluate_model(model: object, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder) -> Dict[str, object]:
    """Evaluate a single model and return performance metrics."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    matrix = confusion_matrix(y_test, y_pred)
    return {"accuracy": accuracy, "report": report, "confusion_matrix": matrix}


def compare_models(models: Dict[str, object], X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder) -> Dict[str, Dict[str, object]]:
    """Evaluate all trained models and compare results."""
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, label_encoder)
        print(f"{name} accuracy: {results[name]['accuracy']:.4f}")
    return results


def plot_confusion_matrix(matrix: np.ndarray, labels: np.ndarray, filename: str) -> None:
    """Save a confusion matrix plot to a file."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="left")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved confusion matrix to {filename}")


def choose_best_model(results: Dict[str, Dict[str, object]], models: Dict[str, object]) -> object:
    """Choose the model with the highest accuracy."""
    best_name = max(results.items(), key=lambda item: item[1]["accuracy"])[0]
    print(f"Selected best model: {best_name}")
    return models[best_name], best_name


def save_model(model: object, encoder: LabelEncoder, scaler: StandardScaler) -> None:
    """Save the fitted model, encoder, and scaler using joblib."""
    ensure_model_dir()
    joblib.dump({"model": model, "label_encoder": encoder, "scaler": scaler}, MODEL_FILE)
    print(f"Saved model package to {MODEL_FILE}")


def main() -> None:
    """Entry point for model training and evaluation."""
    print("Loading cleaned data for training...")
    cleaned = load_data()
    X, y, encoder, scaler = prepare_data(cleaned)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = train_models(X_train, y_train)
    results = compare_models(models, X_test, y_test, encoder)

    for name, result in results.items():
        print(f"\n=== {name} ===")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(result["report"])

    best_model, best_name = choose_best_model(results, models)
    save_model(best_model, encoder, scaler)
    plot_confusion_matrix(results[best_name]["confusion_matrix"], encoder.classes_, os.path.join(MODEL_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
