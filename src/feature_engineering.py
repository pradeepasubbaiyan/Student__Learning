"""Feature engineering utilities for the Student Learning Analytics System."""

import os
from typing import List

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_FILE = os.path.join(DATA_DIR, "student_analytics_cleaned.csv")


def load_cleaned_data() -> pd.DataFrame:
    """Load the cleaned analytics dataset from disk."""
    return pd.read_csv(CLEANED_FILE)


def add_engagement_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags for students who are showing early signs of being at risk."""
    df = df.copy()
    df["low_attendance_flag"] = df["overall_attendance"] < 0.75
    df["low_quiz_accuracy_flag"] = df["average_quiz_accuracy"] < 0.50
    df["low_mock_average_flag"] = df["mock_test_average"] < 50
    df["low_study_hours_flag"] = df["study_hours"] < 10
    df["low_assignment_score_flag"] = df["average_assignment_score"] < 60
    df["high_delay_flag"] = df["assignment_delay_average"] > 5
    return df


def calculate_learning_gap_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a normalized learning gap score to help prioritize support."""
    df = df.copy()
    df["learning_gap_score"] = (
        (1 - df["overall_attendance"]) * 0.25
        + (1 - df["average_quiz_accuracy"]) * 0.30
        + (1 - (df["mock_test_average"] / 100)) * 0.20
        + (1 - (df["average_assignment_score"] / 100)) * 0.15
        + (1 - (df["study_hours"] / 40)) * 0.10
    )
    df["learning_gap_score"] = df["learning_gap_score"].clip(0, 1)
    return df


def rank_students(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Add a relative rank based on combined student performance metrics."""
    df = df.copy()
    df["student_rank"] = (
        df[feature_columns]
        .rank(method="average", ascending=False)
        .mean(axis=1)
    )
    df["student_rank"] = df["student_rank"].rank(method="dense", ascending=True).astype(int)
    return df


def enrich_features() -> pd.DataFrame:
    """Load cleaned dataset and add derived features for analytics."""
    df = load_cleaned_data()
    df = add_engagement_risk_flags(df)
    df = calculate_learning_gap_score(df)
    df = rank_students(
        df,
        feature_columns=[
            "average_quiz_accuracy",
            "mock_test_average",
            "average_assignment_score",
            "overall_attendance",
            "engagement_score",
        ],
    )
    df["learning_category"] = pd.cut(
        df["learning_gap_score"],
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["Excellent", "Good", "Average", "Needs Attention", "At Risk"],
    )
    return df


def main() -> None:
    """Entry point for feature engineering and enrichment."""
    print("Enriching student analytics features...")
    enriched_df = enrich_features()
    enriched_df.to_csv(os.path.join(DATA_DIR, "student_analytics_enriched.csv"), index=False)
    print("Saved enriched student analytics dataset.")


if __name__ == "__main__":
    main()
