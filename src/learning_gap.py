"""Learning gap analysis for the Student Learning Analytics System."""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ENRICHED_FILE = os.path.join(DATA_DIR, "student_analytics_enriched.csv")


def load_enriched_data() -> pd.DataFrame:
    """Load enriched analytics data from disk."""
    return pd.read_csv(ENRICHED_FILE)


def categorize_student(row: pd.Series) -> str:
    """Assign a learning category to a student based on analytics metrics."""
    if row["learning_gap_score"] >= 0.75 or row["performance"] == "At Risk":
        return "At Risk"
    if row["learning_gap_score"] >= 0.55:
        return "Needs Attention"
    if row["learning_gap_score"] >= 0.35:
        return "Average"
    if row["learning_gap_score"] >= 0.15:
        return "Good"
    return "Excellent"


def analyze_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Add gap analysis columns and label students."""
    df = df.copy()
    df["attendance_gap"] = (0.75 - df["overall_attendance"]).clip(lower=0)
    df["quiz_gap"] = (0.5 - df["average_quiz_accuracy"]).clip(lower=0)
    df["mock_gap"] = (50 - df["mock_test_average"]).clip(lower=0)
    df["assignment_gap"] = (60 - df["average_assignment_score"]).clip(lower=0)
    df["study_hours_gap"] = (10 - df["study_hours"]).clip(lower=0)

    df["student_gap_category"] = df.apply(categorize_student, axis=1)
    return df


def get_at_risk_students(df: pd.DataFrame) -> pd.DataFrame:
    """Return students categorized as At Risk or Needs Attention."""
    return df[df["student_gap_category"].isin(["Needs Attention", "At Risk"])].copy()


def main() -> None:
    """Entry point to run learning gap analysis."""
    print("Running learning gap analysis...")
    enriched = load_enriched_data()
    gap_df = analyze_gaps(enriched)
    gap_df.to_csv(os.path.join(DATA_DIR, "student_learning_gap.csv"), index=False)
    print("Saved learning gap analysis results.")


if __name__ == "__main__":
    main()
