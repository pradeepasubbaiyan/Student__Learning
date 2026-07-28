"""Recommendation engine for personalized student learning guidance."""

import os
from typing import Dict, List

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ENRICHED_FILE = os.path.join(DATA_DIR, "student_analytics_enriched.csv")

BASE_RECOMMENDATIONS = {
    "attendance": "Attend more live classes and review missed sessions.",
    "quiz_accuracy": "Schedule targeted revision sessions and retake practice quizzes.",
    "assignments": "Follow a weekly planner to complete assignments on time.",
    "study_hours": "Use a daily study schedule with focused time blocks.",
    "mock_marks": "Take additional practice tests to improve exam readiness.",
}


def load_enriched_data() -> pd.DataFrame:
    """Load enriched analytics data from disk."""
    return pd.read_csv(ENRICHED_FILE)


def build_recommendations(row: pd.Series) -> List[str]:
    """Generate personalized recommendations for a single student."""
    recs = []
    if row.get("overall_attendance", 1.0) < 0.75:
        recs.append(BASE_RECOMMENDATIONS["attendance"])
    if row.get("average_quiz_accuracy", 1.0) < 0.55:
        recs.append(BASE_RECOMMENDATIONS["quiz_accuracy"])
    if row.get("average_assignment_score", 100) < 60 or row.get("assignment_delay_average", 0) > 5:
        recs.append(BASE_RECOMMENDATIONS["assignments"])
    if row.get("study_hours", 40) < 10:
        recs.append(BASE_RECOMMENDATIONS["study_hours"])
    if row.get("mock_test_average", 100) < 50:
        recs.append(BASE_RECOMMENDATIONS["mock_marks"])
    if not recs:
        recs.append("Continue the excellent learning habits and maintain steady progress.")
    return recs


def add_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Add recommendation text for every student in the dataset."""
    df = df.copy()
    df["recommendations"] = df.apply(lambda row: build_recommendations(row), axis=1)
    df["recommendation_summary"] = df["recommendations"].apply(lambda recs: " | ".join(recs))
    return df


def get_student_recommendations(student_id: str) -> Dict[str, object]:
    """Fetch recommendations for a single student by ID."""
    df = add_recommendations(load_enriched_data())
    student = df[df["student_id"] == student_id]
    if student.empty:
        return {"student_id": student_id, "recommendations": ["Student not found."]}
    student_row = student.iloc[0]
    return {
        "student_id": student_row["student_id"],
        "name": student_row.get("name", "Unknown"),
        "performance": student_row.get("performance", "Unknown"),
        "learning_category": student_row.get("learning_category", "Unknown"),
        "recommendations": student_row["recommendations"],
    }


def main() -> None:
    """Entry point to generate recommendation data."""
    print("Building personalized recommendations...")
    enriched = load_enriched_data()
    recommended = add_recommendations(enriched)
    recommended.to_csv(os.path.join(DATA_DIR, "student_recommendations.csv"), index=False)
    print("Saved student recommendations dataset.")


if __name__ == "__main__":
    main()
