"""Preprocess synthetic student datasets and build cleaned analytic features."""

import os
from typing import List

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLEANED_FILE = os.path.join(DATA_DIR, "student_analytics_cleaned.csv")


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV file from the data directory."""
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


def remove_duplicates(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    """Drop duplicate rows based on a subset of columns."""
    before = len(df)
    result = df.drop_duplicates(subset=subset)
    after = len(result)
    print(f"Removed {before - after} duplicate rows from {subset}")
    return result


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with context-aware defaults."""
    result = df.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].fillna("Unknown")
        else:
            result[column] = result[column].fillna(result[column].median())
    return result


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    """Return a boolean mask indicating outliers using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def suppress_outliers(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Cap numeric outliers to the 1st and 99th percentiles."""
    result = df.copy()
    for col in numeric_columns:
        if col not in result.columns:
            continue
        low, high = result[col].quantile([0.01, 0.99])
        result[col] = result[col].clip(lower=low, upper=high)
    return result


def aggregate_student_metrics(students: pd.DataFrame, attendance: pd.DataFrame, video_logs: pd.DataFrame, quiz_attempts: pd.DataFrame, mock_tests: pd.DataFrame, assignments: pd.DataFrame, engagement: pd.DataFrame) -> pd.DataFrame:
    """Create aggregated learning metrics for each student."""
    attendance_summary = (
        attendance.groupby("student_id")["present"].mean().rename("overall_attendance")
    )

    video_summary = (
        video_logs.groupby("student_id")
        .agg(
            video_completion_rate=("completion", "mean"),
            average_video_duration=("watch_duration_minutes", "mean"),
            video_sessions=("topic", "count"),
        )
    )

    quiz_summary = (
        quiz_attempts.groupby("student_id")
        .agg(
            average_quiz_accuracy=("accuracy", "mean"),
            average_quiz_score=("score", "mean"),
            quiz_attempts=("quiz_attempt", "count"),
        )
    )

    mock_summary = (
        mock_tests.groupby("student_id")
        .agg(
            mock_test_average=("total_marks", "mean"),
            mock_test_count=("mock_test_id", "nunique"),
        )
    )

    assignment_summary = (
        assignments.groupby("student_id")
        .agg(
            average_assignment_score=("assignment_score", "mean"),
            assignment_delay_average=("delay_days", "mean"),
            on_time_submissions=("status", lambda s: (s == "On Time").sum()),
            delayed_submissions=("status", lambda s: (s == "Delayed").sum()),
            missing_submissions=("status", lambda s: (s == "Missing").sum()),
        )
    )

    engagement_summary = (
        engagement.groupby("student_id")
        .agg(
            engagement_score=("engagement_rating", "mean"),
            weekly_videos_watched=("weekly_videos_watched", "mean"),
            weekly_forum_posts=("weekly_forum_posts", "mean"),
            weekly_study_sessions=("weekly_study_sessions", "mean"),
        )
    )

    # Remove duplicate derived values from the raw student profile so aggregation joins do not collide.
    students = students.drop(columns=["attendance_rate", "video_completion_rate", "engagement_score"], errors="ignore")

    # Combine all metrics for analytics and modeling
    student_metrics = (
        students.set_index("student_id")
        .join(attendance_summary)
        .join(video_summary)
        .join(quiz_summary)
        .join(mock_summary)
        .join(assignment_summary)
        .join(engagement_summary)
        .reset_index()
    )

    student_metrics["overall_attendance"] = student_metrics["overall_attendance"].fillna(0.0)
    student_metrics["video_completion_rate"] = student_metrics["video_completion_rate"].fillna(0.0)
    student_metrics["average_quiz_accuracy"] = student_metrics["average_quiz_accuracy"].fillna(0.0)
    student_metrics["mock_test_average"] = student_metrics["mock_test_average"].fillna(0.0)
    student_metrics["average_assignment_score"] = student_metrics["average_assignment_score"].fillna(0.0)
    student_metrics["engagement_score"] = student_metrics["engagement_score"].fillna(0.0)

    student_metrics["study_hours"] = np.clip(
        student_metrics["study_hours_per_week"].fillna(student_metrics["study_hours_per_week"].median()),
        0,
        40,
    )

    # Maintain clean numeric formats for modeling and dashboards
    numeric_cols = [
        "overall_attendance",
        "video_completion_rate",
        "average_video_duration",
        "average_quiz_accuracy",
        "average_quiz_score",
        "mock_test_average",
        "average_assignment_score",
        "assignment_delay_average",
        "engagement_score",
        "weekly_videos_watched",
        "weekly_forum_posts",
        "weekly_study_sessions",
        "study_hours",
    ]
    student_metrics[numeric_cols] = student_metrics[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return student_metrics


def build_performance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create performance categories based on key metrics."""
    df = df.copy()
    conditions = [
        (df["overall_attendance"] >= 0.85) & (df["average_quiz_accuracy"] >= 0.75) & (df["mock_test_average"] >= 70),
        (df["overall_attendance"] >= 0.70) & (df["average_quiz_accuracy"] >= 0.55) & (df["mock_test_average"] >= 55),
    ]
    choices = ["High Performer", "Average Performer"]
    df["performance"] = np.select(conditions, choices, default="At Risk")
    return df


def save_cleaned_data(df: pd.DataFrame) -> None:
    """Save the cleaned student analytics dataset."""
    df.to_csv(CLEANED_FILE, index=False)
    print(f"Saved cleaned analytics file to {CLEANED_FILE}")


def preprocess_all() -> pd.DataFrame:
    """Load raw data, clean it, and save a cleaned analytics dataset."""
    students = load_csv("students.csv")
    attendance = load_csv("attendance.csv")
    video_logs = load_csv("video_logs.csv")
    quiz_attempts = load_csv("quiz_attempts.csv")
    mock_tests = load_csv("mock_tests.csv")
    assignments = load_csv("assignments.csv")
    engagement = load_csv("engagement.csv")

    students = remove_duplicates(students, subset=["student_id"])
    attendance = remove_duplicates(attendance, subset=["student_id", "session_date"])
    video_logs = remove_duplicates(video_logs, subset=["student_id", "topic"])
    quiz_attempts = remove_duplicates(quiz_attempts, subset=["student_id", "subject", "quiz_attempt"])
    mock_tests = remove_duplicates(mock_tests, subset=["student_id", "mock_test_id"])
    assignments = remove_duplicates(assignments, subset=["student_id", "subject"])
    engagement = remove_duplicates(engagement, subset=["student_id"])

    students = fill_missing_values(students)
    attendance = fill_missing_values(attendance)
    video_logs = fill_missing_values(video_logs)
    quiz_attempts = fill_missing_values(quiz_attempts)
    mock_tests = fill_missing_values(mock_tests)
    assignments = fill_missing_values(assignments)
    engagement = fill_missing_values(engagement)

    student_metrics = aggregate_student_metrics(
        students,
        attendance,
        video_logs,
        quiz_attempts,
        mock_tests,
        assignments,
        engagement,
    )

    student_metrics = suppress_outliers(
        student_metrics,
        numeric_columns=[
            "study_hours",
            "average_quiz_accuracy",
            "average_quiz_score",
            "mock_test_average",
            "average_assignment_score",
            "assignment_delay_average",
            "engagement_score",
            "average_video_duration",
        ],
    )

    student_metrics = build_performance_labels(student_metrics)
    save_cleaned_data(student_metrics)
    return student_metrics


def main() -> None:
    """Entry point for dataset preprocessing."""
    print("Preprocessing raw student datasets...")
    preprocess_all()
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
