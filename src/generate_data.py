"""Generate synthetic student learning data for the Student Learning Analytics System."""

import os
from datetime import datetime, timedelta
import random
import itertools

import numpy as np
import pandas as pd
from faker import Faker

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
STUDENT_COUNT = 2000
SUBJECTS = ["Math", "Physics", "Chemistry", "Biology", "English", "History"]
VIDEO_TOPICS = ["Introduction", "Lecture", "Practice", "Review", "Exam Tips"]
ASSIGNMENT_STATUS = ["On Time", "Delayed", "Missing"]

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


def ensure_data_dir() -> None:
    """Create the data directory if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def generate_students(student_count: int = STUDENT_COUNT) -> pd.DataFrame:
    """Generate student personal profiles."""
    student_ids = [f"S{1000 + i}" for i in range(student_count)]
    names = [fake.name() for _ in range(student_count)]
    genders = np.random.choice(["Male", "Female", "Other"], size=student_count, p=[0.48, 0.48, 0.04])
    join_dates = [fake.date_between(start_date="-2y", end_date="today") for _ in range(student_count)]

    # Use random ability band to influence later performance values
    ability_band = np.random.choice(["High", "Medium", "Low"], size=student_count, p=[0.25, 0.5, 0.25])

    df = pd.DataFrame(
        {
            "student_id": student_ids,
            "name": names,
            "gender": genders,
            "join_date": join_dates,
            "ability_band": ability_band,
        }
    )

    return df


def generate_student_profiles(students: pd.DataFrame) -> pd.DataFrame:
    """Generate and save the base student profile dataset."""
    profiles = students.copy()

    def sample_attendance_rate(ability: str) -> float:
        if ability == "High":
            return float(np.clip(np.random.normal(0.92, 0.04), 0.75, 1.0))
        if ability == "Medium":
            return float(np.clip(np.random.normal(0.82, 0.08), 0.55, 0.98))
        return float(np.clip(np.random.normal(0.68, 0.10), 0.30, 0.88))

    def sample_study_hours(ability: str) -> float:
        if ability == "High":
            return float(np.clip(np.random.normal(16, 3), 8, 28))
        if ability == "Medium":
            return float(np.clip(np.random.normal(11, 3.5), 4, 20))
        return float(np.clip(np.random.normal(6, 2.5), 1, 14))

    profiles["attendance_rate"] = profiles["ability_band"].apply(sample_attendance_rate)
    profiles["study_hours_per_week"] = profiles["ability_band"].apply(sample_study_hours)
    profiles["study_hours_per_week"] = profiles["study_hours_per_week"].apply(
        lambda value: float(np.clip(np.random.normal(value, 2.0), 1.0, 30.0))
    )
    profiles["engagement_score"] = np.clip(
        0.2 * profiles["attendance_rate"]
        + 0.3 * (profiles["study_hours_per_week"] / 30)
        + 0.1 * np.random.normal(0.5, 0.12, len(profiles)),
        0.05,
        0.98,
    )

    # Derive consistency-based metrics that will guide dataset realism
    profiles["video_completion_rate"] = np.clip(
        profiles["attendance_rate"] * 0.7 + profiles["engagement_score"] * 0.25 + np.random.normal(0.0, 0.08, len(profiles)),
        0.2,
        1.0,
    )

    # Create a base performance strength value for later labels
    performance_strength = (
        0.5 * profiles["attendance_rate"]
        + 0.25 * (profiles["study_hours_per_week"] / 30)
        + 0.25 * profiles["video_completion_rate"]
    )
    profiles["performance_strength"] = np.clip(performance_strength, 0.0, 1.0)

    return profiles


def generate_attendance(profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate attendance log data for each student over 30 class sessions."""
    records = []
    start_date = datetime.today() - timedelta(days=120)
    for _, row in profiles.iterrows():
        total_sessions = 30
        present_probability = row["attendance_rate"]
        for session_index in range(total_sessions):
            record_date = (start_date + timedelta(days=4 * session_index)).date()
            present = np.random.rand() < present_probability
            records.append(
                {
                    "student_id": row["student_id"],
                    "session_date": record_date,
                    "present": int(present),
                }
            )

    attendance_df = pd.DataFrame(records)
    return attendance_df


def generate_video_logs(profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate video engagement logs per student and topic."""
    rows = []
    for _, row in profiles.iterrows():
        watched_topics = np.random.choice(VIDEO_TOPICS, size=5, replace=True)
        for topic in watched_topics:
            watch_duration = np.clip(
                np.random.normal(18, 10) * row["video_completion_rate"],
                2,
                45,
            )
            completed = np.random.rand() < row["video_completion_rate"]
            rows.append(
                {
                    "student_id": row["student_id"],
                    "topic": topic,
                    "watch_duration_minutes": round(watch_duration, 1),
                    "completion": int(completed),
                }
            )
    return pd.DataFrame(rows)


def generate_quiz_attempts(profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate quiz scores per subject for each student."""
    rows = []
    for _, row in profiles.iterrows():
        for subject in SUBJECTS:
            accuracy = np.clip(
                np.random.normal(0.5 + 0.4 * row["performance_strength"], 0.14),
                0.05,
                0.98,
            )
            quiz_count = random.randint(3, 6)
            for attempt in range(1, quiz_count + 1):
                rows.append(
                    {
                        "student_id": row["student_id"],
                        "subject": subject,
                        "quiz_attempt": attempt,
                        "accuracy": round(np.clip(accuracy + np.random.normal(0, 0.08), 0.01, 1.0), 3),
                        "score": int(np.round(accuracy * 100 + np.random.normal(0, 7))),
                    }
                )
    return pd.DataFrame(rows)


def generate_mock_tests(profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate full mock test records with marks across subjects."""
    rows = []
    for _, row in profiles.iterrows():
        mock_count = random.randint(2, 4)
        for test_index in range(1, mock_count + 1):
            base_score = 40 + 55 * row["performance_strength"]
            variation = np.random.normal(0, 10)
            total_marks = int(np.clip(base_score + variation, 10, 100))
            rows.append(
                {
                    "student_id": row["student_id"],
                    "mock_test_id": f"M{test_index}",
                    "date": (datetime.today() - timedelta(days=random.randint(10, 100))).date(),
                    "total_marks": total_marks,
                }
            )
    return pd.DataFrame(rows)


def generate_assignments(profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate assignment records with submission quality and delays."""
    rows = []
    for _, row in profiles.iterrows():
        for subject in SUBJECTS:
            score = np.clip(np.random.normal(65 + 25 * row["performance_strength"], 12), 20, 100)
            delay_days = int(
                np.clip(
                    np.random.normal(2 + 8 * (1 - row["attendance_rate"]), 3),
                    0,
                    20,
                )
            )
            status = (
                "On Time" if delay_days == 0 else "Delayed" if delay_days <= 10 else "Missing"
            )
            rows.append(
                {
                    "student_id": row["student_id"],
                    "subject": subject,
                    "assignment_score": int(score),
                    "delay_days": delay_days,
                    "status": status,
                    "submitted_date": (datetime.today() - timedelta(days=random.randint(1, 40))).date(),
                }
            )
    return pd.DataFrame(rows)


def generate_engagement(profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate engagement survey and behavioral metrics."""
    rows = []
    for _, row in profiles.iterrows():
        weekly_videos = int(np.clip(np.random.normal(4 + 6 * row["video_completion_rate"], 1.8), 0, 10))
        weekly_forum_posts = int(np.clip(np.random.normal(1 + 4 * row["engagement_score"], 1.2), 0, 10))
        weekly_study_sessions = int(np.clip(np.random.normal(3 + 5 * row["engagement_score"], 1.5), 0, 10))
        rows.append(
            {
                "student_id": row["student_id"],
                "weekly_videos_watched": weekly_videos,
                "weekly_forum_posts": weekly_forum_posts,
                "weekly_study_sessions": weekly_study_sessions,
                "engagement_rating": round(np.clip(row["engagement_score"] * 5 + np.random.normal(0, 0.5), 1, 5), 1),
            }
        )
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save a DataFrame to a CSV file inside the data directory."""
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved {filename} ({len(df)} rows)")


def create_datasets() -> None:
    """Generate all synthetic datasets and save them to disk."""
    ensure_data_dir()

    students = generate_students()
    profiles = generate_student_profiles(students)

    save_csv(profiles.drop(columns=["performance_strength"]), "students.csv")
    save_csv(generate_attendance(profiles), "attendance.csv")
    save_csv(generate_video_logs(profiles), "video_logs.csv")
    save_csv(generate_quiz_attempts(profiles), "quiz_attempts.csv")
    save_csv(generate_mock_tests(profiles), "mock_tests.csv")
    save_csv(generate_assignments(profiles), "assignments.csv")
    save_csv(generate_engagement(profiles), "engagement.csv")


def main() -> None:
    """Entry point for dataset generation."""
    print("Starting synthetic dataset generation...")
    create_datasets()
    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
