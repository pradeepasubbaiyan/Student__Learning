"""Streamlit dashboard for Student Learning Analytics."""

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ENRICHED_FILE = os.path.join(DATA_DIR, "student_analytics_enriched.csv")
RECOMMENDATION_FILE = os.path.join(DATA_DIR, "student_recommendations.csv")


def load_data() -> pd.DataFrame:
    """Load analytics and recommendation data."""
    students = pd.read_csv(ENRICHED_FILE)
    recommendations = pd.read_csv(RECOMMENDATION_FILE)
    return students, recommendations


def format_metric(value, decimals=2) -> str:
    """Format metric values for display."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def show_student_profile(student: pd.Series, rec: pd.Series) -> None:
    """Render student profile metrics and recommendations."""
    st.subheader(f"{student['name']} ({student['student_id']})")
    cols = st.columns(4)
    cols[0].metric("Attendance %", format_metric(student["overall_attendance"] * 100))
    cols[1].metric("Study Hours", format_metric(student["study_hours"]))
    cols[2].metric("Quiz Accuracy", format_metric(student["average_quiz_accuracy"] * 100))
    cols[3].metric("Mock Avg", format_metric(student["mock_test_average"]))

    st.markdown("**Learning Status**")
    status_cols = st.columns(3)
    status_cols[0].metric("Performance", student["performance"])
    status_cols[1].metric("Category", student["learning_category"])
    status_cols[2].metric("Gap Score", format_metric(student["learning_gap_score"]))

    with st.expander("Recommendations"):
        for item in rec["recommendations"].split(" | "):
            st.write(f"- {item}")


def show_dashboard_charts(students: pd.DataFrame) -> None:
    """Render overview charts for the analytics dashboard."""
    st.header("Dashboard Overview")

    st.subheader("Attendance Distribution")
    fig = px.histogram(students, x="overall_attendance", nbins=25, title="Attendance Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Study Hours Histogram")
    fig = px.histogram(students, x="study_hours", nbins=25, title="Study Hours Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quiz Accuracy vs Mock Average")
    fig = px.scatter(
        students,
        x="average_quiz_accuracy",
        y="mock_test_average",
        color="learning_category",
        title="Quiz Accuracy vs Mock Test Average",
        hover_data=["student_id", "name"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Performance by Category")
    perf_counts = students["performance"].value_counts().reset_index()
    perf_counts.columns = ["performance", "count"]
    fig = px.pie(perf_counts, values="count", names="performance", title="Performance Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_columns = [
        "overall_attendance",
        "study_hours",
        "video_completion_rate",
        "average_quiz_accuracy",
        "mock_test_average",
        "average_assignment_score",
        "engagement_score",
    ]
    corr = students[corr_columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_columns)))
    ax.set_yticks(range(len(corr_columns)))
    ax.set_xticklabels(corr_columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_columns)
    for i in range(len(corr_columns)):
        for j in range(len(corr_columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
    st.pyplot(fig)


def display_top_metrics(students: pd.DataFrame) -> None:
    """Display summary cards for top-level metrics."""
    st.title("Student Learning Analytics System")
    st.markdown("A complete analytics dashboard for student performance monitoring and personalized recommendations.")

    categories = students["learning_category"].value_counts().reindex(
        ["Excellent", "Good", "Average", "Needs Attention", "At Risk"], fill_value=0
    )
    cols = st.columns(5)
    for idx, category in enumerate(categories.index):
        cols[idx].metric(category, int(categories.loc[category]))


def run_app() -> None:
    """Launch the Streamlit application."""
    students, recommendations = load_data()

    st.sidebar.title("Student Search")
    selected_name = st.sidebar.text_input("Search by name")
    selected_id = st.sidebar.text_input("Search by student ID")
    filter_category = st.sidebar.selectbox(
        "Filter by learning category", ["All", "Excellent", "Good", "Average", "Needs Attention", "At Risk"]
    )

    filtered_students = students.copy()
    if selected_name:
        filtered_students = filtered_students[filtered_students["name"].str.contains(selected_name, case=False, na=False)]
    if selected_id:
        filtered_students = filtered_students[filtered_students["student_id"].str.contains(selected_id, case=False, na=False)]
    if filter_category != "All":
        filtered_students = filtered_students[filtered_students["learning_category"] == filter_category]

    display_top_metrics(students)
    show_dashboard_charts(students)

    st.header("Student Lookup")
    # Build friendly options showing both ID and name (easier to read in the dropdown)
    student_options = (
        filtered_students.apply(lambda r: f"{r['student_id']} — {r['name']}", axis=1).tolist()
    )
    if not student_options:
        st.info("No students match the current filters. Clear filters or choose a different category.")
        selected_student = None
    else:
        selected_label = st.selectbox("Select a student", student_options)
        # Extract student_id from the selected label (format: "ID — Name")
        selected_student = selected_label.split(" — ")[0] if selected_label else None

    if selected_student:
        student_row = students[students["student_id"] == selected_student].iloc[0]
        recommendation_row = recommendations[recommendations["student_id"] == selected_student].iloc[0]
        show_student_profile(student_row, recommendation_row)

        st.subheader("Detailed Performance Overview")
        st.write(student_row[[
            "overall_attendance",
            "study_hours",
            "video_completion_rate",
            "average_quiz_accuracy",
            "mock_test_average",
            "average_assignment_score",
            "assignment_delay_average",
            "engagement_score",
            "weekly_videos_watched",
            "weekly_forum_posts",
            "weekly_study_sessions",
        ]])

        st.subheader("Student Learning Gap Chart")
        gap_df = pd.DataFrame(
            {
                "Metric": [
                    "Attendance Gap",
                    "Quiz Gap",
                    "Mock Gap",
                    "Assignment Gap",
                    "Study Hours Gap",
                ],
                "Value": [
                    max(0, 0.75 - student_row["overall_attendance"]),
                    max(0, 0.50 - student_row["average_quiz_accuracy"]),
                    max(0, 50 - student_row["mock_test_average"]),
                    max(0, 60 - student_row["average_assignment_score"]),
                    max(0, 10 - student_row["study_hours"]),
                ],
            }
        )
        fig = px.bar(gap_df, x="Metric", y="Value", title="Learning Gap Components", text="Value")
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
