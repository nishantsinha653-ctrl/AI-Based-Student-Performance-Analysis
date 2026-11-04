"""
AI-Based Student Performance Analysis and Prediction
Educational demonstration using synthetic data only.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_performance.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["attendance_pct", "assignment_pct", "practice_hours", "previous_score"]
REQUIRED = FEATURES + ["final_score"]

def load_and_validate_data():
    df = pd.read_csv(DATA_FILE)
    missing_columns = [col for col in REQUIRED if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("The dataset is empty.")
    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["attendance_pct", "assignment_pct", "previous_score", "final_score"]].isna().any().any():
        raise ValueError("Score and percentage columns contain missing/non-numeric values.")
    if not df["attendance_pct"].between(0, 100).all():
        raise ValueError("attendance_pct must be between 0 and 100.")
    if not df["assignment_pct"].between(0, 100).all():
        raise ValueError("assignment_pct must be between 0 and 100.")
    if not df["previous_score"].between(0, 100).all() or not df["final_score"].between(0, 100).all():
        raise ValueError("Scores must be between 0 and 100.")
    if (df["practice_hours"] < 0).any():
        raise ValueError("practice_hours cannot be negative.")
    return df

def make_performance_category(score):
    if score < 50:
        return "Low"
    if score < 75:
        return "Medium"
    return "High"

def create_charts(df):
    plt.figure(figsize=(7, 5))
    plt.scatter(df["attendance_pct"], df["final_score"], alpha=0.7)
    plt.title("Attendance vs Final Score (Synthetic Data)")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Final score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "attendance_vs_score.png", dpi=150)
    plt.close()

    counts = df["performance_category"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Performance Category Distribution (Synthetic Data)")
    plt.xlabel("Performance category")
    plt.ylabel("Number of fictional records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_distribution.png", dpi=150)
    plt.close()

def main():
    df = load_and_validate_data()
    df["performance_category"] = df["final_score"].apply(make_performance_category)
    create_charts(df)

    X = df[FEATURES]
    y = df["performance_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\\nAI-BASED STUDENT PERFORMANCE ANALYSIS")
    print("-------------------------------------")
    print(f"Dataset records: {len(df)} (all fictional / synthetic)")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\\nMissing values:")
    print(df[REQUIRED].isna().sum().to_string())
    print("\\nSummary statistics:")
    print(df[REQUIRED].describe().round(2).to_string())
    print("\\nCategory counts:")
    print(df["performance_category"].value_counts().to_string())
    print(f"\\nTest accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=["Low", "Medium", "High"])
    plt.title("Confusion Matrix (Synthetic Demonstration)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"\\nCharts saved in: {OUTPUT_DIR}")
    print("\\nIMPORTANT: This is a classroom demonstration, not a validated tool for judging real students.")

if __name__ == "__main__":
    main()
