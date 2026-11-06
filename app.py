from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "student_performance.csv"
FEATURES = ["attendance_pct", "assignment_pct", "practice_hours", "previous_score"]
REQUIRED = FEATURES + ["final_score"]
CATEGORIES = ["Low", "Medium", "High"]

st.set_page_config(page_title="Student Performance Dashboard", page_icon="📊", layout="wide")
st.title("📊 Student Performance Analysis Dashboard")
st.caption("Python • Pandas • Scikit-learn • Streamlit")
st.warning("Educational demonstration only. This project uses fictional/synthetic data. Predictions and accuracy are not validated for real students and must not be used for grading, admissions, discipline, or other decisions.")

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError("Keep app.py in the same folder as student_performance.csv.")
    df = pd.read_csv(DATA_FILE)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    for c in REQUIRED:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[REQUIRED].isna().any().any():
        raise ValueError("The dataset contains missing or non-numeric required values.")
    if not df["attendance_pct"].between(0, 100).all() or not df["assignment_pct"].between(0, 100).all():
        raise ValueError("Attendance and assignment percentages must be between 0 and 100.")
    if not df["previous_score"].between(0, 100).all() or not df["final_score"].between(0, 100).all():
        raise ValueError("Scores must be between 0 and 100.")
    if (df["practice_hours"] < 0).any() or df.empty:
        raise ValueError("Practice hours must be non-negative and dataset cannot be empty.")
    df["performance_category"] = pd.cut(
        df["final_score"], bins=[-0.001, 50, 75, 100.001], labels=CATEGORIES, right=False
    ).astype(str)
    return df

@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df["performance_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=150, random_state=42)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, y_test, pred, accuracy_score(y_test, pred)

try:
    df = load_data()
    model, y_test, predictions, accuracy = train_model(df)
except Exception as exc:
    st.error(f"Could not prepare dashboard: {exc}")
    st.stop()

st.sidebar.header("Filters")
categories = st.sidebar.multiselect("Performance category", CATEGORIES, default=CATEGORIES)
min_attendance = st.sidebar.slider("Minimum attendance (%)", 0, 100, 0)
view = df[df["performance_category"].isin(categories) & (df["attendance_pct"] >= min_attendance)]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total records", f"{len(df):,}")
k2.metric("Average final score", f"{df.final_score.mean():.1f}/100")
k3.metric("Average attendance", f"{df.attendance_pct.mean():.1f}%")
k4.metric("Test accuracy", f"{accuracy:.1%}")

overview, records, predict_tab, evaluation = st.tabs(["📈 Overview", "📋 Student data", "🔮 Try a prediction", "🧠 Model evaluation"])
with overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Records by performance category")
        st.bar_chart(view["performance_category"].value_counts().reindex(CATEGORIES, fill_value=0))
    with right:
        st.subheader("Attendance vs final score")
        st.scatter_chart(view, x="attendance_pct", y="final_score", color="performance_category", size="assignment_pct")
    st.subheader("Average final score by category")
    avg = view.groupby("performance_category")["final_score"].mean().reindex(CATEGORIES).dropna()
    st.bar_chart(avg)
    st.caption(f"Showing {len(view)} of {len(df)} synthetic records after filters.")

with records:
    st.subheader("Student dataset")
    st.dataframe(view[REQUIRED + ["performance_category"]], use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                       "filtered_student_performance_demo.csv", "text/csv")

with predict_tab:
    st.subheader("Try a sample prediction")
    st.write("Enter illustrative values. The result is a demo model category, not a real student assessment.")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            attendance = st.slider("Attendance (%)", 0, 100, 80)
            assignment = st.slider("Assignment completion (%)", 0, 100, 75)
        with c2:
            practice = st.number_input("Weekly practice hours", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
            previous = st.slider("Previous score", 0, 100, 65)
        submitted = st.form_submit_button("Predict category", type="primary")
    if submitted:
        sample = pd.DataFrame([{
            "attendance_pct": attendance, "assignment_pct": assignment,
            "practice_hours": practice, "previous_score": previous
        }])
        result = model.predict(sample)[0]
        st.success(f"Demo model prediction: **{result}**")
        st.caption("Do not use this output to judge a real learner.")

with evaluation:
    st.subheader("Random Forest baseline")
    st.metric("Held-out test accuracy", f"{accuracy:.1%}")
    st.write("Trained on 75% of the synthetic records and evaluated on the remaining 25%.")
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=CATEGORIES, display_labels=CATEGORIES, ax=ax, colorbar=False)
    ax.set_title("Confusion matrix — synthetic demonstration")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("This score applies only to this generated sample and split; it is not real-world accuracy.")

st.divider()
st.caption("Learning project • Synthetic data • Not for consequential decisions")
