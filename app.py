"""
app.py
-------
AI Resume Builder — main Streamlit entry point.
Handles page config, global CSS (dark/light aware), sidebar navigation,
cached model loading, and renders the Home / About pages inline. All
other pages live in /pages and are dispatched from here.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from utils.database import init_db
from utils.model_loader import ModelBundle

st.set_page_config(
    page_title="AI Resume Builder",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — works with both Streamlit light and dark themes by relying on
# CSS variables where possible and translucent accents rather than hardcoded
# light backgrounds.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .app-card {
        background: var(--secondary-background-color, rgba(127,127,127,0.08));
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(127,127,127,0.18);
        margin-bottom: 1rem;
    }
    .metric-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(37, 99, 235, 0.12);
        color: #2563EB;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.4rem;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        color: rgba(127,127,127,0.95);
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .section-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.78rem;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 0.4rem;
    }
    .feature-icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
    }
    div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained ML models...")
def load_models() -> ModelBundle:
    bundle = ModelBundle()
    bundle.load()

    if not bundle.loaded:
        # Self-healing fallback: whether the committed models are missing
        # or simply incompatible with whatever numpy/scikit-learn version
        # got installed on this deployment, retrain from scratch in the
        # current environment rather than leaving the app unusable.
        with st.spinner(
            "First-time setup: training ML models in this environment "
            "(committed models were missing or incompatible)... this takes ~1-2 minutes."
        ):
            try:
                from dataset.generate_dataset import build_datasets
                from train import main as train_main

                dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
                if not os.path.exists(os.path.join(dataset_dir, "resume_dataset.csv")):
                    build_datasets(n_per_category=120)

                train_main()
                bundle = ModelBundle()
                bundle.load()
            except Exception as e:
                bundle.load_error = f"Auto-training failed: {type(e).__name__}: {e}"
                bundle.loaded = False

    return bundle


init_db()
models = load_models()

if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "last_resume_text" not in st.session_state:
    st.session_state.last_resume_text = ""
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

PAGES = [
    "Home",
    "Resume Builder",
    "Resume Analyzer",
    "ATS Score Predictor",
    "Resume Classification",
    "Job Matching",
    "Skill Recommendation",
    "Resume Ranking",
    "Model Performance",
    "About",
]

with st.sidebar:
    st.markdown("### 📄 AI Resume Builder")
    st.caption("Machine Learning powered resume toolkit")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    if models.loaded:
        st.success("ML models loaded ✅", icon="✅")
    else:
        st.error("Model auto-training failed. See main panel.", icon="⚠️")
    st.caption("Built with scikit-learn + Streamlit")

if not models.loaded and page != "About":
    st.error(
        f"**Models could not be loaded or trained automatically.**\n\n{models.load_error}\n\n"
        "This app normally self-heals by retraining models on first run if the "
        "committed ones are missing or incompatible — that attempt failed too, which "
        "usually points to a deeper environment issue (e.g. a missing system package). "
        "Try running `python train.py` manually from the project root and check the "
        "console output for the underlying error, then refresh this page."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------
if page == "Home":
    st.markdown('<div class="app-title">AI Resume Builder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Build, analyze, and optimize resumes using traditional '
        'Machine Learning — no LLMs involved.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="app-card"><div class="feature-icon">🧠</div>'
                     '<b>Resume Classification</b><br><span style="opacity:0.8;font-size:0.9rem">'
                     'Predicts your job category using a trained classifier.</span></div>',
                     unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="app-card"><div class="feature-icon">🎯</div>'
                     '<b>ATS Score Prediction</b><br><span style="opacity:0.8;font-size:0.9rem">'
                     'Regression model scores ATS-friendliness 0-100.</span></div>',
                     unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="app-card"><div class="feature-icon">🔍</div>'
                     '<b>Job Matching</b><br><span style="opacity:0.8;font-size:0.9rem">'
                     'TF-IDF + cosine similarity resume-to-JD matching.</span></div>',
                     unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="app-card"><div class="feature-icon">📊</div>'
                     '<b>Resume Ranking</b><br><span style="opacity:0.8;font-size:0.9rem">'
                     'Rank multiple resumes by score, match, and skills.</span></div>',
                     unsafe_allow_html=True)

    st.markdown("### Get Started")
    gcol1, gcol2, gcol3 = st.columns(3)
    with gcol1:
        st.info("**Build a resume** \u2192 Go to *Resume Builder* to fill in your details and "
                "download a polished ATS-friendly PDF.")
    with gcol2:
        st.info("**Analyze an existing resume** \u2192 Go to *Resume Analyzer* to upload a PDF/DOCX "
                "and get ML-driven feedback.")
    with gcol3:
        st.info("**Compare against a job posting** \u2192 Go to *Job Matching* to see your match "
                "percentage and missing keywords.")

    if models.loaded and models.metrics:
        st.markdown("### Model Snapshot")
        best_clf = models.metrics["classification"]["best_model"]
        best_clf_metrics = models.metrics["classification"]["results"][best_clf]
        best_reg = models.metrics["regression"]["best_model"]
        best_reg_metrics = models.metrics["regression"]["results"][best_reg]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best Classifier", best_clf)
        m2.metric("Classifier F1 Score", f"{best_clf_metrics['f1_score']:.3f}")
        m3.metric("Best ATS Regressor", best_reg)
        m4.metric("ATS Model R²", f"{best_reg_metrics['r2_score']:.3f}")

elif page == "About":
    st.markdown('<div class="app-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
This **AI Resume Builder** is a production-style project that demonstrates an end-to-end
**traditional Machine Learning** pipeline applied to resume intelligence — deliberately
built *without* Large Language Models.

#### Machine Learning Components
- **Resume Classification** — Logistic Regression, Random Forest, SVM, and Naive Bayes
  are trained and compared; the best performer (by weighted F1) is deployed.
- **ATS Score Prediction** — Linear Regression, Random Forest Regressor, and Gradient
  Boosting Regressor are trained and compared; the best performer (by R²) is deployed.
- **Resume ↔ Job Description Matching** — TF-IDF vectorization + cosine similarity.
- **Skill Recommendation** — rule-based mapping from predicted category to in-demand
  skills, filtered against skills already detected in the resume.
- **Resume Ranking** — combines ATS score, job match %, and detected skill count across
  multiple uploaded resumes.

#### Tech Stack
Python · Streamlit · scikit-learn · pandas · NumPy · NLTK · spaCy · Joblib ·
pdfplumber · python-docx · ReportLab · Plotly · SQLite

#### Notes on the Training Data
Public resume datasets (e.g. Kaggle's Resume Dataset) require external downloads that
aren't available in every deployment environment. This project ships with
`dataset/generate_dataset.py`, which synthesizes a structurally realistic resume corpus
(same schema: `Resume`, `Category`) so `train.py` runs out of the box. **Swap in a real
CSV with the same column names to retrain on real-world data** — no other code changes
required.

#### Disclaimer
ATS scores and job-match percentages are **model estimates**, not guarantees of how any
specific company's applicant tracking system will score a resume. Use them as directional
guidance, not absolute truth.
""")

elif page == "Resume Builder":
    from pages import resume_builder
    resume_builder.render(models)

elif page == "Resume Analyzer":
    from pages import resume_analyzer
    resume_analyzer.render(models)

elif page == "ATS Score Predictor":
    from pages import ats_predictor
    ats_predictor.render(models)

elif page == "Resume Classification":
    from pages import resume_classification
    resume_classification.render(models)

elif page == "Job Matching":
    from pages import job_matching_page
    job_matching_page.render(models)

elif page == "Skill Recommendation":
    from pages import skill_recommendation_page
    skill_recommendation_page.render(models)

elif page == "Resume Ranking":
    from pages import resume_ranking
    resume_ranking.render(models)

elif page == "Model Performance":
    from pages import model_performance
    model_performance.render(models)
