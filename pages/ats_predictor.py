"""
pages/ats_predictor.py
------------------------
Standalone ATS Score Predictor page. Paste or reuse resume text, get an
ATS score gauge plus a strengths/weaknesses/suggestions breakdown from
the trained regression model.
"""

import streamlit as st

from utils.ats_analysis import analyze_resume
from utils.charts import ats_gauge
from utils.resume_input import resume_input_widget
from utils.text_processing import extract_years_of_experience, preprocess_for_model


def render(models):
    st.markdown("## 🎯 ATS Score Predictor")
    st.caption("Predicts how well your resume is likely to score with an Applicant Tracking System.")

    resume_text = resume_input_widget("ats")

    if st.button("Predict ATS Score", type="primary"):
        if not resume_text.strip():
            st.warning("Please provide a resume first (upload, paste, or reuse your last one).")
        else:
            clean = preprocess_for_model(resume_text)
            years = extract_years_of_experience(resume_text)
            score = models.predict_ats_score(clean, experience_years=years)
            analysis = analyze_resume(resume_text, score)

            col1, col2 = st.columns([1, 1.4])
            with col1:
                st.plotly_chart(ats_gauge(score), width='stretch')
                if score >= 75:
                    st.success("Great! This resume is well optimized for ATS parsing.")
                elif score >= 50:
                    st.warning("Decent, but there's room to improve ATS-friendliness.")
                else:
                    st.error("This resume may struggle to pass ATS filters. See suggestions below.")

            with col2:
                st.markdown("#### ✅ Strengths")
                for s in analysis["strengths"]:
                    st.markdown(f"- {s}")
                st.markdown("#### ⚠️ Weaknesses")
                if analysis["weaknesses"]:
                    for w in analysis["weaknesses"]:
                        st.markdown(f"- {w}")
                else:
                    st.markdown("- None detected!")
                st.markdown("#### 💡 Improvement Suggestions")
                for sug in analysis["suggestions"]:
                    st.markdown(f"- {sug}")
