"""
pages/resume_analyzer.py
--------------------------
Resume Analyzer page: upload a PDF/DOCX/TXT resume (or reuse one built
in Resume Builder), run it through the classifier + ATS regressor, and
show a full breakdown: predicted category, ATS score gauge, strengths,
weaknesses, and improvement suggestions.
"""

import streamlit as st

from utils.ats_analysis import analyze_resume
from utils.charts import ats_gauge, category_probability_bar, skill_distribution_bar
from utils.database import save_resume_record
from utils.resume_input import resume_input_widget
from utils.text_processing import extract_years_of_experience, preprocess_for_model


def render(models):
    st.markdown("## 🔍 Resume Analyzer")
    st.caption("Upload a resume, paste text, or reuse your last one — and get ML-driven feedback.")

    resume_text = resume_input_widget("analyzer")

    if resume_text:
        if st.button("🔬 Analyze Resume", type="primary"):
            clean = preprocess_for_model(resume_text)
            years = extract_years_of_experience(resume_text)

            predicted_category, confidence, proba_map = models.predict_category(clean)
            ats_score = models.predict_ats_score(clean, experience_years=years)
            analysis = analyze_resume(resume_text, ats_score)

            st.session_state.last_resume_text = resume_text
            st.session_state.last_analysis = {
                "category": predicted_category, "confidence": confidence,
                "proba_map": proba_map, "ats_score": ats_score, "analysis": analysis,
            }

            try:
                save_resume_record(
                    name=predicted_category, resume_text=resume_text, source="upload",
                    predicted_category=predicted_category, category_confidence=confidence,
                    ats_score=ats_score, skills=analysis["skills_found"],
                )
            except Exception:
                pass

    result = st.session_state.get("last_analysis")
    if result:
        st.divider()
        st.markdown("### Results")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Predicted Category", result["category"], f"{result['confidence']*100:.1f}% confidence")
            st.plotly_chart(category_probability_bar(result["proba_map"]), width='stretch')
        with c2:
            st.plotly_chart(ats_gauge(result["analysis"]["score"]), width='stretch')

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Word Count", result["analysis"]["word_count"])
        m2.metric("Action Verbs Used", result["analysis"]["action_verb_count"])
        m3.metric("Quantified Bullets", result["analysis"]["quantified_bullets"])
        m4.metric("Years of Experience", result["analysis"]["years_experience"])

        if result["analysis"]["skills_found"]:
            st.plotly_chart(skill_distribution_bar(result["analysis"]["skills_found"]), width='stretch')

        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.markdown("#### ✅ Strengths")
            for s in result["analysis"]["strengths"]:
                st.markdown(f"- {s}")
        with scol2:
            st.markdown("#### ⚠️ Weaknesses")
            if result["analysis"]["weaknesses"]:
                for w in result["analysis"]["weaknesses"]:
                    st.markdown(f"- {w}")
            else:
                st.markdown("- None detected!")
        with scol3:
            st.markdown("#### 💡 Suggestions")
            for sug in result["analysis"]["suggestions"]:
                st.markdown(f"- {sug}")
