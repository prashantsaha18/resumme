"""
pages/skill_recommendation_page.py
-------------------------------------
Skill Recommendation page: predicts the job category from resume text,
then recommends in-demand skills for that category that aren't already
present in the resume.
"""

import streamlit as st

from utils.resume_input import resume_input_widget
from utils.skill_recommender import recommend_skills
from utils.text_processing import preprocess_for_model


def render(models):
    st.markdown("## 💡 Skill Recommendation")
    st.caption("Get skill suggestions tailored to your predicted job role.")

    resume_text = resume_input_widget("skills")

    if st.button("Recommend Skills", type="primary"):
        if not resume_text.strip():
            st.warning("Please provide a resume first (upload, paste, or reuse your last one).")
        else:
            clean = preprocess_for_model(resume_text)
            category, confidence, _ = models.predict_category(clean)
            rec = recommend_skills(resume_text, category)

            st.metric("Predicted Role", category, f"{confidence*100:.1f}% confidence")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### ✅ Detected Skills")
                if rec["detected_skills"]:
                    for s in rec["detected_skills"]:
                        st.markdown(f"<span class='metric-pill'>{s}</span>", unsafe_allow_html=True)
                else:
                    st.write("No recognized skills detected.")
            with c2:
                st.markdown("#### 🚀 Recommended Skills to Add")
                if rec["recommended_skills"]:
                    for s in rec["recommended_skills"]:
                        st.markdown(f"- **{s}**")
                    st.caption(f"These are commonly requested for {category} roles and weren't "
                               "found in your resume.")
                else:
                    st.write("Your resume already covers the top recommended skills for this role!")
