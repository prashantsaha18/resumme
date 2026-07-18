"""
pages/resume_classification.py
---------------------------------
Standalone Resume Classification page: paste resume text and see the
predicted job category, confidence score, and full probability
distribution across all trained categories.
"""

import streamlit as st

from utils.charts import category_probability_bar
from utils.resume_input import resume_input_widget
from utils.text_processing import preprocess_for_model


def render(models):
    st.markdown("## 🧠 Resume Classification")
    st.caption("Predicts the most likely job category for a resume using a trained ML classifier.")

    resume_text = resume_input_widget("classify")

    if st.button("Classify Resume", type="primary"):
        if not resume_text.strip():
            st.warning("Please provide a resume first (upload, paste, or reuse your last one).")
        else:
            clean = preprocess_for_model(resume_text)
            category, confidence, proba_map = models.predict_category(clean)

            c1, c2 = st.columns([1, 1.6])
            with c1:
                st.metric("Predicted Category", category)
                st.metric("Confidence Score", f"{confidence*100:.1f}%")
                if models.metrics:
                    best_clf = models.metrics["classification"]["best_model"]
                    st.caption(f"Model used: **{best_clf}**")
            with c2:
                st.plotly_chart(category_probability_bar(proba_map), width='stretch')
