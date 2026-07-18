"""
pages/job_matching_page.py
-----------------------------
Resume <-> Job Description Matching page. Upload/paste a resume and a
job description; shows match percentage, matched keywords, missing
keywords, and recommended keywords to add.
"""

import streamlit as st

from utils.charts import match_gauge
from utils.job_matching import full_match_report
from utils.resume_parser import extract_resume_text


def render(models):
    st.markdown("## 🔗 Resume vs Job Description Matching")
    st.caption("TF-IDF + Cosine Similarity based matching between your resume and a target job description.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Resume")
        resume_source = st.radio("Resume input", ["Paste text", "Upload file", "Use my built resume"],
                                  horizontal=True, key="jm_resume_source")
        resume_text = ""
        if resume_source == "Paste text":
            resume_text = st.text_area("Resume text", height=250, key="jm_resume_text")
        elif resume_source == "Upload file":
            uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], key="jm_upload")
            if uploaded:
                try:
                    resume_text = extract_resume_text(uploaded)
                except Exception as e:
                    st.error(f"Failed to parse file: {e}")
        else:
            resume_text = st.session_state.get("last_resume_text", "")
            st.text_area("Resume text (from Resume Builder)", value=resume_text, height=250, disabled=True)

    with col2:
        st.markdown("#### Job Description")
        jd_text = st.text_area("Paste the job description", height=290, key="jm_jd_text",
                                placeholder="Paste the job description here...")

    if st.button("Compute Match", type="primary"):
        if not resume_text.strip() or not jd_text.strip():
            st.warning("Please provide both a resume and a job description.")
        else:
            report = full_match_report(resume_text, jd_text)

            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.plotly_chart(match_gauge(report["match_percentage"], title="Match %"),
                                 width='stretch')
                st.metric("Verdict", report["verdict"])
            with c2:
                st.markdown("#### ✅ Matched Keywords")
                if report["matched_keywords"]:
                    st.write(", ".join(report["matched_keywords"]))
                else:
                    st.write("No overlapping keywords detected.")

                st.markdown("#### ❌ Missing Keywords")
                if report["missing_keywords"]:
                    st.write(", ".join(report["missing_keywords"]))
                    st.caption("Consider adding these to your resume if you genuinely have this experience.")
                else:
                    st.write("None — your resume covers all detected job description keywords!")

                if report["additional_resume_keywords"]:
                    st.markdown("#### ➕ Additional Skills in Your Resume")
                    st.write(", ".join(report["additional_resume_keywords"]))

                if report["recommended_keywords"]:
                    st.markdown("#### 🌟 Recommended Phrases from the Job Description")
                    st.write(", ".join(report["recommended_keywords"]))
                    st.caption("Multi-word terms from the job posting (via NLP phrase extraction) "
                               "that aren't in your resume yet.")
