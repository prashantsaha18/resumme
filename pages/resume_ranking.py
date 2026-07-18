"""
pages/resume_ranking.py
--------------------------
Resume Ranking page: upload multiple resumes (optionally against a
target job description) and rank them by a composite score combining
ATS score, job match %, detected skill count, and years of experience.
"""

import pandas as pd
import streamlit as st

from utils.ats_analysis import analyze_resume
from utils.charts import ranking_bar_chart
from utils.job_matching import compute_match_percentage
from utils.resume_parser import extract_resume_text
from utils.text_processing import extract_years_of_experience, preprocess_for_model


def composite_score(ats_score: float, match_pct: float, n_skills: int, years: int) -> float:
    """
    Weighted composite: ATS score and job match dominate, skills and
    experience act as smaller boosts. Clamped to [0, 100].
    """
    skill_component = min(n_skills, 15) / 15 * 100
    exp_component = min(years, 10) / 10 * 100

    score = (
        0.35 * ats_score +
        0.35 * match_pct +
        0.20 * skill_component +
        0.10 * exp_component
    )
    return round(min(100, max(0, score)), 2)


def render(models):
    st.markdown("## 📊 Resume Ranking")
    st.caption("Upload multiple resumes and rank them by ATS score, job match, skills, and experience.")

    jd_text = st.text_area(
        "Optional: paste a job description to rank resumes against a specific role",
        height=120, placeholder="Paste a job description here (optional)...",
    )

    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, or TXT) — select multiple files",
        type=["pdf", "docx", "txt"], accept_multiple_files=True,
    )

    if st.button("Rank Resumes", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
        else:
            rows = []
            for f in uploaded_files:
                try:
                    text = extract_resume_text(f)
                except Exception as e:
                    st.error(f"Failed to parse {f.name}: {e}")
                    continue

                if not text.strip():
                    st.warning(f"No text extracted from {f.name}, skipping.")
                    continue

                clean = preprocess_for_model(text)
                years = extract_years_of_experience(text)
                category, confidence, _ = models.predict_category(clean)
                ats_score = models.predict_ats_score(clean, experience_years=years)
                analysis = analyze_resume(text, ats_score)
                match_pct = compute_match_percentage(text, jd_text) if jd_text.strip() else 0.0

                score = composite_score(
                    ats_score, match_pct, len(analysis["skills_found"]), years
                )

                rows.append({
                    "File": f.name,
                    "Predicted Category": category,
                    "Confidence": f"{confidence*100:.1f}%",
                    "ATS Score": round(ats_score, 1),
                    "Job Match %": round(match_pct, 1) if jd_text.strip() else "N/A",
                    "Skills Detected": len(analysis["skills_found"]),
                    "Experience (yrs)": years,
                    "Composite Score": score,
                })

            if rows:
                df = pd.DataFrame(rows).sort_values("Composite Score", ascending=False).reset_index(drop=True)
                df.index += 1
                st.markdown("### Ranking Table")
                st.dataframe(df, width='stretch')

                st.plotly_chart(
                    ranking_bar_chart(df["File"].tolist(), df["Composite Score"].tolist(),
                                       title="Composite Ranking Score"),
                    width='stretch',
                )

                st.caption(
                    "Composite Score = 35% ATS Score + 35% Job Match % (if a job description was "
                    "provided) + 20% Skills Coverage + 10% Experience."
                )

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Ranking as CSV", data=csv,
                                    file_name="resume_ranking.csv", mime="text/csv")
