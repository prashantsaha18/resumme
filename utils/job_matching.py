"""
utils/job_matching.py
-----------------------
Resume <-> Job Description matching using TF-IDF + Cosine Similarity,
plus keyword-gap analysis (missing / recommended keywords).
"""

from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_processing import extract_key_phrases, extract_skills, preprocess_for_model


def compute_match_percentage(resume_text: str, job_description: str) -> float:
    """TF-IDF cosine similarity between resume and job description, as a percentage."""
    resume_clean = preprocess_for_model(resume_text)
    jd_clean = preprocess_for_model(job_description)

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    except ValueError:
        # Happens if vocabulary is empty after cleaning (e.g. all stopwords)
        return 0.0

    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(float(similarity) * 100, 2)


def keyword_gap_analysis(resume_text: str, job_description: str) -> Dict[str, List[str]]:
    """Compares recognized skills/keywords in the resume vs. the job description."""
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(job_description))

    missing = sorted(jd_skills - resume_skills)
    matched = sorted(resume_skills & jd_skills)
    extra = sorted(resume_skills - jd_skills)

    return {
        "matched_keywords": matched,
        "missing_keywords": missing,
        "additional_resume_keywords": extra,
    }


def recommended_keywords_from_jd(resume_text: str, job_description: str, top_n: int = 10) -> List[str]:
    """
    Beyond the fixed skills vocabulary, surfaces multi-word phrases from the
    job description (via spaCy noun-chunk extraction) that don't already
    appear in the resume text. Returns [] if spaCy's model isn't available.
    """
    jd_phrases = extract_key_phrases(job_description, top_n=30)
    resume_lower = resume_text.lower()
    recommended = [p for p in jd_phrases if p not in resume_lower]
    return recommended[:top_n]


def full_match_report(resume_text: str, job_description: str) -> Dict:
    match_pct = compute_match_percentage(resume_text, job_description)
    gaps = keyword_gap_analysis(resume_text, job_description)
    recommended_phrases = recommended_keywords_from_jd(resume_text, job_description)

    if match_pct >= 75:
        verdict = "Excellent Match"
    elif match_pct >= 50:
        verdict = "Good Match"
    elif match_pct >= 30:
        verdict = "Moderate Match"
    else:
        verdict = "Low Match"

    return {
        "match_percentage": match_pct,
        "verdict": verdict,
        "recommended_keywords": recommended_phrases,
        **gaps,
    }
